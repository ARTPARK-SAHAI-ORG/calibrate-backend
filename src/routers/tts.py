import csv
import subprocess
import tempfile
import threading
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Path as PathParam
from pydantic import BaseModel, ConfigDict, Field

from db import (
    create_job,
    get_job,
    update_job,
)
from eval_common import (
    EVAL_JOB_TYPES,
    claim_eval_queue_slot,
    begin_eval_rerun,
    require_providers,
    apply_eval_job_timeout,
    load_eval_job_for_status,
    record_eval_task_crash,
    build_completed_provider_result,
    build_eval_status_response,
    load_eval_job_for_retry,
    normalize_and_enrich_provider_results,
    record_eval_failure,
    upload_leaderboard,
    upload_provider_dir,
    finalize_eval_results,
    run_calibrate_eval,
    write_evaluator_config,
    VisibilityRequest,
    VisibilityResponse,
    build_in_progress_provider_result,
    build_intermediate_provider_result,
    build_run_config_fallback,
    find_provider_output_dir,
    queued_provider_result,
    read_metrics_json,
    read_results_csv,
    resolve_evaluators_for_eval_job,
    set_eval_job_visibility,
)
from dataset_utils import (
    present_dataset_identity,
    resolve_dataset_inputs,
    resolve_eval_rerun_inputs_from_job_details,
)
from auth_utils import get_current_org, OrgContext
from utils import (
    TaskStatus,
    ProviderResult,
    TaskCreateResponse,
    TaskStatusResponse,
    presign_tts_provider_results_audio,
    get_s3_client,
    get_s3_output_config,
    try_start_queued_job,
    register_job_starter,
    generate_presigned_download_url,
    get_calibrate_agent_cli,
    read_evaluators_map_from_config,
    upload_file_to_s3,
    upload_top_level_files_to_s3,
)


def _start_tts_job_from_queue(job: dict) -> bool:
    """Start a TTS evaluation job from the queue.

    This is called by the job queue manager when there's capacity to run a new job.
    """
    job_id = job["uuid"]
    details = job.get("details", {})

    request = _tts_request_from_job_details(details)
    s3_bucket = details.get("s3_bucket", "")

    # Start background task in a separate thread
    thread = threading.Thread(
        target=run_tts_evaluation_task,
        args=(job_id, request, s3_bucket),
        daemon=True,
    )
    thread.start()

    return True


# Register the job starter for TTS evaluation jobs
register_job_starter("tts-eval", _start_tts_job_from_queue)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["tts"])

_EXAMPLE_ID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"


def _collect_tts_intermediate_results(
    output_dir: Path,
    providers: list,
    task_id: str,
    s3_bucket: str,
    expected_total: int,
) -> list:
    """Read whatever intermediate results are available from disk for each provider.

    Uploads audio files to S3 and replaces local paths with S3 keys.
    """
    evaluator_id_by_metric_key = read_evaluators_map_from_config(output_dir)
    s3 = get_s3_client()
    provider_results = []
    for provider in providers:
        provider_output_dir = find_provider_output_dir(output_dir, provider)
        results_data = read_results_csv(provider_output_dir)
        metrics_data = read_metrics_json(provider_output_dir)
        if results_data and provider_output_dir:
            audio_path_to_s3_key = upload_provider_dir(
                s3,
                provider_output_dir,
                s3_bucket,
                f"tts/evals/{task_id}/outputs/{provider}",
                swallow_errors=True,
            )
            _rewrite_audio_paths(results_data, audio_path_to_s3_key)
        else:
            results_data = None
        provider_results.append(
            build_intermediate_provider_result(
                provider,
                results_data,
                metrics_data,
                evaluator_id_by_metric_key,
                expected_total,
            )
        )
    return provider_results


def _presign_in_progress_audio(
    results_data: List[dict],
    s3,
    s3_bucket: str,
    provider_output_dir: Optional[Path],
    results_prefix: str,
) -> None:
    """Upload each row's freshly-synthesized clip and swap in a presigned URL.

    Rows whose clip is missing or fails to upload get `audio_path=None` rather
    than a local path the browser could never fetch.
    """
    for row in results_data:
        audio_path = row.get("audio_path")
        if not audio_path or audio_path.startswith("http"):
            continue
        local_path = Path(audio_path)
        if not (local_path.exists() and provider_output_dir):
            row["audio_path"] = None
            continue
        try:
            s3_key = f"{results_prefix}/{local_path.relative_to(provider_output_dir)}"
            upload_file_to_s3(s3, local_path, s3_bucket, s3_key)
            row["audio_s3_path"] = s3_key
            row["audio_path"] = generate_presigned_download_url(s3_key)
        except Exception:
            row["audio_path"] = None


def _rewrite_audio_paths(results_data: List[dict], audio_path_to_s3_key: dict) -> int:
    """Swap each row's local audio path for its S3 key. Returns how many were mapped."""
    mapped = 0
    for result_row in results_data:
        audio_s3_key = audio_path_to_s3_key.get(result_row.get("audio_path"))
        if audio_s3_key:
            result_row["audio_path"] = audio_s3_key
            mapped += 1
    return mapped


class TTSEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Existing TTS dataset to evaluate. **Provide this OR inline `texts`, not both**",
        examples=[_EXAMPLE_ID],
    )
    texts: Optional[List[str]] = Field(
        None,
        description="Texts to synthesize, one per item. **Required when `dataset_id` is omitted**",
    )
    dataset_name: Optional[str] = Field(
        None,
        description="Name for a new dataset created from the inline inputs. Ignored when `dataset_id` is set. Omit to not create one",
    )
    providers: List[str] = Field(
        description='TTS providers to compare, e.g. `["smallest", "cartesia", "openai"]`. At least one required'
    )
    language: str = Field(description='Language to synthesize in, e.g. `"english"` or `"hindi"`')
    evaluator_uuids: Optional[List[str]] = Field(
        None,
        description="Evaluators to score synthesized audio. Each must be a `tts` evaluator in your workspace. Omit to use the default TTS evaluator",
    )


def _tts_request_from_job_details(details: dict) -> TTSEvaluationRequest:
    return TTSEvaluationRequest(
        texts=details.get("texts", []),
        providers=details.get("providers", []),
        language=details.get("language", ""),
    )


def run_tts_evaluation_task(
    task_id: str,
    request: TTSEvaluationRequest,
    s3_bucket: str,
):
    """Run the TTS evaluation in the background."""
    try:
        logger.info(
            f"Running TTS evaluation task {task_id} with {len(request.providers)} providers"
        )
        update_job(task_id, status=TaskStatus.IN_PROGRESS.value)

        s3 = get_s3_client()

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Create input CSV file
                input_csv = temp_path / "input.csv"
                with open(input_csv, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["id", "text"])
                    for idx, text in enumerate(request.texts):
                        writer.writerow([idx, text])

                # Create output directory
                output_dir = temp_path / "output"
                output_dir.mkdir()

                # Run calibrate tts command with all providers at once
                # The CLI now handles parallelization internally and generates leaderboard
                eval_cmd = (
                    [
                        get_calibrate_agent_cli(),
                        "tts",
                        "-p",
                    ]
                    + request.providers
                    + [
                        "-l",
                        request.language,
                        "-i",
                        str(input_csv),
                        "-o",
                        str(output_dir),
                    ]
                )

                job_details = (get_job(task_id) or {}).get("details", {}) or {}
                config_path = write_evaluator_config(task_id, job_details, temp_path)
                if config_path:
                    eval_cmd.extend(["--config", str(config_path)])

                logger.info(f"Running TTS eval command: {' '.join(eval_cmd)}")
                run_calibrate_eval(eval_cmd, task_id, output_dir, temp_path, "TTS")

                # Read results for each provider
                provider_results = []
                evaluator_id_by_metric_key = read_evaluators_map_from_config(output_dir)
                for provider in request.providers:
                    provider_output_dir = find_provider_output_dir(
                        output_dir, provider
                    )
                    if provider_output_dir:
                        metrics_data = read_metrics_json(provider_output_dir)
                        results_data = read_results_csv(provider_output_dir)

                        audio_path_to_s3_key = upload_provider_dir(
                            s3,
                            provider_output_dir,
                            s3_bucket,
                            f"tts/evals/{task_id}/outputs/{provider}",
                        )
                        # A provider that synthesized nothing we could map back to
                        # S3 has no playable audio, so it counts as failed even
                        # though calibrate exited cleanly.
                        mapped_count = (
                            _rewrite_audio_paths(results_data, audio_path_to_s3_key)
                            if results_data
                            else 0
                        )

                        provider_results.append(
                            build_completed_provider_result(
                                provider,
                                results_data,
                                metrics_data,
                                evaluator_id_by_metric_key,
                                mapped_count > 0,
                            )
                        )
                    else:
                        provider_results.append(
                            ProviderResult(
                                provider=provider,
                                success=False,
                                message=f"No output found for provider {provider}",
                            )
                        )

                upload_top_level_files_to_s3(
                    s3,
                    output_dir,
                    s3_bucket,
                    f"tts/evals/{task_id}/outputs",
                )

                leaderboard_summary = upload_leaderboard(
                    s3, output_dir, s3_bucket, f"tts/evals/{task_id}/leaderboard"
                )

                config_file = build_run_config_fallback(
                    output_dir,
                    temp_path,
                    {
                        "providers": request.providers,
                        "language": request.language,
                        "text_count": len(request.texts),
                    },
                )
                config_s3_key = f"tts/evals/{task_id}/config.json"
                upload_file_to_s3(s3, config_file, s3_bucket, config_s3_key)
                logger.info(f"Uploaded config file to S3: {config_s3_key}")

                finalize_eval_results(
                    task_id, provider_results, leaderboard_summary
                )

            except Exception as e:
                if isinstance(e, subprocess.CalledProcessError):
                    message = f"TTS evaluation failed: {e.stderr if hasattr(e, 'stderr') else str(e)}"
                else:
                    message = f"Unexpected error during TTS evaluation: {str(e)}"
                record_eval_failure(
                    task_id,
                    e,
                    message,
                    s3,
                    output_dir,
                    s3_bucket,
                    f"tts/evals/{task_id}/outputs",
                    lambda: _collect_tts_intermediate_results(
                        output_dir,
                        request.providers,
                        task_id,
                        s3_bucket,
                        len(request.texts),
                    ),
                )

    except Exception as e:
        record_eval_task_crash(task_id, e)
    finally:
        # Try to start the next queued job
        try_start_queued_job(EVAL_JOB_TYPES)


@router.post("/evaluate", response_model=TaskCreateResponse, summary="Run TTS evaluation")
async def evaluate_tts(
    request: TTSEvaluationRequest, ctx: OrgContext = Depends(get_current_org)
):
    """Benchmark TTS providers against text inputs as a background job"""
    require_providers(request.providers)

    resolved = resolve_dataset_inputs(
        dataset_id=request.dataset_id,
        org_uuid=ctx.org_uuid,
        expected_type="tts",
        texts=request.texts,
        dataset_name=request.dataset_name,
    )
    texts = resolved.texts
    resolved_dataset_id = resolved.dataset_id
    resolved_dataset_name = resolved.dataset_name
    dataset_item_ids = resolved.item_ids

    request.texts = texts

    try:
        s3_bucket = get_s3_output_config()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    resolved_evaluators = resolve_evaluators_for_eval_job(
        uuids=request.evaluator_uuids,
        org_uuid=ctx.org_uuid,
        expected_evaluator_type="tts",
        default_slug="default-tts-audio-quality",
        default_data_type="audio",
    )

    can_start, initial_status = claim_eval_queue_slot(ctx.org_uuid)

    job_id = create_job(
        job_type="tts-eval",
        org_uuid=ctx.org_uuid,
        user_id=ctx.user_id,
        status=initial_status,
        details={
            "texts": texts,
            "providers": request.providers,
            "language": request.language,
            "s3_bucket": s3_bucket,
            "dataset_id": resolved_dataset_id,
            "dataset_name": resolved_dataset_name,
            "dataset_item_ids": dataset_item_ids,
            "evaluators": resolved_evaluators,
        },
        results=None,
    )

    if can_start:
        # Start background task in a separate thread
        thread = threading.Thread(
            target=run_tts_evaluation_task,
            args=(job_id, request, s3_bucket),
            daemon=True,
        )
        thread.start()
        logger.info(f"Started TTS evaluation job {job_id} immediately")
    else:
        logger.info(f"Queued TTS evaluation job {job_id}")

    return TaskCreateResponse(
        task_id=job_id,
        status=initial_status,
        dataset_id=resolved_dataset_id,
        dataset_name=resolved_dataset_name,
    )


@router.post(
    "/evaluate/{task_id}/retry",
    response_model=TaskCreateResponse,
    summary="Retry TTS evaluation",
)
async def retry_tts_evaluation(
    task_id: str = PathParam(
        description="The TTS evaluation to re-run",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Re-run the same TTS evaluation job with its stored providers and evaluators, re-reading the dataset when one is linked"""
    _job, details, providers = load_eval_job_for_retry(
        task_id, "tts-eval", ctx.org_uuid
    )

    try:
        s3_bucket = get_s3_output_config()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    resolved = resolve_eval_rerun_inputs_from_job_details(
        details,
        org_uuid=ctx.org_uuid,
        expected_type="tts",
    )

    rerun_details = {
        "texts": resolved.texts,
        "providers": providers,
        "language": details.get("language", ""),
        "s3_bucket": s3_bucket,
        "dataset_id": resolved.dataset_id,
        "dataset_name": resolved.dataset_name,
        "dataset_item_ids": resolved.item_ids,
        "evaluators": details.get("evaluators", []),
    }

    can_start, initial_status = begin_eval_rerun(
        task_id, ctx.org_uuid, rerun_details
    )

    request = _tts_request_from_job_details(rerun_details)
    if can_start:
        thread = threading.Thread(
            target=run_tts_evaluation_task,
            args=(task_id, request, s3_bucket),
            daemon=True,
        )
        thread.start()
        logger.info(f"Re-started TTS evaluation job {task_id}")
    else:
        logger.info(f"Re-queued TTS evaluation job {task_id}")

    return TaskCreateResponse(
        task_id=task_id,
        status=initial_status,
        dataset_id=rerun_details.get("dataset_id"),
        dataset_name=rerun_details.get("dataset_name"),
    )


@router.patch(
    "/evaluate/{task_id}/visibility",
    response_model=VisibilityResponse,
    summary="Update TTS evaluation visibility",
)
async def update_tts_visibility(
    body: VisibilityRequest,
    task_id: str = PathParam(
        description="The TTS evaluation to update",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Update public sharing for a TTS evaluation"""
    return set_eval_job_visibility(
        task_id, "tts-eval", ctx.org_uuid, body.is_public
    )


@router.get(
    "/evaluate/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get TTS evaluation status",
)
async def get_tts_evaluation_status(
    task_id: str = PathParam(
        description="The TTS evaluation to poll",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get the status and results of a TTS evaluation"""
    job, status, results, details = load_eval_job_for_status(task_id, ctx.org_uuid)

    status = apply_eval_job_timeout(
        task_id,
        job,
        details,
        results,
        status,
        lambda out_dir, providers: _collect_tts_intermediate_results(
            out_dir,
            providers,
            task_id,
            details.get("s3_bucket", ""),
            len(details.get("texts") or []),
        ),
    )

    # Get list of all requested providers from job details
    requested_providers = details.get("providers", [])

    # Build provider results
    provider_results = results.get("provider_results")
    output_dir_str = details.get("output_dir")
    if provider_results is None and status == TaskStatus.IN_PROGRESS.value:
        # Job is in progress - try to read intermediate results from disk
        expected_total = len(details.get("texts", []))
        if output_dir_str:
            output_dir = Path(output_dir_str)
            s3 = get_s3_client()
            s3_bucket = details.get("s3_bucket", "")
            provider_results = []
            for provider in requested_providers:
                provider_output_dir = find_provider_output_dir(
                    output_dir, provider
                )
                results_data = read_results_csv(provider_output_dir)
                metrics_data = read_metrics_json(provider_output_dir)
                if results_data:
                    _presign_in_progress_audio(
                        results_data,
                        s3,
                        s3_bucket,
                        provider_output_dir,
                        f"tts/evals/{task_id}/outputs/{provider}",
                    )
                provider_results.append(
                    build_in_progress_provider_result(
                        provider,
                        results_data,
                        metrics_data,
                        expected_total,
                        "texts",
                    )
                )

    if provider_results is None:
        # Job hasn't completed yet or no output dir available, show all as queued
        provider_results = [
            queued_provider_result(provider) for provider in requested_providers
        ]

    normalize_and_enrich_provider_results(provider_results, details)

    if provider_results:
        presign_tts_provider_results_audio(provider_results, status)

    dataset_id, dataset_name = present_dataset_identity(details, org_uuid=ctx.org_uuid)

    return build_eval_status_response(
        task_id, status, job, details, results, provider_results,
        dataset_id, dataset_name,
    )
