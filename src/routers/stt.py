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
    get_s3_client,
    get_s3_output_config,
    download_file_from_s3,
    try_start_queued_job,
    register_job_starter,
    get_calibrate_agent_cli,
    read_evaluators_map_from_config,
    presign_audio_path,
    upload_file_to_s3,
    upload_top_level_files_to_s3,
)


def _start_stt_job_from_queue(job: dict) -> bool:
    """Start an STT evaluation job from the queue.

    This is called by the job queue manager when there's capacity to run a new job.
    """
    job_id = job["uuid"]
    details = job.get("details", {})

    request = _stt_request_from_job_details(details)
    s3_bucket = details.get("s3_bucket", "")

    # Start background task in a separate thread
    thread = threading.Thread(
        target=run_evaluation_task,
        args=(job_id, request, s3_bucket),
        daemon=True,
    )
    thread.start()

    return True


# Register the job starter for STT evaluation jobs
register_job_starter("stt-eval", _start_stt_job_from_queue)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stt", tags=["stt"])

_EXAMPLE_ID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"


def _collect_intermediate_results(
    output_dir: Path, providers: list, expected_total: int
) -> list:
    """Read whatever intermediate results are available from disk for each provider."""
    evaluator_id_by_metric_key = read_evaluators_map_from_config(output_dir)
    provider_results = []
    for provider in providers:
        provider_output_dir = find_provider_output_dir(output_dir, provider)
        provider_results.append(
            build_intermediate_provider_result(
                provider,
                read_results_csv(provider_output_dir),
                read_metrics_json(provider_output_dir),
                evaluator_id_by_metric_key,
                expected_total,
            )
        )
    return provider_results


class STTEvaluationRequest(BaseModel):
    # Reject unknown fields so legacy frontends sending the dropped `evaluators` shape get
    # a loud 422 instead of silently running without their custom evaluators.
    model_config = ConfigDict(extra="forbid")

    dataset_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Existing STT dataset to evaluate. **Provide this OR inline `audio_paths` + `texts`, not both**",
        examples=[_EXAMPLE_ID],
    )
    audio_paths: Optional[List[str]] = Field(
        None,
        description="Audio files to transcribe, one `s3://bucket/key` URI per item. **Required when `dataset_id` is omitted**. Must align 1:1 with `texts`",
    )
    texts: Optional[List[str]] = Field(
        None,
        description="Ground-truth transcripts to score against, one per audio file. **Required when `dataset_id` is omitted**. Must align 1:1 with `audio_paths`",
    )
    dataset_name: Optional[str] = Field(
        None,
        description="Name for a new dataset created from the inline inputs. Ignored when `dataset_id` is set. Omit to not create one",
    )
    providers: List[str] = Field(
        description='STT providers to compare, e.g. `["deepgram", "openai", "sarvam"]`. At least one required'
    )
    language: str = Field(description='Spoken language for the audio, e.g. `"english"` or `"hindi"`')
    evaluator_uuids: Optional[List[str]] = Field(
        None,
        description="Evaluators to score transcriptions. Each must be an `stt` evaluator in your workspace. Omit to run transcription metrics only, with no LLM judge",
    )
    sarvam_judges: bool = Field(
        True,
        description="Run the Sarvam LLM judge bundle alongside the always-computed WER and CER: intent, entity, and forgiving LLM-WER and LLM-CER scores. Adds an extra judge call for each transcribed row",
    )


def _stt_request_from_job_details(details: dict) -> STTEvaluationRequest:
    return STTEvaluationRequest(
        audio_paths=details.get("audio_paths", []),
        texts=details.get("texts", []),
        providers=details.get("providers", []),
        language=details.get("language", ""),
        sarvam_judges=details.get("sarvam_judges", True),
    )


def run_evaluation_task(
    task_id: str,
    request: STTEvaluationRequest,
    s3_bucket: str,
):
    """Run the STT evaluation in the background."""
    try:
        logger.info(
            f"Running evaluation task {task_id} with {len(request.providers)} providers"
        )
        update_job(task_id, status=TaskStatus.IN_PROGRESS.value)

        s3 = get_s3_client()

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Create directory structure
                input_dir = temp_path / "input"
                input_dir.mkdir()
                audios_dir = input_dir / "audios"
                audios_dir.mkdir(parents=True)

                # Download audio files from S3 and create CSV
                stt_csv_path = input_dir / "stt.csv"
                with open(stt_csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["id", "text"])

                    for idx, (audio_path, gt_text) in enumerate(
                        zip(request.audio_paths, request.texts)
                    ):
                        if not audio_path:
                            raise ValueError(
                                f"STT item at index {idx} has no audio_path"
                            )
                        # Parse S3 path (format: s3://bucket/key or bucket/key)
                        if audio_path.startswith("s3://"):
                            parts = audio_path[5:].split("/", 1)
                            bucket = parts[0]
                            key = parts[1] if len(parts) > 1 else ""
                        else:
                            parts = audio_path.split("/", 1)
                            bucket = parts[0]
                            key = parts[1] if len(parts) > 1 else ""

                        # Generate audio ID
                        audio_id = f"audio_{idx + 1}"

                        # Download audio file directly to audios folder
                        local_audio_path = audios_dir / f"{audio_id}.wav"

                        logger.info(
                            f"Downloading audio file from {bucket}/{key} to {local_audio_path}"
                        )
                        download_file_from_s3(s3, bucket, key, local_audio_path)

                        # Write CSV row
                        writer.writerow([audio_id, gt_text])

                # Create output directory
                output_dir = temp_path / "output"
                output_dir.mkdir()

                # Run calibrate stt command with all providers at once
                # The CLI now handles parallelization internally and generates leaderboard
                eval_cmd = (
                    [
                        get_calibrate_agent_cli(),
                        "stt",
                        "-p",
                    ]
                    + request.providers
                    + [
                        "-l",
                        request.language,
                        "-i",
                        str(input_dir),
                        "-o",
                        str(output_dir),
                    ]
                )

                job_details = (get_job(task_id) or {}).get("details", {}) or {}

                # Sarvam judge bundle is a metrics-axis toggle independent of the
                # evaluator list. The CLI includes it by default; pass
                # --skip-llm-judges only to opt out. Snapshotted into details at
                # submit time so a queued/retried run remembers its mode.
                if not job_details.get("sarvam_judges", True):
                    eval_cmd.append("--skip-llm-judges")

                config_path = write_evaluator_config(task_id, job_details, input_dir)
                if config_path:
                    eval_cmd.extend(["--config", str(config_path)])

                logger.info(f"Running STT eval command: {' '.join(eval_cmd)}")
                run_calibrate_eval(eval_cmd, task_id, output_dir, temp_path, "STT")

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

                        upload_provider_dir(
                            s3,
                            provider_output_dir,
                            s3_bucket,
                            f"stt/evals/{task_id}/outputs/{provider}",
                        )

                        provider_results.append(
                            build_completed_provider_result(
                                provider,
                                results_data,
                                metrics_data,
                                evaluator_id_by_metric_key,
                                True,
                            )
                        )
                    else:
                        provider_results.append(
                            ProviderResult(
                                provider=provider,
                                success=False,
                            )
                        )

                # Run-level artifacts (whole-run ``logs``, ``leaderboard.csv``, backend stdout/stderr)
                upload_top_level_files_to_s3(
                    s3,
                    output_dir,
                    s3_bucket,
                    f"stt/evals/{task_id}/outputs",
                )

                leaderboard_summary = upload_leaderboard(
                    s3, output_dir, s3_bucket, f"stt/evals/{task_id}/leaderboard"
                )

                config_file = build_run_config_fallback(
                    output_dir,
                    temp_path,
                    {
                        "providers": request.providers,
                        "language": request.language,
                        "audio_count": len(request.audio_paths),
                    },
                )
                config_s3_key = f"stt/evals/{task_id}/config.json"
                upload_file_to_s3(s3, config_file, s3_bucket, config_s3_key)
                logger.info(f"Uploaded config file to S3: {config_s3_key}")

                finalize_eval_results(
                    task_id, provider_results, leaderboard_summary
                )

            except Exception as e:
                if isinstance(e, subprocess.CalledProcessError):
                    message = f"STT evaluation failed: {e.stderr if hasattr(e, 'stderr') else str(e)}"
                else:
                    message = f"Unexpected error during STT evaluation: {str(e)}"
                record_eval_failure(
                    task_id,
                    e,
                    message,
                    s3,
                    output_dir,
                    s3_bucket,
                    f"stt/evals/{task_id}/outputs",
                    lambda: _collect_intermediate_results(
                        output_dir, request.providers, len(request.audio_paths)
                    ),
                )

    except Exception as e:
        record_eval_task_crash(task_id, e)
    finally:
        # Try to start the next queued job
        try_start_queued_job(EVAL_JOB_TYPES)


@router.post("/evaluate", response_model=TaskCreateResponse, summary="Run STT evaluation")
async def evaluate_stt(
    request: STTEvaluationRequest, ctx: OrgContext = Depends(get_current_org)
):
    """Benchmark STT providers against a dataset as a background job"""
    require_providers(request.providers)

    resolved = resolve_dataset_inputs(
        dataset_id=request.dataset_id,
        org_uuid=ctx.org_uuid,
        expected_type="stt",
        texts=request.texts,
        audio_paths=request.audio_paths,
        dataset_name=request.dataset_name,
    )
    audio_paths = resolved.audio_paths
    texts = resolved.texts
    resolved_dataset_id = resolved.dataset_id
    resolved_dataset_name = resolved.dataset_name
    dataset_item_ids = resolved.item_ids

    request.audio_paths = audio_paths
    request.texts = texts

    try:
        s3_bucket = get_s3_output_config()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    resolved_evaluators = resolve_evaluators_for_eval_job(
        uuids=request.evaluator_uuids,
        org_uuid=ctx.org_uuid,
        expected_evaluator_type="stt",
    )

    can_start, initial_status = claim_eval_queue_slot(ctx.org_uuid)

    job_id = create_job(
        job_type="stt-eval",
        org_uuid=ctx.org_uuid,
        user_id=ctx.user_id,
        status=initial_status,
        details={
            "audio_paths": audio_paths,
            "texts": texts,
            "providers": request.providers,
            "language": request.language,
            "s3_bucket": s3_bucket,
            "dataset_id": resolved_dataset_id,
            "dataset_name": resolved_dataset_name,
            "dataset_item_ids": dataset_item_ids,
            "evaluators": resolved_evaluators,
            "sarvam_judges": request.sarvam_judges,
        },
        results=None,
    )

    if can_start:
        # Start background task in a separate thread
        thread = threading.Thread(
            target=run_evaluation_task,
            args=(job_id, request, s3_bucket),
            daemon=True,
        )
        thread.start()
        logger.info(f"Started STT evaluation job {job_id} immediately")
    else:
        logger.info(f"Queued STT evaluation job {job_id}")

    return TaskCreateResponse(
        task_id=job_id,
        status=initial_status,
        dataset_id=resolved_dataset_id,
        dataset_name=resolved_dataset_name,
    )


@router.post(
    "/evaluate/{task_id}/retry",
    response_model=TaskCreateResponse,
    summary="Retry STT evaluation",
)
async def retry_stt_evaluation(
    task_id: str = PathParam(
        description="The STT evaluation to re-run",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Re-run the same STT evaluation job with its stored providers and evaluators, re-reading the dataset when one is linked"""
    _job, details, providers = load_eval_job_for_retry(
        task_id, "stt-eval", ctx.org_uuid
    )

    try:
        s3_bucket = get_s3_output_config()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    resolved = resolve_eval_rerun_inputs_from_job_details(
        details,
        org_uuid=ctx.org_uuid,
        expected_type="stt",
    )

    rerun_details = {
        "audio_paths": resolved.audio_paths or [],
        "texts": resolved.texts,
        "providers": providers,
        "language": details.get("language", ""),
        "s3_bucket": s3_bucket,
        "dataset_id": resolved.dataset_id,
        "dataset_name": resolved.dataset_name,
        "dataset_item_ids": resolved.item_ids,
        "evaluators": details.get("evaluators", []),
        "sarvam_judges": details.get("sarvam_judges", True),
    }

    can_start, initial_status = begin_eval_rerun(
        task_id, ctx.org_uuid, rerun_details
    )

    request = _stt_request_from_job_details(rerun_details)
    if can_start:
        thread = threading.Thread(
            target=run_evaluation_task,
            args=(task_id, request, s3_bucket),
            daemon=True,
        )
        thread.start()
        logger.info(f"Re-started STT evaluation job {task_id}")
    else:
        logger.info(f"Re-queued STT evaluation job {task_id}")

    return TaskCreateResponse(
        task_id=task_id,
        status=initial_status,
        dataset_id=rerun_details.get("dataset_id"),
        dataset_name=rerun_details.get("dataset_name"),
    )


@router.patch(
    "/evaluate/{task_id}/visibility",
    response_model=VisibilityResponse,
    summary="Update STT evaluation visibility",
)
async def update_stt_visibility(
    body: VisibilityRequest,
    task_id: str = PathParam(
        description="The STT evaluation to update",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Update public sharing for an STT evaluation"""
    return set_eval_job_visibility(
        task_id, "stt-eval", ctx.org_uuid, body.is_public
    )


@router.get(
    "/evaluate/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get STT evaluation status",
)
async def get_evaluation_status(
    task_id: str = PathParam(
        description="The STT evaluation to poll",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get the status and results of an STT evaluation"""
    job, status, results, details = load_eval_job_for_status(task_id, ctx.org_uuid)

    status = apply_eval_job_timeout(
        task_id,
        job,
        details,
        results,
        status,
        lambda out_dir, providers: _collect_intermediate_results(
            out_dir, providers, len(details.get("audio_paths") or [])
        ),
    )

    # Get list of all requested providers from job details
    requested_providers = details.get("providers", [])

    # Build provider results
    provider_results = results.get("provider_results")
    output_dir_str = details.get("output_dir")
    output_dir_root = Path(output_dir_str) if (
        output_dir_str and Path(output_dir_str).exists()
    ) else None
    if provider_results is None and status == TaskStatus.IN_PROGRESS.value:
        # Job is in progress - try to read intermediate results from disk
        expected_total = len(details.get("audio_paths", []))
        if output_dir_root:
            output_dir = output_dir_root
            provider_results = []
            for provider in requested_providers:
                provider_output_dir = find_provider_output_dir(output_dir, provider)
                provider_results.append(
                    build_in_progress_provider_result(
                        provider,
                        read_results_csv(provider_output_dir),
                        read_metrics_json(provider_output_dir),
                        expected_total,
                        "files",
                    )
                )

    if provider_results is None:
        # Job hasn't completed yet or no output dir available, show all as queued
        provider_results = [
            queued_provider_result(provider) for provider in requested_providers
        ]

    normalize_and_enrich_provider_results(provider_results, details)

    # Enrich each result row with a presigned audio URL from the dataset.
    # Only presign IDs that actually appear in results to avoid unnecessary
    # S3 calls during early polling when results are still empty.
    audio_paths = details.get("audio_paths", [])
    if audio_paths:
        # Collect IDs actually present in results
        needed_ids: set[str] = set()
        for provider_result in provider_results:
            for row in provider_result.get("results") or []:
                if row.get("id"):
                    needed_ids.add(row["id"])

        if needed_ids:
            audio_url_map = {}
            for idx, path in enumerate(audio_paths):
                audio_id = f"audio_{idx + 1}"
                if audio_id in needed_ids:
                    audio_url_map[audio_id] = presign_audio_path(path)

            for provider_result in provider_results:
                for row in provider_result.get("results") or []:
                    row["audio_url"] = audio_url_map.get(row.get("id", ""))

    dataset_id, dataset_name = present_dataset_identity(details, org_uuid=ctx.org_uuid)

    return build_eval_status_response(
        task_id, status, job, details, results, provider_results,
        dataset_id, dataset_name,
    )
