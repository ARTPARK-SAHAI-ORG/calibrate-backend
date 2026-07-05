import os
import csv
import json
import subprocess
import tempfile
import time
import traceback
import threading
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, ConfigDict

from db import (
    create_job,
    get_job,
    get_evaluator,
    get_evaluator_version,
    update_job,
    update_job_visibility,
)
from dataset_utils import resolve_dataset_inputs
from auth_utils import get_current_org, OrgContext
from llm_judge import build_evaluator_cli_payload
from utils import (
    TaskStatus,
    ProviderResult,
    TaskCreateResponse,
    TaskStatusResponse,
    build_evaluator_runs_for_eval_job,
    compute_share_token_toggle,
    enrich_evaluator_runs_with_current_names,
    load_evaluator_metric_key_map,
    post_process_provider_results,
    get_s3_client,
    get_s3_output_config,
    download_file_from_s3,
    can_start_job,
    try_start_queued_job,
    register_job_starter,
    is_job_timed_out,
    kill_process_group,
    capture_exception_to_sentry,
    get_calibrate_agent_cli,
    normalize_metrics,
    read_leaderboard_xlsx,
    read_evaluators_map_from_config,
    presign_audio_path,
    upload_directory_tree_to_s3,
    upload_file_to_s3,
    upload_top_level_files_to_s3,
)

# Job types that share the same queue
EVAL_JOB_TYPES = ["stt-eval", "tts-eval", "annotation-eval"]


def _is_job_aborted(task_id: str) -> bool:
    """Check if an STT eval job was aborted by the user."""
    job = get_job(task_id)
    return bool(job and (job.get("details") or {}).get("aborted"))


def _resolve_evaluators_for_job(
    uuids: Optional[List[str]],
    org_uuid: str,
    default_slug: str,
    expected_evaluator_type: str,
) -> List[dict]:
    """Resolve evaluator UUIDs into a list of fully-hydrated dicts ready to serialize
    into the calibrate CLI config.

    - Falls back to the default-slug evaluator when no UUIDs are provided.
    - Pins each evaluator to its current live version at submission time.
    - Enforces `evaluator.evaluator_type == expected_evaluator_type`. 400 on mismatch.
    """
    from db import get_evaluator_by_slug  # local import to avoid circular

    resolved: List[dict] = []
    effective_refs: List[dict] = [
        {"evaluator_uuid": uid, "version_uuid": None, "variable_values": None}
        for uid in (uuids or [])
    ]

    if not effective_refs:
        default = get_evaluator_by_slug(default_slug)
        if default and default.get("live_version_id"):
            effective_refs = [
                {
                    "evaluator_uuid": default["uuid"],
                    "version_uuid": default["live_version_id"],
                    "variable_values": None,
                }
            ]

    for ref in effective_refs:
        evaluator = get_evaluator(ref["evaluator_uuid"])
        if not evaluator:
            raise HTTPException(
                status_code=404, detail=f"Evaluator {ref['evaluator_uuid']} not found"
            )
        if (
            evaluator.get("org_uuid") is not None
            and evaluator["org_uuid"] != org_uuid
        ):
            raise HTTPException(
                status_code=404, detail=f"Evaluator {ref['evaluator_uuid']} not found"
            )
        if evaluator.get("evaluator_type") != expected_evaluator_type:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Evaluator {ref['evaluator_uuid']} has evaluator_type="
                    f"'{evaluator.get('evaluator_type')}' but this job requires "
                    f"'{expected_evaluator_type}' evaluators."
                ),
            )
        version_uuid = ref["version_uuid"] or evaluator.get("live_version_id")
        if not version_uuid:
            raise HTTPException(
                status_code=400,
                detail=f"Evaluator {ref['evaluator_uuid']} has no live version",
            )
        version = get_evaluator_version(version_uuid)
        if not version or version["evaluator_id"] != evaluator["uuid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Evaluator version {version_uuid} not found for evaluator {ref['evaluator_uuid']}",
            )
        resolved.append(
            {
                "uuid": evaluator["uuid"],
                "name": evaluator["name"],
                "evaluator_type": evaluator.get(
                    "evaluator_type", expected_evaluator_type
                ),
                "data_type": evaluator.get("data_type", "text"),
                "kind": evaluator.get("kind", "single"),
                "output_type": evaluator.get("output_type", "binary"),
                "evaluator_version_id": version["uuid"],
                "judge_model": version["judge_model"],
                "system_prompt": version["system_prompt"],
                "output_config": version.get("output_config"),
                "variables": version.get("variables"),
                "variable_values": ref.get("variable_values") or {},
            }
        )
    return resolved


def _start_stt_job_from_queue(job: dict) -> bool:
    """Start an STT evaluation job from the queue.

    This is called by the job queue manager when there's capacity to run a new job.
    """
    job_id = job["uuid"]
    details = job.get("details", {})

    # Reconstruct request from job details
    request = STTEvaluationRequest(
        audio_paths=details.get("audio_paths", []),
        texts=details.get("texts", []),
        providers=details.get("providers", []),
        language=details.get("language", ""),
    )
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


def _collect_intermediate_results(
    output_dir: Path, providers: list, expected_total: int
) -> list:
    """Read whatever intermediate results are available from disk for each provider.

    Returns a list of ProviderResult objects preserving any partial results.
    A provider is only marked ``success=True`` when it has BOTH a complete
    row count (``>= expected_total``) AND an aggregate ``metrics.json`` on
    disk — matching the contract used by the in-progress GET reader. Any
    weaker signal (some rows but no metrics, or fewer rows than expected)
    means calibrate crashed mid-run for that provider, so we surface the
    partial rows but mark ``success=False`` to avoid lying to the FE.
    """
    evaluator_id_by_metric_key = read_evaluators_map_from_config(output_dir)
    provider_results = []
    for provider in providers:
        provider_output_dir = _find_provider_output_dir(output_dir, provider)
        results_data = _read_results_csv(provider_output_dir)
        metrics_data = _read_metrics_json(provider_output_dir)
        runs = (
            build_evaluator_runs_for_eval_job(
                metrics_data, evaluator_id_by_metric_key
            )
            if metrics_data is not None
            else []
        )
        if results_data:
            provider_done = (
                metrics_data is not None
                and len(results_data) >= expected_total
            )
            provider_results.append(
                ProviderResult(
                    provider=provider,
                    success=True if provider_done else False,
                    metrics=metrics_data,
                    results=results_data,
                    evaluator_runs=runs or None,
                )
            )
        else:
            provider_results.append(
                ProviderResult(
                    provider=provider,
                    success=False,
                )
            )
    return provider_results


class STTEvaluationRequest(BaseModel):
    # Reject unknown fields so legacy frontends sending the dropped `evaluators` shape get
    # a loud 422 instead of silently running without their custom evaluators.
    model_config = ConfigDict(extra="forbid")

    # Option 1: reuse an existing dataset
    dataset_id: Optional[str] = None
    # Option 2: inline upload (legacy / new files)
    audio_paths: Optional[List[str]] = None  # S3 paths to audio files
    texts: Optional[List[str]] = None  # Ground truth text for each audio file
    # When providing inline data, name for the new dataset to save (ignored when dataset_id is set)
    dataset_name: Optional[str] = None
    providers: List[
        str
    ]  # List of STT providers (e.g., ["deepgram", "openai", "sarvam"])
    language: str  # Language (e.g., "english", "hindi")
    # Optional list of evaluator UUIDs to score this run. If omitted, the seeded STT default
    # evaluator is used. Each evaluator must have `evaluator_type == "stt"` (validated at
    # submission time). At submission, each UUID is hydrated against the evaluator's then-live
    # version and pinned into the job details so the run uses a stable rubric.
    evaluator_uuids: Optional[List[str]] = None


def _find_provider_output_dir(output_dir: Path, provider: str) -> Optional[Path]:
    """Find the provider-specific output directory."""
    if not output_dir.exists():
        return None
    for item in output_dir.iterdir():
        if item.is_dir() and provider in item.name.lower():
            return item
    return None


def _read_results_csv(provider_output_dir: Path) -> Optional[List[dict]]:
    """Read results.csv from provider output directory if it exists."""
    if not provider_output_dir:
        return None
    results_file = provider_output_dir / "results.csv"
    if not results_file.exists():
        return None
    try:
        results_data = []
        with open(results_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results_data.append(dict(row))
        return results_data
    except Exception:
        return None


def _read_metrics_json(provider_output_dir: Path) -> Optional[dict]:
    """Read metrics.json from provider output directory if it exists.

    Handles both new format (dict) and old format (list of dicts) for backward compatibility.
    """
    if not provider_output_dir:
        return None
    metrics_file = provider_output_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception:
        return None


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

                # Calibrate: --config <path> with {evaluators: [...]}
                job_details = (get_job(task_id) or {}).get("details", {}) or {}
                raw_evaluators = job_details.get("evaluators") or []
                if raw_evaluators:
                    evaluator_payload = build_evaluator_cli_payload(raw_evaluators)
                    config_path = input_dir / "config.json"
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {"evaluators": evaluator_payload}, f, ensure_ascii=False
                        )
                    eval_cmd.extend(["--config", str(config_path)])

                logger.info(f"Running STT eval command: {' '.join(eval_cmd)}")

                # Create temp files for stdout/stderr
                stdout_path = output_dir / "stdout.log"
                stderr_path = output_dir / "stderr.log"

                with (
                    open(stdout_path, "w") as stdout_f,
                    open(stderr_path, "w") as stderr_f,
                ):
                    process = subprocess.Popen(
                        eval_cmd,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        text=True,
                        start_new_session=True,
                        cwd=str(temp_path),
                    )

                    # Store PID and output dir for cleanup and intermediate results
                    update_job(
                        task_id,
                        details={
                            "pid": process.pid,
                            "pgid": process.pid,
                            "output_dir": str(output_dir),
                        },
                    )

                    # Poll for process completion with heartbeat to keep updated_at fresh
                    # This prevents the job from being marked as timed out during long runs
                    HEARTBEAT_INTERVAL = 2  # seconds
                    aborted = False
                    while process.poll() is None:
                        if _is_job_aborted(task_id):
                            logger.info(
                                f"STT eval {task_id} aborted, killing process group and stopping"
                            )
                            kill_process_group(process.pid, task_id)
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                try:
                                    process.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    pass
                            aborted = True
                            break
                        time.sleep(HEARTBEAT_INTERVAL)
                        if process.poll() is None:
                            # Process still running, send heartbeat to refresh updated_at
                            update_job(task_id)

                    # Re-check the abort flag after the loop exits: an abort
                    # request may arrive AFTER the subprocess has exited
                    # naturally (so we never set `aborted` via the inner
                    # branch) but BEFORE the worker reaches the final
                    # update_job below. Without this guard, the normal
                    # completion path overwrites the abort endpoint's
                    # status=done / error=user_aborted with success.
                    if aborted or _is_job_aborted(task_id):
                        logger.info(
                            f"STT eval {task_id} was aborted by user, skipping final processing"
                        )
                        return

                # Read stdout/stderr
                with open(stdout_path, "r") as f:
                    stdout = f.read()
                with open(stderr_path, "r") as f:
                    stderr = f.read()

                if process.returncode != 0:
                    logger.error(f"STT eval failed with code {process.returncode}")
                    logger.error(f"stderr: {stderr}")
                    raise subprocess.CalledProcessError(
                        process.returncode, eval_cmd, stdout, stderr
                    )

                logger.info("STT eval command completed successfully")

                # Read results for each provider
                provider_results = []
                evaluator_id_by_metric_key = read_evaluators_map_from_config(output_dir)
                for provider in request.providers:
                    provider_output_dir = _find_provider_output_dir(
                        output_dir, provider
                    )
                    if provider_output_dir:
                        metrics_data = _read_metrics_json(provider_output_dir)
                        results_data = _read_results_csv(provider_output_dir)

                        # Upload provider results to S3
                        results_prefix = f"stt/evals/{task_id}/outputs/{provider}"
                        for root, dirs, files in os.walk(provider_output_dir):
                            for file in files:
                                local_file_path = Path(root) / file
                                relative_path = local_file_path.relative_to(
                                    provider_output_dir
                                )
                                s3_key = f"{results_prefix}/{relative_path}"
                                upload_file_to_s3(
                                    s3, local_file_path, s3_bucket, s3_key
                                )

                        eruns = (
                            build_evaluator_runs_for_eval_job(
                                metrics_data,
                                evaluator_id_by_metric_key,
                            )
                            if metrics_data is not None
                            else []
                        )
                        provider_results.append(
                            ProviderResult(
                                provider=provider,
                                success=True,
                                metrics=metrics_data,
                                results=results_data,
                                evaluator_runs=eruns or None,
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

                # Read leaderboard from output directory
                leaderboard_dir = output_dir / "leaderboard"
                leaderboard_summary = None

                # Log what's in output_dir for debugging
                logger.info(
                    f"Output directory contents: {[f.name for f in output_dir.iterdir()]}"
                )

                if leaderboard_dir.exists():
                    logger.info(f"Leaderboard directory exists: {leaderboard_dir}")
                    leaderboard_summary = read_leaderboard_xlsx(leaderboard_dir)

                    # Upload leaderboard to S3
                    leaderboard_prefix = f"stt/evals/{task_id}/leaderboard"
                    for root, dirs, files in os.walk(leaderboard_dir):
                        for file in files:
                            local_file_path = Path(root) / file
                            relative_path = local_file_path.relative_to(leaderboard_dir)
                            s3_key = f"{leaderboard_prefix}/{relative_path}"
                            upload_file_to_s3(s3, local_file_path, s3_bucket, s3_key)
                else:
                    logger.warning(
                        f"Leaderboard directory does not exist: {leaderboard_dir}"
                    )

                # Prefer calibrate's run-root config.json because it contains evaluator IDs/maps.
                config_file = output_dir / "config.json"
                if not config_file.exists():
                    config_data = {
                        "providers": request.providers,
                        "language": request.language,
                        "audio_count": len(request.audio_paths),
                    }
                    config_file = temp_path / "config.json"
                    with open(config_file, "w", encoding="utf-8") as f:
                        json.dump(config_data, f, indent=2)
                config_s3_key = f"stt/evals/{task_id}/config.json"
                upload_file_to_s3(s3, config_file, s3_bucket, config_s3_key)
                logger.info(f"Uploaded config file to S3: {config_s3_key}")

                # Check if all providers succeeded
                all_succeeded = all(r.success for r in provider_results)
                final_status = (
                    TaskStatus.DONE.value if all_succeeded else TaskStatus.FAILED.value
                )

                error_msg = None
                if not all_succeeded:
                    failed = [r.provider for r in provider_results if not r.success]
                    error_msg = f"Some providers failed: {', '.join(failed)}"

                # Last-chance abort check: results parsing + S3 upload above
                # can take seconds-to-minutes, plenty of room for an abort
                # request to arrive after the post-loop guard.
                if _is_job_aborted(task_id):
                    logger.info(
                        f"STT eval {task_id} aborted before final update, skipping"
                    )
                    return

                # Update job with results
                update_job(
                    task_id,
                    status=final_status,
                    results={
                        "provider_results": [r.model_dump() for r in provider_results],
                        "leaderboard_summary": leaderboard_summary,
                        "error": error_msg,
                    },
                )

            except subprocess.CalledProcessError as e:
                if _is_job_aborted(task_id):
                    logger.info(
                        f"STT eval {task_id} aborted, skipping error update"
                    )
                    return
                traceback.print_exc()
                capture_exception_to_sentry(e)
                error_results = {
                    "error": f"STT evaluation failed: {e.stderr if hasattr(e, 'stderr') else str(e)}",
                }
                # Preserve any intermediate results already written to disk
                try:
                    if output_dir.exists():
                        intermediate = _collect_intermediate_results(
                            output_dir,
                            request.providers,
                            len(request.audio_paths),
                        )
                        if intermediate:
                            error_results["provider_results"] = [
                                r.model_dump() for r in intermediate
                            ]
                        if output_dir.exists():
                            upload_directory_tree_to_s3(
                                s3,
                                output_dir,
                                s3_bucket,
                                f"stt/evals/{task_id}/outputs",
                            )
                except Exception:
                    pass
                update_job(
                    task_id,
                    status=TaskStatus.FAILED.value,
                    results=error_results,
                )
            except Exception as e:
                if _is_job_aborted(task_id):
                    logger.info(
                        f"STT eval {task_id} aborted, skipping error update"
                    )
                    return
                traceback.print_exc()
                capture_exception_to_sentry(e)
                error_results = {
                    "error": f"Unexpected error during STT evaluation: {str(e)}",
                }
                # Preserve any intermediate results already written to disk
                try:
                    if output_dir.exists():
                        intermediate = _collect_intermediate_results(
                            output_dir,
                            request.providers,
                            len(request.audio_paths),
                        )
                        if intermediate:
                            error_results["provider_results"] = [
                                r.model_dump() for r in intermediate
                            ]
                        if output_dir.exists():
                            upload_directory_tree_to_s3(
                                s3,
                                output_dir,
                                s3_bucket,
                                f"stt/evals/{task_id}/outputs",
                            )
                except Exception:
                    pass
                update_job(
                    task_id,
                    status=TaskStatus.FAILED.value,
                    results=error_results,
                )

    except Exception as e:
        if _is_job_aborted(task_id):
            logger.info(
                f"STT eval {task_id} aborted, skipping error update"
            )
        else:
            traceback.print_exc()
            capture_exception_to_sentry(e)
            update_job(
                task_id,
                status=TaskStatus.FAILED.value,
                results={"error": f"Task failed: {str(e)}"},
            )
    finally:
        # Try to start the next queued job
        try_start_queued_job(EVAL_JOB_TYPES)


@router.post("/evaluate", response_model=TaskCreateResponse)
async def evaluate_stt(
    request: STTEvaluationRequest, ctx: OrgContext = Depends(get_current_org)
):
    """
    Start a background task to evaluate multiple STT providers with audio files from S3.

    Returns a task ID that can be used to poll for status and results.
    """
    if not request.providers:
        raise HTTPException(
            status_code=400,
            detail="At least one provider must be specified",
        )

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

    resolved_evaluators = _resolve_evaluators_for_job(
        uuids=request.evaluator_uuids,
        org_uuid=ctx.org_uuid,
        default_slug="default-stt-transcription",
        expected_evaluator_type="stt",
    )

    can_start = can_start_job(EVAL_JOB_TYPES, ctx.org_uuid)
    initial_status = (
        TaskStatus.IN_PROGRESS.value if can_start else TaskStatus.QUEUED.value
    )

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


class VisibilityRequest(BaseModel):
    is_public: bool


class VisibilityResponse(BaseModel):
    is_public: bool
    share_token: str | None = None


@router.post("/evaluate/{task_id}/abort", response_model=TaskStatusResponse)
async def abort_stt_evaluation(
    task_id: str, user_id: str = Depends(get_current_user_id)
):
    """Abort a running STT evaluation job, preserving partial provider results.

    Kills the running calibrate subprocess, collects whatever each provider
    managed to write to disk, and saves the partial results with status=done.
    """
    job = get_job(task_id, user_id=user_id)
    if not job or job.get("type") != "stt-eval":
        raise HTTPException(status_code=404, detail="Task not found")

    if job.get("status") != TaskStatus.IN_PROGRESS.value:
        raise HTTPException(
            status_code=400, detail="Can only abort in-progress jobs"
        )

    details = job.get("details") or {}

    # Mark aborted FIRST so the polling loop's guard fires immediately.
    update_job(task_id, details={"aborted": True})

    # Kill running process group
    pid = details.get("pid") or details.get("pgid")
    if pid:
        kill_process_group(pid, task_id)

    # Collect partial results from disk (best effort) and upload partials.
    requested_providers = details.get("providers", [])
    output_dir_str = details.get("output_dir")
    intermediate_map: Dict[str, Dict[str, Any]] = {}
    if output_dir_str:
        try:
            output_dir = Path(output_dir_str)
            if output_dir.exists():
                intermediate = _collect_intermediate_results(
                    output_dir,
                    requested_providers,
                    len(details.get("audio_paths") or []),
                )
                intermediate_map = {
                    r.provider: r.model_dump() for r in intermediate
                }
                try:
                    s3 = get_s3_client()
                    s3_bucket = details.get("s3_bucket") or get_s3_output_config()
                    upload_directory_tree_to_s3(
                        s3,
                        output_dir,
                        s3_bucket,
                        f"stt/evals/{task_id}/outputs",
                    )
                except Exception as exc:
                    logger.warning(
                        f"Failed to upload partial outputs for aborted STT job {task_id}: {exc}"
                    )
        except Exception as exc:
            logger.warning(
                f"Failed to collect intermediate results on abort: {exc}"
            )

    # Re-read results from the DB right before the final write. The
    # worker may have committed a fresher `provider_results` snapshot
    # between our initial read and now (kill_process_group sends SIGTERM;
    # the subprocess takes a moment to die, and the polling thread can
    # land one more write in that window). Using the stale snapshot
    # would overwrite those newer partials.
    fresh_job = get_job(task_id, user_id=user_id) or {}
    results = dict(fresh_job.get("results") or {})
    existing_provider_results = results.get("provider_results", []) or []
    existing_success_map = {
        pr.get("provider"): pr
        for pr in existing_provider_results
        if pr.get("success") is True
    }

    merged_results: List[Dict[str, Any]] = []
    if intermediate_map or existing_provider_results:
        for provider in requested_providers:
            if provider in existing_success_map:
                merged_results.append(existing_success_map[provider])
            elif provider in intermediate_map:
                merged_results.append(intermediate_map[provider])
            else:
                merged_results.append(
                    {
                        "provider": provider,
                        "success": False,
                        "metrics": None,
                        "results": None,
                    }
                )

    if not merged_results and existing_provider_results:
        merged_results = existing_provider_results

    results["provider_results"] = merged_results
    results["error"] = "user_aborted"

    update_job(
        task_id,
        status=TaskStatus.DONE.value,
        results=results,
    )

    # Drain queue
    try_start_queued_job(EVAL_JOB_TYPES)

    # Re-read for fresh state
    updated = get_job(task_id, user_id=user_id) or {}
    updated_results = updated.get("results") or {}
    return TaskStatusResponse(
        task_id=task_id,
        status=TaskStatus.DONE.value,
        language=details.get("language"),
        dataset_id=details.get("dataset_id"),
        dataset_name=details.get("dataset_name"),
        provider_results=updated_results.get("provider_results"),
        leaderboard_summary=updated_results.get("leaderboard_summary"),
        error=updated_results.get("error"),
        is_public=bool(updated.get("is_public")),
        share_token=updated.get("share_token"),
    )


@router.patch("/evaluate/{task_id}/visibility", response_model=VisibilityResponse)
async def update_stt_visibility(
    task_id: str,
    body: VisibilityRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Toggle public sharing for an STT evaluation job."""
    job = get_job(task_id, org_uuid=ctx.org_uuid)
    if not job or job.get("type") != "stt-eval":
        raise HTTPException(status_code=404, detail="Task not found")

    token_to_persist, token_to_return = compute_share_token_toggle(
        job, body.is_public
    )
    update_job_visibility(task_id, body.is_public, token_to_persist)
    return VisibilityResponse(
        is_public=body.is_public, share_token=token_to_return
    )


@router.get("/evaluate/{task_id}", response_model=TaskStatusResponse)
async def get_evaluation_status(
    task_id: str, ctx: OrgContext = Depends(get_current_org)
):
    """
    Get the status of an STT evaluation task.

    Returns the current status and, if done, the provider results and leaderboard path.
    """
    job = get_job(task_id, org_uuid=ctx.org_uuid)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")

    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    # Check for timeout on in-progress jobs
    if status == TaskStatus.IN_PROGRESS.value:
        updated_at = job.get("updated_at")
        if updated_at and is_job_timed_out(updated_at):
            logger.warning(f"Job {task_id} timed out, marking as failed")

            # Kill running process
            pid = details.get("pid") or details.get("pgid")
            if pid:
                kill_process_group(pid, task_id)

            # Preserve intermediate results from disk before marking as failed
            # IMPORTANT: Merge with existing results, don't overwrite successful ones
            requested_providers = details.get("providers", [])
            output_dir_str = details.get("output_dir")
            existing_provider_results = results.get("provider_results", [])

            # Build a map of existing successful results (don't overwrite these)
            existing_success_map = {}
            for pr in existing_provider_results:
                if pr.get("success") is True:
                    existing_success_map[pr.get("provider")] = pr

            if output_dir_str:
                try:
                    output_dir = Path(output_dir_str)
                    if output_dir.exists():
                        intermediate = _collect_intermediate_results(
                            output_dir,
                            requested_providers,
                            len(details.get("audio_paths") or []),
                        )
                        # Merge: keep existing successful results, add new ones from disk
                        merged_results = []
                        intermediate_map = {
                            r.provider: r.model_dump() for r in intermediate
                        }
                        for provider in requested_providers:
                            if provider in existing_success_map:
                                # Keep the existing successful result
                                merged_results.append(existing_success_map[provider])
                            elif provider in intermediate_map:
                                # Use intermediate result from disk
                                merged_results.append(intermediate_map[provider])
                            else:
                                # Provider not found anywhere, mark as failed
                                merged_results.append(
                                    {
                                        "provider": provider,
                                        "success": False,
                                        "metrics": None,
                                        "results": None,
                                    }
                                )
                        results["provider_results"] = merged_results
                except Exception as exc:
                    logger.warning(
                        f"Failed to collect intermediate results on timeout: {exc}"
                    )
                    # Even on exception, preserve any existing successful results
                    if existing_provider_results:
                        results["provider_results"] = existing_provider_results

            # Mark job as failed
            results["error"] = "Job timed out after 5 minutes of inactivity"
            update_job(
                task_id,
                status=TaskStatus.FAILED.value,
                results=results,
            )
            status = TaskStatus.FAILED.value

            # Try to start the next queued job
            try_start_queued_job(EVAL_JOB_TYPES)

    # Get list of all requested providers from job details
    requested_providers = details.get("providers", [])

    # Build provider results
    provider_results = results.get("provider_results")
    # `evaluator_id_by_metric_key` lets the post-processor build
    # `evaluator_runs[]` mid-flight from a freshly-landed metrics.json.
    evaluator_id_by_metric_key = load_evaluator_metric_key_map(details)
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
                provider_output_dir = _find_provider_output_dir(output_dir, provider)
                results_data = _read_results_csv(provider_output_dir)
                metrics_data = _read_metrics_json(provider_output_dir)
                if results_data:
                    # If all files are processed and metrics are ready, mark as done
                    provider_done = (
                        len(results_data) >= expected_total and metrics_data is not None
                    )
                    provider_results.append(
                        {
                            "provider": provider,
                            "success": True if provider_done else None,
                            "message": (
                                f"Done ({len(results_data)} files processed)"
                                if provider_done
                                else f"Running... ({len(results_data)} files processed)"
                            ),
                            "metrics": metrics_data,
                            "results": results_data,
                        }
                    )
                else:
                    provider_results.append(
                        {
                            "provider": provider,
                            "success": None,
                            "message": "Queued...",
                            "metrics": None,
                            "results": None,
                        }
                    )

    if provider_results is None:
        # Job hasn't completed yet or no output dir available, show all as queued
        provider_results = [
            {
                "provider": provider,
                "success": None,
                "message": "Queued...",
                "metrics": None,
                "results": None,
            }
            for provider in requested_providers
        ]

    # Normalize metrics format for backward compatibility (list -> dict)
    for provider_result in provider_results:
        if provider_result.get("metrics"):
            provider_result["metrics"] = normalize_metrics(provider_result["metrics"])

    enrich_evaluator_runs_with_current_names(
        provider_results, details.get("evaluators") or []
    )

    # Canonical post-processing: lift per-row outputs into evaluator_outputs[uuid],
    # type-coerce values, build evaluator_runs from in-progress metrics, surface
    # per-row judge errors. Idempotent — safe across in-progress / done / failed.
    post_process_provider_results(
        provider_results,
        evaluator_snapshots=details.get("evaluators") or [],
        evaluator_id_by_metric_key=evaluator_id_by_metric_key,
    )

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

    return TaskStatusResponse(
        task_id=task_id,
        status=status,
        language=details.get("language"),
        dataset_id=details.get("dataset_id"),
        dataset_name=details.get("dataset_name"),
        provider_results=provider_results,
        leaderboard_summary=results.get("leaderboard_summary"),
        error=results.get("error"),
        is_public=bool(job.get("is_public")),
        share_token=job.get("share_token"),
    )
