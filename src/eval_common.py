"""Pieces shared by the STT and TTS evaluation routers.

Both flows run the same shape of job: resolve evaluators, spawn one calibrate
subprocess for all providers, then read `results.csv` / `metrics.json` back per
provider. What lives here is everything that differs only by media type.

Symbols the routers' tests patch by module path (`routers.stt.upload_file_to_s3`,
`routers.stt.time.sleep`, ...) deliberately stay in the routers, so extraction
stops at the first line that touches one.
"""

import csv
import json
import logging
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Callable, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

from db import (
    get_evaluator,
    get_evaluator_by_slug,
    get_evaluator_version,
    get_job,
    update_job,
    update_job_visibility,
)
from llm_judge import build_evaluator_cli_payload, refresh_evaluators_to_live
from utils import (
    ProviderResult,
    TaskStatus,
    TaskStatusResponse,
    build_evaluator_runs_for_eval_job,
    can_start_job,
    capture_exception_to_sentry,
    compute_share_token_toggle,
    enrich_evaluator_runs_with_current_names,
    load_evaluator_metric_key_map,
    normalize_metrics,
    post_process_provider_results,
    is_job_timed_out,
    kill_process_group,
    read_leaderboard_xlsx,
    try_start_queued_job,
    upload_directory_tree_to_s3,
    upload_file_to_s3,
)

logger = logging.getLogger(__name__)

# Job types that share the same queue
EVAL_JOB_TYPES = ["stt-eval", "tts-eval", "annotation-eval"]

_HEARTBEAT_INTERVAL = 2  # seconds


def resolve_evaluators_for_eval_job(
    uuids: Optional[List[str]],
    org_uuid: str,
    expected_evaluator_type: str,
    default_slug: Optional[str] = None,
    default_data_type: str = "text",
) -> List[dict]:
    """Resolve evaluator UUIDs into fully-hydrated dicts ready to serialize into
    the calibrate CLI config.

    - Returns an empty list when no UUIDs are given and no `default_slug` is set:
      the run then skips the LLM judge entirely.
    - Pins each evaluator to its current live version at submission time.
    - Enforces `evaluator.evaluator_type == expected_evaluator_type`. 400 on mismatch.
    """
    resolved: List[dict] = []
    effective_refs: List[dict] = [
        {"evaluator_uuid": uid, "version_uuid": None, "variable_values": None}
        for uid in (uuids or [])
    ]

    if not effective_refs and default_slug:
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
        if evaluator.get("org_uuid") is not None and evaluator["org_uuid"] != org_uuid:
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
                "data_type": evaluator.get("data_type", default_data_type),
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


def find_provider_output_dir(output_dir: Path, provider: str) -> Optional[Path]:
    """Find the provider-specific output directory."""
    if not output_dir.exists():
        return None
    for item in output_dir.iterdir():
        if item.is_dir() and provider in item.name.lower():
            return item
    return None


def read_results_csv(provider_output_dir: Path) -> Optional[List[dict]]:
    """Read results.csv from provider output directory if it exists."""
    if not provider_output_dir:
        return None
    results_file = provider_output_dir / "results.csv"
    if not results_file.exists():
        return None
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except Exception:
        return None


def read_metrics_json(provider_output_dir: Path) -> Optional[dict]:
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
            return json.load(f)
    except Exception:
        return None


def build_intermediate_provider_result(
    provider: str,
    results_data: Optional[List[dict]],
    metrics_data: Optional[dict],
    evaluator_id_by_metric_key: dict,
    expected_total: int,
) -> ProviderResult:
    """Shape one provider's partial on-disk output into a `ProviderResult`.

    `success=True` requires BOTH a complete row count and an aggregate
    `metrics.json`. Any weaker signal means calibrate crashed mid-run, so the
    partial rows are surfaced with `success=False` rather than lying to the FE.
    """
    if not results_data:
        return ProviderResult(provider=provider, success=False)
    runs = (
        build_evaluator_runs_for_eval_job(metrics_data, evaluator_id_by_metric_key)
        if metrics_data is not None
        else []
    )
    provider_done = metrics_data is not None and len(results_data) >= expected_total
    return ProviderResult(
        provider=provider,
        success=provider_done,
        metrics=metrics_data,
        results=results_data,
        evaluator_runs=runs or None,
    )


def build_in_progress_provider_result(
    provider: str,
    results_data: Optional[List[dict]],
    metrics_data: Optional[dict],
    expected_total: int,
    unit: str,
) -> dict:
    """Shape one provider's partial on-disk output for the in-progress GET reader.

    `success` is None (not False) while the run is still going, so the FE shows
    "running" rather than a failure.
    """
    if not results_data:
        return queued_provider_result(provider)
    provider_done = len(results_data) >= expected_total and metrics_data is not None
    prefix = "Done" if provider_done else "Running..."
    return {
        "provider": provider,
        "success": True if provider_done else None,
        "message": f"{prefix} ({len(results_data)} {unit} processed)",
        "metrics": metrics_data,
        "results": results_data,
    }


def queued_provider_result(provider: str) -> dict:
    return {
        "provider": provider,
        "success": None,
        "message": "Queued...",
        "metrics": None,
        "results": None,
    }


def merge_timeout_provider_results(
    requested_providers: List[str],
    existing_provider_results: List[dict],
    intermediate: List[ProviderResult],
) -> List[dict]:
    """Combine already-persisted results with whatever was left on disk at timeout.

    Existing `success: true` entries win, so a provider that finished before the
    run stalled is never downgraded by a partial re-read.
    """
    existing_success_map = {
        pr.get("provider"): pr
        for pr in existing_provider_results
        if pr.get("success") is True
    }
    intermediate_map = {r.provider: r.model_dump() for r in intermediate}

    merged = []
    for provider in requested_providers:
        if provider in existing_success_map:
            merged.append(existing_success_map[provider])
        elif provider in intermediate_map:
            merged.append(intermediate_map[provider])
        else:
            merged.append(
                {
                    "provider": provider,
                    "success": False,
                    "metrics": None,
                    "results": None,
                }
            )
    return merged


def build_run_config_fallback(
    output_dir: Path,
    temp_path: Path,
    config_data: dict,
) -> Path:
    """Return calibrate's own run-root config.json, or write a fallback.

    Calibrate's copy is preferred because it carries the evaluator IDs and
    metric-key map the read path needs.
    """
    config_file = output_dir / "config.json"
    if config_file.exists():
        return config_file
    config_file = temp_path / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    return config_file


def write_evaluator_config(task_id: str, job_details: dict, dest_dir: Path) -> Optional[Path]:
    """Write the calibrate `--config` file for a job's evaluators.

    Re-hydrates each evaluator to its CURRENT live version at run time
    (consistent with LLM tests / simulations), so editing an evaluator while the
    job is queued takes effect. The live-at-run-time snapshot is persisted back
    into details so finished-run reads render the exact version that ran.

    Returns None when the job has no evaluators, in which case no LLM judge runs.
    """
    raw_evaluators = job_details.get("evaluators") or []
    if not raw_evaluators:
        return None

    raw_evaluators = refresh_evaluators_to_live(raw_evaluators)
    update_job(task_id, details={"evaluators": raw_evaluators})
    config_path = dest_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {"evaluators": build_evaluator_cli_payload(raw_evaluators)},
            f,
            ensure_ascii=False,
        )
    return config_path


def run_calibrate_eval(
    eval_cmd: List[str], task_id: str, output_dir: Path, cwd: Path, label: str
) -> None:
    """Spawn the calibrate CLI and block until it exits, raising on failure.

    stdout/stderr go to files rather than pipes so a chatty run can't deadlock
    on a full buffer. The poll loop touches the job every couple of seconds to
    keep `updated_at` fresh, otherwise a long run trips the inactivity timeout.
    """
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
            cwd=str(cwd),
        )

        update_job(
            task_id,
            details={
                "pid": process.pid,
                "pgid": process.pid,
                "output_dir": str(output_dir),
            },
        )

        while process.poll() is None:
            time.sleep(_HEARTBEAT_INTERVAL)
            if process.poll() is None:
                update_job(task_id)

    with open(stdout_path, "r") as f:
        stdout = f.read()
    with open(stderr_path, "r") as f:
        stderr = f.read()

    if process.returncode != 0:
        logger.error(f"{label} eval failed with code {process.returncode}")
        logger.error(f"stderr: {stderr}")
        raise subprocess.CalledProcessError(
            process.returncode, eval_cmd, stdout, stderr
        )

    logger.info(f"{label} eval command completed successfully")


def require_providers(providers: List[str]) -> None:
    """Reject an eval request that names no provider to compare."""
    if not providers:
        raise HTTPException(
            status_code=400,
            detail="At least one provider must be specified",
        )


def claim_eval_queue_slot(org_uuid: str) -> tuple:
    """Ask the eval queue for a slot. Returns `(can_start, initial_status)`."""
    can_start = can_start_job(EVAL_JOB_TYPES, org_uuid)
    return can_start, (
        TaskStatus.IN_PROGRESS.value if can_start else TaskStatus.QUEUED.value
    )


def begin_eval_rerun(task_id: str, org_uuid: str, rerun_details: dict) -> tuple:
    """Reset a job's details for a re-run and claim a queue slot.

    Returns `(can_start, initial_status)`; the caller starts the thread only
    when `can_start`, otherwise the queue picks the job up later.
    """
    can_start, initial_status = claim_eval_queue_slot(org_uuid)
    update_job(
        task_id,
        status=initial_status,
        results={},
        details=rerun_details,
        replace_details=True,
    )
    return can_start, initial_status


def record_eval_task_crash(task_id: str, exc: Exception) -> None:
    """Mark a job failed after an error outside the per-run error handling."""
    traceback.print_exc()
    capture_exception_to_sentry(exc)
    update_job(
        task_id,
        status=TaskStatus.FAILED.value,
        results={"error": f"Task failed: {str(exc)}"},
    )


def load_eval_job_for_status(task_id: str, org_uuid: str) -> tuple:
    """Fetch an eval job for polling, or 404."""
    job = get_job(task_id, org_uuid=org_uuid)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")
    return job, job["status"], job.get("results") or {}, job.get("details") or {}


def apply_eval_job_timeout(
    task_id: str,
    job: dict,
    details: dict,
    results: dict,
    status: str,
    collect_intermediate: Callable[[Path, List[str]], List[ProviderResult]],
) -> str:
    """Fail a stalled in-progress job, preserving on-disk results. Returns the new status.

    Kills the calibrate process group first, otherwise the orphan keeps writing
    into a job the API has already given up on.
    """
    if status != TaskStatus.IN_PROGRESS.value:
        return status
    updated_at = job.get("updated_at")
    if not (updated_at and is_job_timed_out(updated_at)):
        return status

    logger.warning(f"Job {task_id} timed out, marking as failed")

    pid = details.get("pid") or details.get("pgid")
    if pid:
        kill_process_group(pid, task_id)

    requested_providers = details.get("providers", [])
    output_dir_str = details.get("output_dir")
    existing_provider_results = results.get("provider_results", [])

    if output_dir_str:
        try:
            output_dir = Path(output_dir_str)
            if output_dir.exists():
                results["provider_results"] = merge_timeout_provider_results(
                    requested_providers,
                    existing_provider_results,
                    collect_intermediate(output_dir, requested_providers),
                )
        except Exception as exc:
            logger.warning(f"Failed to collect intermediate results on timeout: {exc}")
            if existing_provider_results:
                results["provider_results"] = existing_provider_results

    results["error"] = "Job timed out after 5 minutes of inactivity"
    update_job(task_id, status=TaskStatus.FAILED.value, results=results)
    try_start_queued_job(EVAL_JOB_TYPES)
    return TaskStatus.FAILED.value


def build_completed_provider_result(
    provider: str,
    results_data: Optional[List[dict]],
    metrics_data: Optional[dict],
    evaluator_id_by_metric_key: dict,
    success: bool,
) -> ProviderResult:
    """Shape one provider's finished output. Caller decides `success`."""
    eruns = (
        build_evaluator_runs_for_eval_job(metrics_data, evaluator_id_by_metric_key)
        if metrics_data is not None
        else []
    )
    return ProviderResult(
        provider=provider,
        success=success,
        metrics=metrics_data,
        results=results_data,
        evaluator_runs=eruns or None,
    )


def load_eval_job_for_retry(task_id: str, job_type: str, org_uuid: str) -> tuple:
    """Fetch a finished eval job and its providers, or raise the retry 404/400."""
    job = get_job(task_id, org_uuid=org_uuid)
    if not job or job.get("type") != job_type:
        raise HTTPException(status_code=404, detail="Task not found")
    if job["status"] == TaskStatus.IN_PROGRESS.value:
        raise HTTPException(
            status_code=400,
            detail="Cannot retry a job that is still in progress",
        )

    details = job.get("details") or {}
    providers = details.get("providers") or []
    if not providers:
        raise HTTPException(
            status_code=400,
            detail="Original job is missing provider configuration",
        )
    return job, details, providers


def build_eval_status_response(
    task_id: str,
    status: str,
    job: dict,
    details: dict,
    results: dict,
    provider_results: List[dict],
    dataset_id: Optional[str],
    dataset_name: Optional[str],
):
    """Assemble the poll response shared by the STT and TTS status endpoints."""
    return TaskStatusResponse(
        task_id=task_id,
        status=status,
        language=details.get("language"),
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        provider_results=provider_results,
        leaderboard_summary=results.get("leaderboard_summary"),
        error=results.get("error"),
        is_public=bool(job.get("is_public")),
        share_token=job.get("share_token"),
    )


def normalize_and_enrich_provider_results(
    provider_results: List[dict], details: dict
) -> None:
    """Apply the read-path fixups every eval poll needs, in order."""
    for provider_result in provider_results:
        if provider_result.get("metrics"):
            provider_result["metrics"] = normalize_metrics(provider_result["metrics"])

    evaluator_snapshots = details.get("evaluators") or []
    enrich_evaluator_runs_with_current_names(provider_results, evaluator_snapshots)
    post_process_provider_results(
        provider_results,
        evaluator_snapshots=evaluator_snapshots,
        evaluator_id_by_metric_key=load_evaluator_metric_key_map(details),
    )


def upload_provider_dir(
    s3,
    provider_output_dir: Path,
    s3_bucket: str,
    results_prefix: str,
    swallow_errors: bool = False,
) -> dict:
    """Upload one provider's output tree to S3, returning local audio path -> S3 key.

    `swallow_errors` is for the partial-results path, where a failed upload must
    not mask the original error that got us there.
    """
    audio_path_to_s3_key = {}
    for root, _dirs, files in os.walk(provider_output_dir):
        for file in files:
            local_file_path = Path(root) / file
            relative_path = local_file_path.relative_to(provider_output_dir)
            s3_key = f"{results_prefix}/{relative_path}"
            try:
                upload_file_to_s3(s3, local_file_path, s3_bucket, s3_key)
            except Exception:
                if not swallow_errors:
                    raise
                continue
            if file.endswith((".wav", ".mp3", ".ogg")):
                audio_path_to_s3_key[str(local_file_path)] = s3_key
    return audio_path_to_s3_key


def upload_leaderboard(
    s3, output_dir: Path, s3_bucket: str, leaderboard_prefix: str
) -> Optional[dict]:
    """Read calibrate's leaderboard and mirror it to S3. None when it wasn't written."""
    leaderboard_dir = output_dir / "leaderboard"
    logger.info(
        f"Output directory contents: {[f.name for f in output_dir.iterdir()]}"
    )
    if not leaderboard_dir.exists():
        logger.warning(f"Leaderboard directory does not exist: {leaderboard_dir}")
        return None

    logger.info(f"Leaderboard directory exists: {leaderboard_dir}")
    leaderboard_summary = read_leaderboard_xlsx(leaderboard_dir)
    upload_provider_dir(s3, leaderboard_dir, s3_bucket, leaderboard_prefix)
    return leaderboard_summary


def record_eval_failure(
    task_id: str,
    exc: Exception,
    error_message: str,
    s3,
    output_dir: Path,
    s3_bucket: str,
    outputs_prefix: str,
    collect_intermediate: Callable[[], List[ProviderResult]],
) -> None:
    """Mark an eval job failed, preserving whatever landed on disk first.

    STT/TTS intermediate results are disk-only while in progress, so skipping
    this collection would lose every successful provider's data on one crash.
    """
    traceback.print_exc()
    capture_exception_to_sentry(exc)
    error_results = {"error": error_message}
    try:
        if output_dir.exists():
            intermediate = collect_intermediate()
            if intermediate:
                error_results["provider_results"] = [
                    r.model_dump() for r in intermediate
                ]
            upload_directory_tree_to_s3(s3, output_dir, s3_bucket, outputs_prefix)
    except Exception:
        pass
    update_job(task_id, status=TaskStatus.FAILED.value, results=error_results)


def finalize_eval_results(
    task_id: str, provider_results: List[ProviderResult], leaderboard_summary: Optional[dict]
) -> None:
    """Write the terminal status and results for a finished eval job."""
    failed = [r.provider for r in provider_results if not r.success]
    update_job(
        task_id,
        status=TaskStatus.FAILED.value if failed else TaskStatus.DONE.value,
        results={
            "provider_results": [r.model_dump() for r in provider_results],
            "leaderboard_summary": leaderboard_summary,
            "error": f"Some providers failed: {', '.join(failed)}" if failed else None,
        },
    )


class VisibilityRequest(BaseModel):
    is_public: bool = Field(
        description="`true` to make the job publicly shareable. `false` to make it private"
    )


class VisibilityResponse(BaseModel):
    is_public: bool = Field(description="Whether the job is now publicly shareable")
    share_token: str | None = Field(
        None,
        description="Opaque token for the public share URL when `is_public` is true",
    )


def set_eval_job_visibility(
    task_id: str, job_type: str, org_uuid: str, is_public: bool
) -> VisibilityResponse:
    job = get_job(task_id, org_uuid=org_uuid)
    if not job or job.get("type") != job_type:
        raise HTTPException(status_code=404, detail="Task not found")

    token_to_persist, token_to_return = compute_share_token_toggle(job, is_public)
    update_job_visibility(task_id, is_public, token_to_persist)
    return VisibilityResponse(is_public=is_public, share_token=token_to_return)
