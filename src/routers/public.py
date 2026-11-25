"""Public endpoints for shared eval and run results.

View STT/TTS runs, agent tests, benchmarks, simulations, and annotation
results via share links, no sign-in required. Annotator labelling jobs use
a separate token for read-write access.
"""

import asyncio
import logging
import os
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from calibrate_agent.chat import agent_reply
from calibrate_agent.connections import TextAgentConnection
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt

from db import (
    DEFAULT_EVALUATORS_SEED,
    get_evaluator_by_slug,
    get_evaluator_version,
    get_job_by_share_token,
    get_agent_test_job_by_share_token,
    get_simulation_job_by_share_token,
    get_simulation_jobs_for_simulation,
    get_annotation_job_by_token,
    get_annotation_job_by_view_token,
    get_annotation_task,
    get_org_by_invite_token,
    get_evaluator_ids_for_job,
    get_evaluators_for_job,
    get_annotator,
    get_job_items,
    get_annotations_for_job,
    upsert_annotation,
    update_annotation_job_status,
    get_evaluator_runs_for_job,
    get_eval_job_items,
    count_chat_replies_today,
    get_chat_conversation,
    count_live_traces,
    create_trace_with_eval_run,
    get_agent_by_chat_token,
    get_member_role,
    get_trace,
    release_chat_reply,
    reserve_chat_reply,
    save_chat_conversation,
    get_tools_for_agent,
    set_trace_feedback,
)
from provider_keys import provider_env
from provider_status import PROVIDER_ENV_VARS
from routers.agents import _sanitize_headers
from routers.traces import TraceFeedbackResponse, TraceFeedbackUpdate
from auth_utils import get_optional_user_id
from routers.org_limits import daily_message_limit, effective_limits, record_limit_hit
from shared_enums import DEFAULT_AGENT_INTERACTION_TYPE, EvaluatorType
from testing.fake_calibrate_agent import FAKE_RESPONSE
from utils import (
    default_agent_llm_model,
    build_tool_configs,
    capture_exception_to_sentry,
    env_bool,
    env_int,
    public_url_error,
    with_calibrate_eval_header,
    TaskStatus,
    AnnotationStatus,
    SimulationRunType,
    AnnotationTaskTypeLiteral,
    OutputTypeLiteral,
    EVALUATOR_TYPE_DESCRIPTION,
    OUTPUT_TYPE_DESCRIPTION,
    ProviderResult,
    enrich_evaluator_runs_with_current_names,
    generate_presigned_download_url,
    get_s3_output_config,
    load_evaluator_metric_key_map,
    normalize_metrics,
    post_process_provider_results,
    presign_annotation_items_audio,
    presign_audio_path,
    presign_tts_provider_results_audio,
)

# Re-use the audio URL helper from simulations (no circular import risk)
from routers.simulations import (
    SimulationEvaluatorRef,
    apply_simulation_job_evaluator_enrichment,
    _get_audio_urls_from_s3_key,
)
from routers.agent_tests import (
    _enrich_test_results_with_evaluators,
    _enrich_model_results_with_evaluators,
    _build_evaluators_block_for_test_run,
    _summarize_case_rows,
    evaluator_totals_from_rows,
    test_uuid_by_calibrate_id,
    _tool_call_evaluator_for_run,
    _RunDetailMode,
    find_case_result,
    LEADERBOARD_SUMMARY_DESCRIPTION,
    LEADERBOARD_SUMMARY_EXAMPLE,
    TestCaseResult,
    _RUN_NAME_DESCRIPTION as RUN_NAME_DESCRIPTION,
    run_display_name,
)
from routers.annotation_tasks import (
    _build_evaluators_block_for_eval_job,
    _strip_details_evaluators,
    _strip_run_evaluator_blocks,
    _human_agreement_for_run,
    _shape_eval_job_for_response,
)
from annotation_eval_runner import (
    ANNOTATION_EVAL_JOB_TYPE,
    TOOL_CALL_EVALUATOR_TYPE,
    required_evaluator_ids_for_item,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/public", tags=["public"])

_EXAMPLE_TASK_UUID = "a3b2c1d0-e5f4-3210-abcd-ef1234567890"
_EXAMPLE_DATASET_UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
_EXAMPLE_EVALUATOR_UUID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
_EXAMPLE_JOB_UUID = "b1c2d3e4-f5a6-7890-bcde-f12345678901"
_EXAMPLE_ITEM_UUID = "c1d2e3f4-a5b6-4789-abcd-ef0123456789"
_EXAMPLE_ANNOTATION_TASK_UUID = "d4e5f6a7-b8c9-4012-def0-234567890abc"


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class PublicSTTResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="STT eval job ID",
        examples=[_EXAMPLE_TASK_UUID],
    )
    status: TaskStatus = Field(description="Job status")
    language: Optional[str] = Field(None, description="Evaluated language code")
    dataset_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Source dataset ID",
        examples=[_EXAMPLE_DATASET_UUID],
    )
    dataset_name: Optional[str] = Field(None, description="Source dataset name")
    provider_results: Optional[List[ProviderResult]] = Field(
        None, description="Transcription results and metrics for each provider"
    )
    leaderboard_summary: Optional[List[Dict[str, Any]]] = Field(
        None, description="Ranked provider comparison"
    )
    error: Optional[str] = Field(None, description="Failure message")


class PublicTTSResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="TTS eval job ID",
        examples=[_EXAMPLE_TASK_UUID],
    )
    status: TaskStatus = Field(description="Job status")
    language: Optional[str] = Field(None, description="Evaluated language code")
    dataset_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Source dataset ID",
        examples=[_EXAMPLE_DATASET_UUID],
    )
    dataset_name: Optional[str] = Field(None, description="Source dataset name")
    provider_results: Optional[List[ProviderResult]] = Field(
        None, description="Synthesis results and metrics for each provider"
    )
    leaderboard_summary: Optional[List[Dict[str, Any]]] = Field(
        None, description="Ranked provider comparison"
    )
    error: Optional[str] = Field(None, description="Failure message")


_ABORTED_DESCRIPTION = (
    "Whether a user stopped this run before it finished. The results collected "
    "up to that point are kept, and test cases that never ran are counted "
    "neither as passed nor as failed"
)


class PublicTestRunResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="LLM test run job ID",
        examples=[_EXAMPLE_TASK_UUID],
    )
    name: str = Field(description=RUN_NAME_DESCRIPTION)
    status: TaskStatus = Field(description="Run status")
    total_tests: Optional[int] = Field(None, description="Total test cases in the run")
    passed: Optional[int] = Field(None, description="Test cases that passed")
    failed: Optional[int] = Field(None, description="Test cases that failed")
    # Top-level evaluator block — name/description/output_type/rubric
    # shared across every judge_results row. Rows reference back via
    # `evaluator_uuid` so the rubric isn't duplicated per test case.
    evaluators: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Shared evaluator definitions, each with its name, description, output type, and rubric. Rows reference these by evaluator ID",
    )
    evaluator_summary: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Totals for each evaluator over the whole run, matching the shape a benchmark reports for each model. Only evaluators that returned a verdict appear",
    )
    results: Optional[List[Dict[str, Any]]] = Field(
        None, description="Results for each test case"
    )
    # Aggregated latency/cost/total_tokens: {mean, min, max, count}. Values are
    # `Any` — don't assume int: total_tokens is per-run an int but its aggregate
    # `mean` can be fractional. None when calibrate omits it (eval-only / no cost
    # or token usage reported).
    latency_ms: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated latency (`{p50, p95, p99, count}`)",
    )
    cost: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated cost in USD (`{mean, min, max, count}`)",
    )
    total_tokens: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated token usage (`{mean, min, max, count}`)",
    )
    unanswered_tests: Optional[int] = Field(
        None,
        description="Number of test cases that produced no answer because the agent or the judge could not be reached, which makes the pass rate an unfair measure of the agent",
    )
    stopped_early: bool = Field(
        False,
        description="Whether the run stopped before starting every test case, after too many failed in a row",
    )
    aborted: bool = Field(False, description=_ABORTED_DESCRIPTION)
    error: Optional[str] = Field(
        None,
        description="Why the run could not be carried out, when it failed before producing any result",
    )


class PublicBenchmarkResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="LLM benchmark job ID",
        examples=[_EXAMPLE_TASK_UUID],
    )
    name: str = Field(description=RUN_NAME_DESCRIPTION)
    status: TaskStatus = Field(description="Run status")
    # Same as PublicTestRunResponse.evaluators — shared by every model's
    # test_results inside model_results[] (all models run the same suite).
    evaluators: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Shared evaluator definitions referenced by every model's results",
    )
    model_results: Optional[List[Dict[str, Any]]] = Field(
        None, description="Test results for each model"
    )
    leaderboard_summary: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=LEADERBOARD_SUMMARY_DESCRIPTION,
        examples=[LEADERBOARD_SUMMARY_EXAMPLE],
    )
    stopped_early: bool = Field(
        False,
        description="Whether any model's run stopped before starting every test case, after too many failed in a row",
    )
    aborted: bool = Field(False, description=_ABORTED_DESCRIPTION)
    error: Optional[str] = Field(
        None,
        description="Why the run could not be carried out, when it failed before producing any result",
    )


class PublicSimulationRunResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="Simulation run job ID",
        examples=[_EXAMPLE_TASK_UUID],
    )
    name: str = Field(description="Display name for the run, e.g. `Run 1`")
    status: TaskStatus = Field(description="Run status")
    type: SimulationRunType = Field(description="Run mode")
    updated_at: str = Field(description="When the run was last updated (ISO 8601 UTC)")
    total_simulations: Optional[int] = Field(None, description="Number of simulations in the run")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Aggregated run metrics")
    simulation_results: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Results for each simulation. Voice runs include freshly presigned audio URLs",
    )
    evaluators: Optional[List[SimulationEvaluatorRef]] = Field(
        None, description="Evaluators applied to the run"
    )
    error: Optional[str] = Field(None, description="Failure message")


class PublicAnnotationEvalTaskRef(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Annotation task ID",
        examples=[_EXAMPLE_ANNOTATION_TASK_UUID],
    )
    name: str = Field(description="Annotation task name")
    type: AnnotationTaskTypeLiteral = Field(
        description="Annotation task type"
    )
    description: Optional[str] = Field(None, description="Task description")


class PublicAnnotationEvalResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="Parent annotation task ID",
        examples=[_EXAMPLE_ANNOTATION_TASK_UUID],
    )
    job_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the annotation evaluator run",
        examples=[_EXAMPLE_JOB_UUID],
    )
    status: AnnotationStatus = Field(description="Run status. Share links only expose completed runs")
    created_at: Optional[str] = Field(None, description="When the run was created (ISO 8601 UTC)")
    completed_at: Optional[str] = Field(None, description="When the run finished (ISO 8601 UTC)")
    updated_at: Optional[str] = Field(None, description="When the run was last updated (ISO 8601 UTC)")
    task: PublicAnnotationEvalTaskRef = Field(description="Parent annotation task summary")
    # `details` mirrors the authenticated GET shape so a shared FE component
    # can read `job.details?.evaluators` against either endpoint without
    # branching. Only the safe-to-share keys are forwarded — operational
    # fields (pid, pgid, s3_prefix, user_id) are intentionally stripped.
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Safe-to-share job metadata. Operational fields are stripped",
    )
    # Top-level mirrors retained for the existing public consumers that
    # were already reading them. Both shapes carry the same data.
    evaluators: Optional[List[Dict[str, Any]]] = Field(
        None, description="Evaluator definitions applied in the run"
    )
    item_count: Optional[int] = Field(None, description="Number of items evaluated")
    items: Optional[List[Dict[str, Any]]] = Field(None, description="Evaluated items")
    runs: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Evaluator run rows for each item. Each keys back into `evaluators[]` by evaluator and version ID",
    )
    human_agreement: Optional[Dict[str, Any]] = Field(
        None, description="Human-vs-evaluator agreement metrics"
    )
    error: Optional[str] = Field(None, description="Failure message")


class PublicDefaultEvaluatorVersionResponse(BaseModel):
    output_config: Optional[Dict[str, Any]] = Field(
        None, description="The rubric: the rows of the rating scale, each a number and its label"
    )


class PublicDefaultEvaluatorResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Evaluator ID",
        examples=[_EXAMPLE_EVALUATOR_UUID],
    )
    name: str = Field(description="Evaluator display name")
    description: Optional[str] = Field(None, description="Evaluator description")
    evaluator_type: EvaluatorType = Field(
        description=EVALUATOR_TYPE_DESCRIPTION
    )
    output_type: OutputTypeLiteral = Field(description=OUTPUT_TYPE_DESCRIPTION)
    live_version: Optional[PublicDefaultEvaluatorVersionResponse] = Field(
        None, description="The version that is currently live"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_simulation_results_with_presigned_urls(
    job: Dict[str, Any],
    simulation_results: List[Dict[str, Any]],
    status: str,
) -> List[Dict[str, Any]]:
    """
    For DONE voice simulations regenerate presigned audio URLs on-the-fly.
    Mirrors the logic in routers/simulations.py::get_simulation_run_status.
    """
    if job.get("type") != "voice" or not simulation_results:
        return simulation_results

    if status == TaskStatus.DONE.value:
        try:
            s3_bucket = get_s3_output_config()
            for sim_result in simulation_results:
                audios_s3_key_prefix = sim_result.get("audios_s3_path")
                if audios_s3_key_prefix:
                    audio_urls = _get_audio_urls_from_s3_key(
                        audios_s3_key_prefix,
                        s3_bucket,
                        transcript=sim_result.get("transcript"),
                    )
                    sim_result["audio_urls"] = audio_urls

                conversation_wav_s3_key = sim_result.get("conversation_wav_s3_key")
                if conversation_wav_s3_key:
                    conversation_wav_url = generate_presigned_download_url(
                        conversation_wav_s3_key, bucket=s3_bucket
                    )
                    sim_result["conversation_wav_url"] = conversation_wav_url or ""
                else:
                    sim_result["conversation_wav_url"] = ""
        except Exception as exc:
            logger.warning(f"Failed to generate audio URLs for public endpoint: {exc}")

    return simulation_results


def _get_simulation_run_name(job: Dict[str, Any]) -> str:
    """Return 'Run N' by looking at the job's position among sibling jobs."""
    simulation_id = job.get("simulation_id")
    if not simulation_id:
        return "Run 1"
    all_jobs = get_simulation_jobs_for_simulation(simulation_id)
    sorted_jobs = sorted(all_jobs, key=lambda j: j.get("created_at", ""))
    for idx, j in enumerate(sorted_jobs, start=1):
        if j["uuid"] == job["uuid"]:
            return f"Run {idx}"
    return "Run 1"


def _ensure_valid_public_share_token(share_token: str) -> None:
    """Allow metadata reads only for callers that already have a valid public share token.

    `get_job_by_share_token` covers stt-eval, tts-eval, and annotation-eval —
    they all live in the generic `jobs` table."""
    if (
        get_job_by_share_token(share_token)
        or get_agent_test_job_by_share_token(share_token)
        or get_simulation_job_by_share_token(share_token)
    ):
        return
    raise HTTPException(status_code=404, detail="Not found")


def _parse_evaluator_types(types: Optional[str]) -> Optional[set[str]]:
    if types is None or not types.strip():
        return None

    allowed = {"stt", "tts", "llm", "llm-general", "conversation", "tool-call"}
    parsed = {item.strip() for item in types.split(",") if item.strip()}
    invalid = parsed - allowed
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"types must contain only: {', '.join(sorted(allowed))}",
        )
    return parsed


def _public_default_evaluator_response(
    evaluator: Dict[str, Any],
) -> PublicDefaultEvaluatorResponse:
    live_version = None
    if evaluator.get("live_version_id"):
        version = get_evaluator_version(evaluator["live_version_id"])
        if version and version.get("evaluator_id") == evaluator["uuid"]:
            live_version = PublicDefaultEvaluatorVersionResponse(
                output_config=version.get("output_config")
            )

    return PublicDefaultEvaluatorResponse(
        uuid=evaluator["uuid"],
        name=evaluator["name"],
        description=evaluator.get("description"),
        evaluator_type=evaluator.get("evaluator_type", "llm"),
        output_type=evaluator.get("output_type", "binary"),
        live_version=live_version,
    )


# ---------------------------------------------------------------------------
# Public GET endpoints (no auth)
# ---------------------------------------------------------------------------


@router.get(
    "/evaluators/defaults",
    response_model=List[PublicDefaultEvaluatorResponse],
    summary="List public default evaluators",
)
def get_public_default_evaluators(
    share_token: str = Query(..., min_length=1, description="Share token that grants access to the linked run"),
    types: Optional[str] = Query(
        None,
        description="Evaluator types to include as a comma-separated list, any of `stt`, `tts`, `llm`, `llm-general`, `conversation`, `tool-call`. Omit for all",
    ),
):
    """List default evaluator metadata when you have a valid share token"""
    _ensure_valid_public_share_token(share_token)
    requested_types = _parse_evaluator_types(types)

    defaults: List[PublicDefaultEvaluatorResponse] = []
    for seed in DEFAULT_EVALUATORS_SEED:
        if requested_types is not None and seed["evaluator_type"] not in requested_types:
            continue
        evaluator = get_evaluator_by_slug(seed["slug"])
        if evaluator and evaluator.get("org_uuid") is None:
            defaults.append(_public_default_evaluator_response(evaluator))
    return defaults


@router.get("/stt/{share_token}", response_model=PublicSTTResponse, summary="Get shared STT run")
def get_public_stt(
    share_token: str = Path(description="Share token for the STT run"),
):
    """Get a shared STT evaluation result"""
    job = get_job_by_share_token(share_token, job_type="stt-eval")
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    task_id = job["uuid"]
    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    provider_results = results.get("provider_results") or []

    # Normalize metrics format for backward compatibility (list -> dict)
    for pr in provider_results:
        if pr.get("metrics"):
            pr["metrics"] = normalize_metrics(pr["metrics"])

    enrich_evaluator_runs_with_current_names(
        provider_results, details.get("evaluators") or []
    )

    # Same canonical post-processing as the authenticated endpoints — public
    # callers see the same shape (evaluator_outputs[uuid], typed values,
    # evaluator_runs[].aggregate) instead of the legacy flat-keyed fallback.
    # `evaluator_id_by_metric_key` (read from on-disk config.json) is what
    # makes the per-row lift safe against reserved-column collisions —
    # without it the helper falls back to snapshot display names and an
    # evaluator named e.g. `wer` would lift the built-in WER value into
    # `evaluator_outputs[uuid].value`. Authenticated handlers already pass
    # this; the public mirrors must too or they'll disagree on the same run.
    post_process_provider_results(
        provider_results,
        evaluator_snapshots=details.get("evaluators") or [],
        evaluator_id_by_metric_key=load_evaluator_metric_key_map(details),
    )

    # Enrich result rows with presigned audio URLs from the dataset
    audio_paths = details.get("audio_paths", [])
    if audio_paths:
        # Collect IDs actually present in results to avoid unnecessary presigning
        needed_ids: set[str] = set()
        for pr in provider_results:
            for row in pr.get("results") or []:
                if row.get("id"):
                    needed_ids.add(row["id"])

        audio_url_map = {}
        for idx, path in enumerate(audio_paths):
            audio_id = f"audio_{idx + 1}"
            if audio_id in needed_ids:
                audio_url_map[audio_id] = presign_audio_path(path)

        for pr in provider_results:
            for row in pr.get("results") or []:
                row["audio_url"] = audio_url_map.get(row.get("id", ""))

    return PublicSTTResponse(
        task_id=task_id,
        status=status,
        language=details.get("language"),
        dataset_id=details.get("dataset_id"),
        dataset_name=details.get("dataset_name"),
        provider_results=provider_results or None,
        leaderboard_summary=results.get("leaderboard_summary"),
        error=results.get("error"),
    )


@router.get("/tts/{share_token}", response_model=PublicTTSResponse, summary="Get shared TTS run")
def get_public_tts(
    share_token: str = Path(description="Share token for the TTS run"),
):
    """Get a shared TTS evaluation result"""
    job = get_job_by_share_token(share_token, job_type="tts-eval")
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    task_id = job["uuid"]
    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    provider_results = results.get("provider_results") or []

    # Normalize metrics format for backward compatibility (list -> dict)
    for pr in provider_results:
        if pr.get("metrics"):
            pr["metrics"] = normalize_metrics(pr["metrics"])

    enrich_evaluator_runs_with_current_names(
        provider_results, details.get("evaluators") or []
    )

    # Same canonical post-processing as the authenticated endpoints — public
    # callers see the same shape (evaluator_outputs[uuid], typed values,
    # evaluator_runs[].aggregate) instead of the legacy flat-keyed fallback.
    # `evaluator_id_by_metric_key` (read from on-disk config.json) is what
    # makes the per-row lift safe against reserved-column collisions —
    # without it the helper falls back to snapshot display names and an
    # evaluator named e.g. `wer` would lift the built-in WER value into
    # `evaluator_outputs[uuid].value`. Authenticated handlers already pass
    # this; the public mirrors must too or they'll disagree on the same run.
    post_process_provider_results(
        provider_results,
        evaluator_snapshots=details.get("evaluators") or [],
        evaluator_id_by_metric_key=load_evaluator_metric_key_map(details),
    )

    provider_results = presign_tts_provider_results_audio(provider_results, status)

    return PublicTTSResponse(
        task_id=task_id,
        status=status,
        language=details.get("language"),
        dataset_id=details.get("dataset_id"),
        dataset_name=details.get("dataset_name"),
        provider_results=provider_results or None,
        leaderboard_summary=results.get("leaderboard_summary"),
        error=results.get("error"),
    )


_SHARED_SUMMARY_MODE_DESCRIPTION = (
    "How much of each test case to return. `full` returns every field of every "
    "case. `summary` returns one light row per case, with its ID, name, "
    "verdict and short reason, leaving out the conversation, the agent's output "
    "and the evaluator verdicts. Read those one case at a time from the same "
    "share token, at `/public/test-run/{share_token}/results/{test_uuid}` "
    "for a run or `/public/benchmark/{share_token}/results/{test_uuid}` for "
    "a benchmark"
)


@router.get("/test-run/{share_token}", response_model=PublicTestRunResponse, summary="Get shared test run")
def get_public_test_run(
    share_token: str = Path(description="Share token for the LLM test run"),
    mode: _RunDetailMode = Query("full", description=_SHARED_SUMMARY_MODE_DESCRIPTION),
):
    """Get a shared LLM test run result"""
    job = get_agent_test_job_by_share_token(share_token, job_type="llm-unit-test")
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    task_id = job["uuid"]
    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    evaluators_snapshot = details.get("evaluators_by_test_id") or {}
    evaluator_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    tool_call_evaluator = _tool_call_evaluator_for_run(details)
    _enrich_test_results_with_evaluators(
        results.get("test_results"),
        evaluators_snapshot,
        evaluator_cache,
        tool_call_evaluator,
        test_uuid_by_calibrate_id(details),
    )
    evaluators_block = _build_evaluators_block_for_test_run(
        evaluators_snapshot,
        test_results=results.get("test_results"),
        evaluator_cache=evaluator_cache,
        tool_call_evaluator=tool_call_evaluator,
    )

    evaluator_summary = evaluator_totals_from_rows(
        results.get("test_results"), evaluators_block, evaluators_snapshot
    )

    if mode == "summary":
        _summarize_case_rows(results.get("test_results"))

    return PublicTestRunResponse(
        task_id=task_id,
        name=run_display_name(job),
        status=status,
        total_tests=results.get("total_tests"),
        passed=results.get("passed"),
        failed=results.get("failed"),
        evaluators=evaluators_block or None,
        evaluator_summary=evaluator_summary,
        results=results.get("test_results"),
        latency_ms=results.get("latency_ms"),
        cost=results.get("cost"),
        total_tokens=results.get("total_tokens"),
        unanswered_tests=results.get("unanswered_tests"),
        stopped_early=bool(results.get("stopped_early")),
        aborted=bool(details.get("aborted")),
        error=str(results["error"]) if results.get("error") else None,
    )


@router.get("/benchmark/{share_token}", response_model=PublicBenchmarkResponse, summary="Get shared benchmark")
def get_public_benchmark(
    share_token: str = Path(description="Share token for the LLM benchmark run"),
    mode: _RunDetailMode = Query("full", description=_SHARED_SUMMARY_MODE_DESCRIPTION),
):
    """Get a shared LLM benchmark result"""
    job = get_agent_test_job_by_share_token(share_token, job_type="llm-benchmark")
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    task_id = job["uuid"]
    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    evaluators_snapshot = details.get("evaluators_by_test_id") or {}
    evaluator_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    tool_call_evaluator = _tool_call_evaluator_for_run(details)
    _enrich_model_results_with_evaluators(
        results.get("model_results"),
        evaluators_snapshot,
        evaluator_cache,
        tool_call_evaluator,
        test_uuid_by_calibrate_id(details),
    )
    evaluators_block = _build_evaluators_block_for_test_run(
        evaluators_snapshot,
        model_results=results.get("model_results"),
        evaluator_cache=evaluator_cache,
        tool_call_evaluator=tool_call_evaluator,
    )

    if mode == "summary":
        for model in results.get("model_results") or []:
            _summarize_case_rows(model.get("test_results"))

    return PublicBenchmarkResponse(
        task_id=task_id,
        name=run_display_name(job),
        status=status,
        evaluators=evaluators_block or None,
        model_results=results.get("model_results"),
        leaderboard_summary=results.get("leaderboard_summary"),
        stopped_early=bool(results.get("stopped_early")),
        aborted=bool(details.get("aborted")),
        error=str(results["error"]) if results.get("error") else None,
    )


@router.get(
    "/test-run/{share_token}/results/{test_uuid}",
    response_model=TestCaseResult,
    summary="Get shared test case result",
)
def get_public_test_run_case(
    share_token: str = Path(description="Share token for the LLM test run"),
    test_uuid: str = Path(
        description="The test whose result to read, as `test_uuid` on the case"
    ),
):
    """Get the full result of one test case in a shared run"""
    job = get_agent_test_job_by_share_token(share_token, job_type="llm-unit-test")
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    return find_case_result(job, test_uuid, None)


@router.get(
    "/benchmark/{share_token}/results/{test_uuid}",
    response_model=TestCaseResult,
    summary="Get shared benchmark case result",
)
def get_public_benchmark_case(
    share_token: str = Path(description="Share token for the LLM benchmark run"),
    test_uuid: str = Path(
        description="The test whose result to read, as `test_uuid` on the case"
    ),
    model: Optional[str] = Query(
        None,
        description="Which model's answer to read",
        examples=["openai/gpt-4.1"],
    ),
):
    """Get the full result of one test case in a shared benchmark"""
    job = get_agent_test_job_by_share_token(share_token, job_type="llm-benchmark")
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    return find_case_result(job, test_uuid, model)


@router.get("/simulation-run/{share_token}", response_model=PublicSimulationRunResponse, summary="Get shared simulation run")
def get_public_simulation_run(
    share_token: str = Path(description="Share token for the simulation run"),
):
    """Get a shared simulation run result"""
    job = get_simulation_job_by_share_token(share_token)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    task_id = job["uuid"]
    status = job["status"]
    results = job.get("results") or {}

    simulation_results = results.get("simulation_results") or []
    simulation_results = _build_simulation_results_with_presigned_urls(
        job, simulation_results, status
    )

    evaluators_out, simulation_results = apply_simulation_job_evaluator_enrichment(
        job.get("details") or {}, simulation_results
    )

    run_name = _get_simulation_run_name(job)

    return PublicSimulationRunResponse(
        task_id=task_id,
        name=run_name,
        status=status,
        type=job["type"],
        updated_at=job["updated_at"],
        total_simulations=results.get("total_simulations"),
        metrics=results.get("metrics"),
        simulation_results=simulation_results or None,
        evaluators=evaluators_out,
        error=results.get("error"),
    )


# ---------------------------------------------------------------------------
# Annotation evaluator-run jobs (public, share_token toggled by owner)
# ---------------------------------------------------------------------------


@router.get(
    "/annotation-eval/{share_token}",
    response_model=PublicAnnotationEvalResponse,
    summary="Get shared annotation eval run",
)
def get_public_annotation_eval(
    share_token: str = Path(description="Share token for the annotation evaluator run"),
):
    """Get a shared annotation evaluator-run result"""
    # Only `done` runs are exposed; in-flight or failed runs return 404.
    job = get_job_by_share_token(share_token, job_type=ANNOTATION_EVAL_JOB_TYPE)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    # Defense-in-depth: never expose an in-flight or failed annotation-eval
    # run via the public link, even if some other code path managed to flip
    # `is_public` before the job reached `done`. The owner PATCH route at
    # `/annotation-tasks/{task_uuid}/evaluator-runs/{job_uuid}/visibility`
    # already enforces this gate, but other generic-jobs visibility routes
    # (stt/tts) wouldn't catch a wrong-type UUID — so we re-assert here.
    if job.get("status") != "done":
        raise HTTPException(status_code=404, detail="Not found")

    shaped = _shape_eval_job_for_response(job)
    task_uuid = shaped.get("task_id")
    task = get_annotation_task(task_uuid) if task_uuid else None
    if not task:
        # The parent task was deleted out from under the share link.
        raise HTTPException(status_code=404, detail="Not found")

    evaluator_runs_in_this_job = get_evaluator_runs_for_job(job["uuid"])
    details = job.get("details") or {}

    # Forward only the safe-to-share keys from the raw `details` blob. Strip
    # operational/identifying ones — `pid`, `pgid`, `s3_prefix`, `user_id`,
    # `output_dir`, etc. — that the auth view exposes but a public viewer
    # has no business seeing. Whitelist (not blacklist) so any future
    # addition to the runner's details dict stays private until explicitly
    # opted in here.
    public_details_whitelist = {
        "task_id",
        "evaluators",
        "item_count",
        "item_ids",
        "metrics",
        "completed_at",
    }
    public_details = {
        k: v for k, v in details.items() if k in public_details_whitelist
    }
    # `evaluators` is promoted to the response's top-level field — drop the
    # slim snapshot from the forwarded `details` to keep the auth and
    # public responses shaped the same way.
    public_details.pop("evaluators", None)

    evaluators_block = _build_evaluators_block_for_eval_job(details, evaluator_runs_in_this_job)

    return PublicAnnotationEvalResponse(
        task_id=task_uuid,
        job_uuid=job["uuid"],
        status=shaped["status"],
        created_at=job.get("created_at"),
        completed_at=shaped.get("completed_at"),
        updated_at=job.get("updated_at"),
        task=PublicAnnotationEvalTaskRef(
            uuid=task["uuid"],
            name=task["name"],
            type=task["type"],
            description=task.get("description"),
        ),
        details=public_details,
        evaluators=evaluators_block,
        item_count=details.get("item_count"),
        items=get_eval_job_items(job["uuid"]),
        # Per-run `evaluator` / `evaluator_version` blobs are intentionally
        # not surfaced — `(evaluator_id, evaluator_version_id)` on each run
        # keys back into the top-level evaluators[] block.
        runs=_strip_run_evaluator_blocks(evaluator_runs_in_this_job),
        human_agreement=_human_agreement_for_run(task_uuid, evaluator_runs_in_this_job),
        error=shaped.get("error"),
    )


# ---------------------------------------------------------------------------
# Annotation jobs (public, token-only)
# ---------------------------------------------------------------------------


def _resolve_public_annotation_job(token: str) -> Dict[str, Any]:
    """Fetch a job by its public_token, ensuring it's a real shareable link
    (not a CSV-import sentinel). Raises 404 otherwise."""
    job = get_annotation_job_by_token(token)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _build_annotation_job_payload(
    job: Dict[str, Any], read_only: bool
) -> Dict[str, Any]:
    """Shared response shape for both the annotator route (read+write) and
    the viewer route (read-only). When `read_only=True`, the response carries
    a `read_only` flag so the FE can disable form inputs. The annotator
    identity is included in both modes — the viewer is meant to see *whose*
    labels they're looking at.

    Each item carries `is_tool_call` (stamped by `get_job_items`): true when
    the row's agent output is a tool call rather than text. The form draws the
    tool-call evaluator only on those rows and every other evaluator only on
    the rest, pairing this flag against each entry's `evaluator_type`."""
    task = get_annotation_task(job["task_id"])
    if not task:
        raise HTTPException(status_code=404, detail="Job not found")
    annotator = get_annotator(job["annotator_id"])
    # Read the SNAPSHOTTED evaluator set so the labelling form columns match
    # exactly what was assigned at job creation, regardless of subsequent
    # link/unlink on the parent task.
    evaluators = get_evaluators_for_job(job["uuid"])
    # Enrich each evaluator with the live version's rubric, variable specs,
    # and derived scale bounds so the FE has everything it needs to render
    # the labelling form without a second roundtrip.
    from llm_judge import _scale_bounds  # local to avoid module-load cycle

    job_reasoning_mode = job.get("reasoning_mode") or "optional"
    for ev in evaluators:
        ev["reasoning_mode"] = ev.get("reasoning_mode") or job_reasoning_mode
        version = (
            get_evaluator_version(ev["live_version_id"])
            if ev.get("live_version_id")
            else None
        )
        output_config = version.get("output_config") if version else None
        scale_min, scale_max = _scale_bounds(output_config)
        ev["output_config"] = output_config
        ev["scale_min"] = scale_min
        ev["scale_max"] = scale_max
        ev["variables"] = version.get("variables") if version else None
    items = get_job_items(job["uuid"])
    # Annotators open this form unauthenticated and may keep it open for hours;
    # sign TTS audio with a long TTL so playback doesn't die mid-session.
    presign_annotation_items_audio(items, task.get("type"))
    annotations = get_annotations_for_job(job["uuid"])

    return {
        "job": {
            "uuid": job["uuid"],
            "status": job["status"],
            "created_at": job["created_at"],
            "completed_at": job.get("completed_at"),
            "comments_enabled": bool(job.get("comments_enabled", 1)),
            "reasoning_mode": job_reasoning_mode,
        },
        "annotator": {
            "uuid": annotator["uuid"] if annotator else None,
            "name": annotator["name"] if annotator else None,
        },
        "task": {
            "uuid": task["uuid"],
            "name": task["name"],
            "type": task["type"],
            "description": task.get("description"),
        },
        "evaluators": evaluators,
        "items": items,
        "annotations": annotations,
        "read_only": read_only,
    }


@router.get("/annotation-jobs/view/{view_token}", summary="Get shared annotation job (read-only)")
def get_public_annotation_job_view(
    view_token: str = Path(description="Read-only view token for the annotation job"),
):
    """Get a read-only view of an annotator's labelling job"""
    job = get_annotation_job_by_view_token(view_token)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    return _build_annotation_job_payload(job, read_only=True)


@router.get("/annotation-jobs/{token}", summary="Get annotation job for annotator")
def get_public_annotation_job(
    token: str = Path(description="Annotator token for the labelling job"),
):
    """Get everything you need to render an annotator's labelling job page"""
    job = _resolve_public_annotation_job(token)
    return _build_annotation_job_payload(job, read_only=False)


class PublicAnnotationValue(BaseModel):
    """One judgement, and whatever the annotator wrote alongside it."""

    model_config = ConfigDict(extra="forbid")

    value: Optional[Union[StrictBool, StrictInt, StrictFloat]] = Field(
        None,
        description="Judgement itself: a bool for a binary evaluator, a number on the rubric's scale for a rating one",
        examples=[True],
    )
    reasoning: Optional[str] = Field(
        None, description="Why the annotator chose that judgement"
    )
    comment: Optional[str] = Field(
        None, description="A note on the row, carried by the row-level entry"
    )


class PublicAnnotationEntry(BaseModel):
    evaluator_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Evaluator ID this judgement is for. `null` marks a row-level overall annotation",
        examples=[_EXAMPLE_EVALUATOR_UUID],
    )
    value: Optional[PublicAnnotationValue] = Field(
        None, description="Judgement to record. `null` removes the stored judgement"
    )


class PublicAnnotationUpsertRequest(BaseModel):
    item_id: str = Field(
        min_length=36,
        max_length=36,
        description="Job item you are annotating",
        examples=[_EXAMPLE_ITEM_UUID],
    )
    annotations: List[PublicAnnotationEntry] = Field(
        description="Judgements to record, one entry per evaluator, plus optionally one row-level entry"
    )


@router.post("/annotation-jobs/{token}/annotations", summary="Upsert annotations for item")
def upsert_public_annotations(
    token: str = Path(description="Annotator token for the labelling job"),
    payload: PublicAnnotationUpsertRequest = ...,
):
    """Record all judgements for one item in a single request"""
    # The first save that stores something moves the job from `pending` to `in_progress`; all slots filled moves it to `completed`.
    job = _resolve_public_annotation_job(token)
    if not payload.annotations:
        raise HTTPException(
            status_code=400, detail="annotations must be non-empty"
        )

    # Validate against the job's snapshotted items (annotation_job_items), not
    # the source annotation_items row — the source may have been edited or
    # soft-deleted after the job was created, but the snapshot is what the
    # annotator is actually labeling.
    job_items = get_job_items(job["uuid"])
    if not any(it["uuid"] == payload.item_id for it in job_items):
        raise HTTPException(status_code=404, detail="Item not found in this job")

    # Validate that every non-null `evaluator_id` is in the job's snapshotted
    # evaluator set (NOT the task's current linked set — the contract is
    # frozen at job creation). Without this, a token holder could upsert
    # annotations against any evaluator UUID — polluting downstream agreement
    # aggregates that read `annotations` joined through `annotation_jobs`.
    # `evaluator_id IS NULL` is the row-level overall annotation case and is
    # always allowed.
    linked_evaluator_ids = set(get_evaluator_ids_for_job(job["uuid"]))
    invalid = [
        entry.evaluator_id
        for entry in payload.annotations
        if entry.evaluator_id is not None
        and entry.evaluator_id not in linked_evaluator_ids
    ]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Evaluator(s) not linked to this task: {invalid}",
        )

    saved_uuids: List[str] = []
    for entry in payload.annotations:
        annotation_uuid = upsert_annotation(
            job_id=job["uuid"],
            item_id=payload.item_id,
            evaluator_id=entry.evaluator_id,
            value=(
                entry.value.model_dump(exclude_unset=True)
                if entry.value is not None
                else None
            ),
        )
        if annotation_uuid:
            saved_uuids.append(annotation_uuid)

    if job["status"] == "pending" and saved_uuids:
        update_annotation_job_status(job["uuid"], "in_progress")

    # Auto-complete: every required slot each item actually shows must have a
    # row, and every item must carry at least one judgement. A tool-call row
    # shows only the tool-call evaluator, every other row shows everything
    # except it (required_evaluator_ids_for_item). The second condition is
    # what makes a job whose evaluators are all optional still
    # require the annotator to visit each item, rather than completing on the
    # first save. We re-check on every save (including post-completion edits)
    # so the status remains accurate: removing a required answer moves a
    # completed job back to `in_progress` and clears `completed_at`, which is
    # set again when the job is next fully filled.
    # Both the items AND the evaluator set are read from the job's snapshot
    # so post-creation link/unlink on the parent task can't shift the
    # completion bar under the annotator.
    job_items = get_job_items(job["uuid"])
    required_evaluator_ids = get_evaluator_ids_for_job(
        job["uuid"], required_only=True
    )
    tool_call_evaluator_ids = {
        ev["uuid"]
        for ev in get_evaluators_for_job(job["uuid"])
        if ev.get("evaluator_type") == TOOL_CALL_EVALUATOR_TYPE
    }
    job_annotations = get_annotations_for_job(job["uuid"])
    annotated_pairs = {
        (a["item_id"], a.get("evaluator_id"))
        for a in job_annotations
        if a.get("evaluator_id") is not None
    }
    expected_pairs = {
        (it["uuid"], ev_id)
        for it in job_items
        for ev_id in required_evaluator_ids_for_item(
            it, required_evaluator_ids, tool_call_evaluator_ids
        )
    }
    items_touched = {a["item_id"] for a in job_annotations}
    completed = (
        bool(job_items)
        and expected_pairs.issubset(annotated_pairs)
        and all(it["uuid"] in items_touched for it in job_items)
    )
    if completed and job["status"] != "completed":
        update_annotation_job_status(
            job["uuid"], "completed", set_completed_at=True
        )
    elif not completed and job["status"] == "completed":
        update_annotation_job_status(
            job["uuid"], "in_progress", clear_completed_at=True
        )

    if completed:
        final_status = "completed"
    elif job["status"] == "pending" and not saved_uuids:
        final_status = "pending"
    else:
        final_status = "in_progress"
    return {
        "saved": saved_uuids,
        "count": len(saved_uuids),
        "status": final_status,
    }


class PublicInviteResponse(BaseModel):
    organization_name: str = Field(description="Name of the workspace you were invited to")


@router.get(
    "/invite/{token}",
    response_model=PublicInviteResponse,
    summary="Get invite",
)
def get_public_invite(
    token: str = Path(
        description="Token from the invite link you were sent",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
):
    """Get the name of the workspace an invite link points at"""
    org = get_org_by_invite_token(token)
    if org is None:
        raise HTTPException(status_code=404, detail="Invite not found")
    return PublicInviteResponse(organization_name=org["name"])


CHAT_REPLY_TIMEOUT_SECONDS = 60
MAX_CHAT_TURNS = 500
# How many times one chat reply may go back to the model after running tools.
DEFAULT_CHAT_MAX_TOOL_ROUNDS = 5
_CHAT_TOKEN_DESCRIPTION = "Token from the chat link you were sent"


def _chat_error(status_code: int, error: str, **extra: Any) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"error": error, **extra})


def _chat_link_agent(token: str, user_id: Optional[str]) -> Dict[str, Any]:
    """The agent behind the token, when its link is public or the caller is a
    logged-in member of its workspace previewing a private link."""
    agent = get_agent_by_chat_token(token)
    if agent is None or not (
        agent.get("chat_link_public")
        or (user_id and get_member_role(agent["org_uuid"], user_id))
    ):
        raise _chat_error(404, "not_found")
    return agent


async def _agent_reply(chat: Dict[str, Any]) -> Dict[str, Any]:
    """`{response, tool_calls, stopped_early, new_messages}` from the agent, tool
    calls in the trace shape. `stopped_early` is true when it used every tool
    round it is allowed and was cut off before replying. `new_messages` is its
    turn as the next message of the conversation will need it."""
    if env_bool("FAKE_AI_PROVIDERS", False):
        return {
            "response": FAKE_RESPONSE,
            "tool_calls": [],
            "new_messages": [{"role": "assistant", "content": FAKE_RESPONSE}],
        }
    agent = chat["agent"]
    config = agent.get("config") or {}
    if "llm" in chat:  # set by _prepare_chat_reply for an agent built in Calibrate
        result = await agent_reply(messages=chat["messages"], **chat["llm"])
        return {
            "response": result["reply"] or None,
            "tool_calls": result["tool_calls"],
            "stopped_early": result.get("stopped") == "max_tool_rounds",
            "new_messages": result["new_messages"],
        }
    error = await run_in_threadpool(public_url_error, config["agent_url"])
    if error:
        raise ValueError(f"agent_url {error}")
    connection = TextAgentConnection(
        url=config["agent_url"],
        headers=_sanitize_headers(with_calibrate_eval_header(config.get("agent_headers"))),
        default_inputs=config.get("default_inputs"),
        type=agent.get("interaction_type") or DEFAULT_AGENT_INTERACTION_TYPE,
    )
    # A connected agent takes plain turns, as in a simulation; its own tool
    # calls stay on its side, so they are left out of what it is sent.
    turns = [
        {"role": m["role"], "content": m["content"]}
        for m in chat["messages"]
        if m["role"] in ("user", "assistant") and m.get("content")
    ]
    reply = await connection.call(turns)
    return {**reply, "new_messages": [{"role": "assistant", "content": reply.get("response")}]}


def _reserve_daily_reply(agent: Dict[str, Any], cap: int) -> None:
    """Take one of today's replies before the agent is asked, or refuse."""
    if not reserve_chat_reply(agent["uuid"], cap):
        today = count_chat_replies_today(agent["uuid"])
        record_limit_hit(agent["org_uuid"], "max_chat_replies_per_day", "chat", today + 1, cap)
        raise _chat_error(429, "daily_limit", limit=cap)


def _prepare_chat_reply(
    token: str, body: "ChatMessageRequest", user_id: Optional[str]
) -> Dict[str, Any]:
    """The checks and reads before the agent is asked, in the order their errors
    are answered. The daily limit is checked later, when the reply is counted."""
    agent = _chat_link_agent(token, user_id)
    org_uuid = agent["org_uuid"]
    limits = effective_limits(org_uuid)

    # Checked when the reply is counted, just before the agent is asked.
    daily_cap = daily_message_limit(agent, limits["max_chat_replies_per_day"])

    # A workspace full of traces still gets its reply; it is just not stored.
    trace_cap = limits["max_traces"]
    current = count_live_traces(org_uuid)
    store_trace = current < trace_cap
    if not store_trace:
        record_limit_hit(org_uuid, "max_traces", "trace", current + 1, trace_cap)

    user_message = {"role": "user", "content": body.message}
    if agent.get("interaction_type") == "general":
        messages, trace_input = [user_message], body.message
    else:
        # Kept by the backend, never sent by the caller, so earlier turns cannot
        # be made up, and kept apart from traces so a full workspace keeps them.
        history = (
            get_chat_conversation(agent["uuid"], body.conversation_id) if body.conversation_id else []
        )
        messages = history + [user_message]
        if len(messages) > MAX_CHAT_TURNS:
            raise _chat_error(409, "conversation_too_long")
        trace_input = messages

    chat = {
        "agent": agent,
        "limits": limits,
        "messages": messages,
        "trace_input": trace_input,
        "conversation_id": body.conversation_id or str(uuid.uuid4()),
        "store_trace": store_trace,
        "daily_cap": daily_cap,
    }
    config = agent.get("config") or {}
    if not config.get("agent_url"):
        llm = config.get("llm", {})
        provider = llm.get("provider", "openrouter")
        # The workspace's own keys when it has any, else the server's, the
        # same choice a test run makes.
        org_env = provider_env(org_uuid)
        env_var = PROVIDER_ENV_VARS.get(provider, ("",))[0]
        api_key = (org_env if org_env is not None else os.environ).get(env_var)
        if org_env is not None and not api_key:
            raise _chat_error(503, "agent_unavailable")
        chat["llm"] = {
            "system_prompt": config.get("system_prompt", ""),
            "tools": build_tool_configs(get_tools_for_agent(agent["uuid"])),
            "model": llm.get("model", default_agent_llm_model()),
            "provider": provider,
            "api_key": api_key,
            "max_tool_rounds": max(1, env_int("CHAT_MAX_TOOL_ROUNDS", DEFAULT_CHAT_MAX_TOOL_ROUNDS)),
        }
    return chat


class PublicAgentChatResponse(BaseModel):
    agent_name: str = Field(description="Name of the agent you can chat with")
    interaction_type: Literal["conversation", "general"] = Field(
        description="`conversation` for an agent that talks back and forth, `general` for one that answers each message on its own"
    )


class ChatToolCall(BaseModel):
    tool: str = Field(description="Name of the tool the agent called")
    arguments: Any = Field(None, description="What the agent passed to the tool")
    output: Any = Field(None, description="What the tool gave back")


class ChatMessageRequest(BaseModel):
    conversation_id: Optional[str] = Field(
        None,
        max_length=200,
        description="Conversation to continue, as returned by an earlier reply. Omit to start a new one",
    )
    message: str = Field(min_length=1, max_length=4000, description="What you say to the agent")


class ChatMessageResponse(BaseModel):
    conversation_id: str = Field(description="Conversation this reply belongs to. Send it with your next message")
    trace_uuid: Optional[str] = Field(
        None,
        description="ID of the stored reply, for leaving feedback on it. Missing when the workspace holds as many traces as it may, so the reply was not stored",
    )
    reply: Optional[str] = Field(None, description="What the agent said")
    tool_calls: List[ChatToolCall] = Field(description="Tools the agent called before replying")
    ended: bool = Field(description="Whether the agent ended the conversation")
    stopped_early: bool = Field(
        False,
        description="Whether the agent was stopped after calling tools as many times in a row as it may, before it replied",
    )


@router.get(
    "/agent-chat/{token}",
    response_model=PublicAgentChatResponse,
    summary="Get chat link",
)
def get_public_agent_chat(
    token: str = Path(description=_CHAT_TOKEN_DESCRIPTION, examples=[_EXAMPLE_JOB_UUID]),
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    """Get the agent a chat link lets you talk to"""
    agent = _chat_link_agent(token, user_id)
    return PublicAgentChatResponse(
        agent_name=agent["name"],
        interaction_type=agent.get("interaction_type") or DEFAULT_AGENT_INTERACTION_TYPE,
    )


@router.post(
    "/agent-chat/{token}/messages",
    response_model=ChatMessageResponse,
    summary="Send chat message",
)
async def send_public_agent_chat_message(
    token: str = Path(description=_CHAT_TOKEN_DESCRIPTION, examples=[_EXAMPLE_JOB_UUID]),
    body: ChatMessageRequest = ...,
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    """Send a message to the agent behind a chat link and get its reply"""
    chat = await run_in_threadpool(_prepare_chat_reply, token, body, user_id)
    agent = chat["agent"]
    await run_in_threadpool(_reserve_daily_reply, agent, chat["daily_cap"])
    try:
        result = await asyncio.wait_for(_agent_reply(chat), CHAT_REPLY_TIMEOUT_SECONDS)
    except BaseException as exc:
        # Also when the person leaves mid-reply, which cancels the request:
        # a reply nobody got is not counted against the day.
        release_chat_reply(agent["uuid"])
        if not isinstance(exc, Exception):
            raise
        await run_in_threadpool(capture_exception_to_sentry, exc)
        raise _chat_error(502, "agent_failed")

    tool_calls = [
        {key: call[key] for key in ("tool", "arguments", "output") if key in call}
        for call in result.get("tool_calls") or []
    ]
    if agent.get("interaction_type") != "general":
        await run_in_threadpool(
            save_chat_conversation,
            agent["uuid"],
            chat["conversation_id"],
            chat["messages"] + result["new_messages"],
        )
    row = None
    if chat["store_trace"]:
        row = await run_in_threadpool(
            lambda: create_trace_with_eval_run(
                org_uuid=agent["org_uuid"],
                agent=agent,
                max_scored_traces=chat["limits"]["max_scored_traces"],
                batch_size=chat["limits"]["trace_scoring_batch_size"],
                conversation_id=chat["conversation_id"],
                input=chat["trace_input"],
                output={
                    "response": result.get("response"),
                    "tool_calls": [call for call in tool_calls if call["tool"] != "end_call"],
                    **({"stopped_early": True} if result.get("stopped_early") else {}),
                },
                source="chat",
            )
        )
    return ChatMessageResponse(
        conversation_id=chat["conversation_id"],
        trace_uuid=row["uuid"] if row else None,
        reply=result.get("response"),
        tool_calls=tool_calls,
        ended=any(call["tool"] == "end_call" for call in tool_calls),
        stopped_early=bool(result.get("stopped_early")),
    )


@router.put(
    "/agent-chat/{token}/traces/{trace_uuid}/feedback",
    response_model=TraceFeedbackResponse,
    summary="Set chat reply feedback",
)
def set_public_agent_chat_feedback(
    token: str = Path(description=_CHAT_TOKEN_DESCRIPTION, examples=[_EXAMPLE_JOB_UUID]),
    trace_uuid: str = Path(description="The reply to leave feedback on", examples=[_EXAMPLE_ITEM_UUID]),
    body: TraceFeedbackUpdate = ...,
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    """Rate a reply from the agent behind a chat link, or clear the rating"""
    agent = _chat_link_agent(token, user_id)
    trace = get_trace(agent["org_uuid"], trace_uuid)
    if (
        trace is None
        or trace["agent_id"] != agent["uuid"]
        or trace["source"] != "chat"
        or not set_trace_feedback(agent["org_uuid"], trace_uuid, body.is_correct, body.comment)
    ):
        raise _chat_error(404, "not_found")
    return {"feedback": get_trace(agent["org_uuid"], trace_uuid)["feedback"]}
