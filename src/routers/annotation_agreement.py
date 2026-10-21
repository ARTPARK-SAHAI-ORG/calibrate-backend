from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from db import (
    get_annotation_task,
    get_annotation_tasks_by_uuids,
    get_annotations_for_task,
    get_annotations_for_org,
    get_evaluator_runs_for_evaluator_org_scoped,
    get_evaluator_runs_for_task,
    get_evaluator_runs_for_org,
    get_evaluator_versions,
    get_evaluators_by_uuids,
)
from auth_utils import get_current_org, OrgContext
from org_scope import ensure_owned_evaluator
from annotation_metrics import (
    aggregate_agreement,
    aggregate_human_evaluator_agreement,
    runs_at_live_versions,
    scales_for,
    trend_series,
    trend_series_evaluator_breakdown,
    trend_series_human_evaluator,
)


router = APIRouter(prefix="/annotation-agreement", tags=["annotation-agreement"])


@router.get("/trend", summary="Get workspace agreement trend")
def agreement_trend(
    bucket: str = Query(
        "week",
        pattern="^(week|month|year)$",
        description="Time bucket for the trend series (`week`, `month`, or `year`)",
    ),
    days: int = Query(
        90, ge=1, le=3650, description="Trailing window in days for the trend series"
    ),
    task_id: Optional[str] = Query(
        None,
        description="Annotation task to scope metrics to. Omit for workspace-wide trends",
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get human-vs-human agreement trends and human alignment for each evaluator"""
    if task_id:
        task = get_annotation_task(task_id)
        if not task or task.get("org_uuid") != ctx.org_uuid:
            raise HTTPException(status_code=404, detail="Annotation task not found")
        annotations = get_annotations_for_task(task_id)
        all_version_runs = get_evaluator_runs_for_task(task_id)
    else:
        annotations = get_annotations_for_org(ctx.org_uuid)
        all_version_runs = get_evaluator_runs_for_org(ctx.org_uuid)

    scales = scales_for(annotations, all_version_runs)
    annotator_agreement, annotator_pair_count = aggregate_agreement(annotations, scales)
    annotator_agreement_trend = trend_series(
        annotations, bucket=bucket, days=days, scales=scales
    )

    evaluator_ids = list(
        dict.fromkeys(
            r["evaluator_id"] for r in all_version_runs if r.get("evaluator_id")
        )
    )

    evaluator_meta = get_evaluators_by_uuids(evaluator_ids)
    live_version_runs = runs_at_live_versions(all_version_runs, evaluator_meta.values())

    evaluator_trends = trend_series_human_evaluator(
        annotations,
        live_version_runs,
        evaluator_ids,
        bucket=bucket,
        days=days,
        scales=scales,
    )
    evaluators_block = []
    for ev_id in evaluator_ids:
        ev = evaluator_meta.get(ev_id) or {}
        evaluator_agreement, evaluator_pair_count = aggregate_human_evaluator_agreement(
            annotations, live_version_runs, ev_id, scales
        )
        evaluators_block.append(
            {
                "evaluator_id": ev_id,
                "name": ev.get("name"),
                "current": evaluator_agreement,
                "pair_count": evaluator_pair_count,
                "series": evaluator_trends.get(ev_id, []),
            }
        )

    return {
        "bucket": bucket,
        "days": days,
        "task_id": task_id,
        "human_human": {
            "current": annotator_agreement,
            "pair_count": annotator_pair_count,
            "series": annotator_agreement_trend,
        },
        "evaluators": evaluators_block,
    }


@router.get("/evaluator/{evaluator_uuid}/trend", summary="Get evaluator agreement trend")
def evaluator_agreement_trend(
    evaluator_uuid: str = Path(
        description="Evaluator to chart agreement for",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    bucket: str = Query(
        "week",
        pattern="^(week|month|year)$",
        description="Time bucket for the trend series (`week`, `month`, or `year`)",
    ),
    days: int = Query(
        90, ge=1, le=3650, description="Trailing window in days for the trend series"
    ),
    task_id: Optional[str] = Query(
        None,
        description="Annotation task to scope metrics to. Omit to include all tasks",
    ),
    version_id: Optional[str] = Query(
        None,
        description="Evaluator version to scope metrics to. Omit to include all versions",
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get human-vs-evaluator agreement trends for one evaluator, broken down by version and task"""
    evaluator = ensure_owned_evaluator(evaluator_uuid, ctx.org_uuid)

    if task_id:
        task = get_annotation_task(task_id)
        if not task or task.get("org_uuid") != ctx.org_uuid:
            raise HTTPException(status_code=404, detail="Annotation task not found")
        annotations = get_annotations_for_task(task_id)
    else:
        annotations = get_annotations_for_org(ctx.org_uuid)

    evaluator_runs = get_evaluator_runs_for_evaluator_org_scoped(
        evaluator_uuid, ctx.org_uuid, task_id=task_id, version_id=version_id
    )

    all_versions = (
        get_evaluator_versions(evaluator_uuid, include_deleted=True)
        if evaluator_runs
        else []
    )
    version_number_by_id = {v["uuid"]: v["version_number"] for v in all_versions}
    live_version_id = evaluator.get("live_version_id")

    seen_versions = list(
        dict.fromkeys(
            r["evaluator_version_id"]
            for r in evaluator_runs
            if r.get("evaluator_version_id")
        )
    )
    seen_tasks = list(
        dict.fromkeys(r["task_id"] for r in evaluator_runs if r.get("task_id"))
    )

    breakdown = trend_series_evaluator_breakdown(
        annotations, evaluator_runs, evaluator_uuid,
        version_ids=seen_versions,
        task_ids=seen_tasks,
        bucket=bucket,
        days=days,
        scales=scales_for(annotations, evaluator_runs),
    )

    def _last(series: List[Dict[str, Any]]) -> tuple:
        if not series:
            return None, 0
        last = series[-1]
        return last["agreement"], last["pair_count"]

    task_meta_by_id = get_annotation_tasks_by_uuids(seen_tasks)

    versions_block: List[Dict[str, Any]] = []
    for vid in seen_versions:
        series = breakdown["by_version"][vid]
        agreement, pair_count = _last(series)
        versions_block.append(
            {
                "version_id": vid,
                "version_number": version_number_by_id.get(vid),
                "is_live": vid == live_version_id,
                "current": agreement,
                "pair_count": pair_count,
                "series": series,
            }
        )

    tasks_block: List[Dict[str, Any]] = []
    for tid in seen_tasks:
        series = breakdown["by_task"][tid]
        agreement, pair_count = _last(series)
        task_meta = task_meta_by_id.get(tid) or {}
        tasks_block.append(
            {
                "task_id": tid,
                "task_name": task_meta.get("name"),
                "current": agreement,
                "pair_count": pair_count,
                "series": series,
            }
        )

    overall_series = breakdown["overall"]
    overall_agreement, overall_pair_count = _last(overall_series)

    return {
        "evaluator_id": evaluator_uuid,
        "evaluator_name": evaluator.get("name"),
        "bucket": bucket,
        "days": days,
        "filters": {"task_id": task_id, "version_id": version_id},
        "overall": {
            "current": overall_agreement,
            "pair_count": overall_pair_count,
            "series": overall_series,
        },
        "versions": versions_block,
        "tasks": tasks_block,
    }
