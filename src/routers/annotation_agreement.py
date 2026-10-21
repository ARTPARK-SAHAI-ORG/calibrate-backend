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
    agreement_between_annotators,
    agreement_between_judge_and_annotators,
    keep_runs_from_live_versions_of,
    rating_scales_for,
    agreement_between_annotators_over_time,
    agreement_between_judge_and_annotators_over_time_by_version_and_task,
    agreement_between_judge_and_annotators_over_time,
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
        all_evaluator_version_runs = get_evaluator_runs_for_task(task_id)
    else:
        annotations = get_annotations_for_org(ctx.org_uuid)
        all_evaluator_version_runs = get_evaluator_runs_for_org(ctx.org_uuid)

    rating_scales = rating_scales_for(annotations, all_evaluator_version_runs)
    annotator_agreement, annotator_pair_count = agreement_between_annotators(
        annotations, rating_scales
    )
    annotator_agreement_over_time = agreement_between_annotators_over_time(
        annotations, bucket=bucket, days=days, rating_scales=rating_scales
    )

    evaluator_ids = list(
        dict.fromkeys(
            run["evaluator_id"]
            for run in all_evaluator_version_runs
            if run.get("evaluator_id")
        )
    )

    evaluator_id_to_evaluator = get_evaluators_by_uuids(evaluator_ids)
    live_evaluator_version_runs = keep_runs_from_live_versions_of(
        all_evaluator_version_runs, evaluator_id_to_evaluator.values()
    )

    evaluator_id_to_agreement_over_time = (
        agreement_between_judge_and_annotators_over_time(
            annotations,
            live_evaluator_version_runs,
            evaluator_ids,
            bucket=bucket,
            days=days,
            rating_scales=rating_scales,
        )
    )
    agreement_per_evaluator = []
    for evaluator_id in evaluator_ids:
        evaluator = evaluator_id_to_evaluator.get(evaluator_id) or {}
        evaluator_agreement, evaluator_pair_count = (
            agreement_between_judge_and_annotators(
                annotations, live_evaluator_version_runs, evaluator_id, rating_scales
            )
        )
        agreement_per_evaluator.append(
            {
                "evaluator_id": evaluator_id,
                "name": evaluator.get("name"),
                "current": evaluator_agreement,
                "pair_count": evaluator_pair_count,
                "series": evaluator_id_to_agreement_over_time.get(evaluator_id, []),
            }
        )

    return {
        "bucket": bucket,
        "days": days,
        "task_id": task_id,
        "human_human": {
            "current": annotator_agreement,
            "pair_count": annotator_pair_count,
            "series": annotator_agreement_over_time,
        },
        "evaluators": agreement_per_evaluator,
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

    evaluator_versions = (
        get_evaluator_versions(evaluator_uuid, include_deleted=True)
        if evaluator_runs
        else []
    )
    version_id_to_version_number = {
        evaluator_version["uuid"]: evaluator_version["version_number"]
        for evaluator_version in evaluator_versions
    }
    live_version_id = evaluator.get("live_version_id")

    version_ids_with_scores = list(
        dict.fromkeys(
            run["evaluator_version_id"]
            for run in evaluator_runs
            if run.get("evaluator_version_id")
        )
    )
    task_ids_with_scores = list(
        dict.fromkeys(run["task_id"] for run in evaluator_runs if run.get("task_id"))
    )

    agreement_over_time = (
        agreement_between_judge_and_annotators_over_time_by_version_and_task(
            annotations,
            evaluator_runs,
            evaluator_uuid,
            version_ids=version_ids_with_scores,
            task_ids=task_ids_with_scores,
            bucket=bucket,
            days=days,
            rating_scales=rating_scales_for(annotations, evaluator_runs),
        )
    )

    def _latest_agreement_and_pair_count(
        agreement_per_bucket: List[Dict[str, Any]],
    ) -> tuple:
        if not agreement_per_bucket:
            return None, 0
        latest_bucket = agreement_per_bucket[-1]
        return latest_bucket["agreement"], latest_bucket["pair_count"]

    task_id_to_task = get_annotation_tasks_by_uuids(task_ids_with_scores)

    agreement_per_version: List[Dict[str, Any]] = []
    for scored_version_id in version_ids_with_scores:
        version_agreement_per_bucket = agreement_over_time["by_version"][
            scored_version_id
        ]
        agreement, pair_count = _latest_agreement_and_pair_count(
            version_agreement_per_bucket
        )
        agreement_per_version.append(
            {
                "version_id": scored_version_id,
                "version_number": version_id_to_version_number.get(scored_version_id),
                "is_live": scored_version_id == live_version_id,
                "current": agreement,
                "pair_count": pair_count,
                "series": version_agreement_per_bucket,
            }
        )

    agreement_per_task: List[Dict[str, Any]] = []
    for scored_task_id in task_ids_with_scores:
        task_agreement_per_bucket = agreement_over_time["by_task"][scored_task_id]
        agreement, pair_count = _latest_agreement_and_pair_count(
            task_agreement_per_bucket
        )
        scored_task = task_id_to_task.get(scored_task_id) or {}
        agreement_per_task.append(
            {
                "task_id": scored_task_id,
                "task_name": scored_task.get("name"),
                "current": agreement,
                "pair_count": pair_count,
                "series": task_agreement_per_bucket,
            }
        )

    overall_agreement_per_bucket = agreement_over_time["overall"]
    overall_agreement, overall_pair_count = _latest_agreement_and_pair_count(
        overall_agreement_per_bucket
    )

    return {
        "evaluator_id": evaluator_uuid,
        "evaluator_name": evaluator.get("name"),
        "bucket": bucket,
        "days": days,
        "filters": {"task_id": task_id, "version_id": version_id},
        "overall": {
            "current": overall_agreement,
            "pair_count": overall_pair_count,
            "series": overall_agreement_per_bucket,
        },
        "versions": agreement_per_version,
        "tasks": agreement_per_task,
    }
