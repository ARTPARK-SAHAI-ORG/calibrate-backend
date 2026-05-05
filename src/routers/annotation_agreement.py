from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from db import (
    get_annotation_task,
    get_annotations_for_task,
    get_annotations_for_user,
    get_evaluator,
    get_evaluator_runs_for_evaluator_user_scoped,
    get_evaluator_runs_for_task,
    get_evaluator_runs_for_user,
    get_evaluator_versions,
    get_evaluators_by_uuids,
)
from auth_utils import get_current_user_id
from annotation_metrics import (
    aggregate_agreement,
    aggregate_human_evaluator_agreement,
    trend_series,
    trend_series_human_evaluator,
    filter_runs_to_live_versions,
)


router = APIRouter(prefix="/annotation-agreement", tags=["annotation-agreement"])


@router.get("/trend")
async def agreement_trend(
    bucket: str = Query("week", pattern="^(week|month|year)$"),
    days: int = Query(90, ge=1, le=3650),
    task_id: Optional[str] = Query(None),
    user_id: str = Depends(get_current_user_id),
):
    """Account-wide human-vs-human agreement trend, plus per-evaluator
    human-vs-evaluator alignment for every evaluator that has produced at
    least one run on the user's data.

    Pass `task_id` to restrict all metrics to a single annotation task.

    Returns:
      - `human_human`: `{ current, pair_count, series }`.
      - `evaluators`: list of `{ evaluator_id, name, current, pair_count, series }`.
    """
    if task_id:
        task = get_annotation_task(task_id)
        if not task or task.get("user_id") != user_id:
            raise HTTPException(status_code=404, detail="Annotation task not found")
        annotations = get_annotations_for_task(task_id)
        raw_runs = get_evaluator_runs_for_task(task_id)
    else:
        annotations = get_annotations_for_user(user_id)
        raw_runs = get_evaluator_runs_for_user(user_id)

    hh_current, hh_pairs = aggregate_agreement(annotations)
    hh_series = trend_series(annotations, bucket=bucket, days=days)

    evaluator_ids = []
    seen = set()
    for r in raw_runs:
        ev_id = r.get("evaluator_id")
        if ev_id and ev_id not in seen:
            seen.add(ev_id)
            evaluator_ids.append(ev_id)

    evaluator_meta = get_evaluators_by_uuids(evaluator_ids)
    live_version_by_evaluator: Dict[str, Optional[str]] = {
        ev_id: (evaluator_meta.get(ev_id) or {}).get("live_version_id")
        for ev_id in evaluator_ids
    }

    runs = filter_runs_to_live_versions(raw_runs, live_version_by_evaluator)

    series_by_id = trend_series_human_evaluator(
        annotations, runs, evaluator_ids, bucket=bucket, days=days
    )
    evaluators_block = []
    for ev_id in evaluator_ids:
        ev = evaluator_meta.get(ev_id) or {}
        cur, pairs = aggregate_human_evaluator_agreement(annotations, runs, ev_id)
        evaluators_block.append(
            {
                "evaluator_id": ev_id,
                "name": ev.get("name"),
                "current": cur,
                "pair_count": pairs,
                "series": series_by_id.get(ev_id, []),
            }
        )

    return {
        "bucket": bucket,
        "days": days,
        "task_id": task_id,
        "human_human": {
            "current": hh_current,
            "pair_count": hh_pairs,
            "series": hh_series,
        },
        "evaluators": evaluators_block,
    }


@router.get("/evaluator/{evaluator_uuid}/trend")
async def evaluator_agreement_trend(
    evaluator_uuid: str,
    bucket: str = Query("week", pattern="^(week|month|year)$"),
    days: int = Query(90, ge=1, le=3650),
    task_id: Optional[str] = Query(None),
    version_id: Optional[str] = Query(None),
    user_id: str = Depends(get_current_user_id),
):
    """Human-vs-evaluator agreement trend for one evaluator, broken down by
    version and by annotation task.

    Optional filters:
      - `task_id`: restrict all metrics to items in a single task.
      - `version_id`: restrict all metrics to a single evaluator version.

    Returns:
      - `overall`: `{ current, pair_count, series }` across all matching runs.
      - `versions`: list of `{ version_id, version_number, is_live, current, pair_count, series }`.
      - `tasks`: list of `{ task_id, task_name, current, pair_count, series }`.
    """
    evaluator = get_evaluator(evaluator_uuid)
    if not evaluator:
        raise HTTPException(status_code=404, detail="Evaluator not found")

    if task_id:
        task = get_annotation_task(task_id)
        if not task or task.get("user_id") != user_id:
            raise HTTPException(status_code=404, detail="Annotation task not found")
        annotations = get_annotations_for_task(task_id)
    else:
        annotations = get_annotations_for_user(user_id)

    runs = get_evaluator_runs_for_evaluator_user_scoped(
        evaluator_uuid, user_id, task_id=task_id, version_id=version_id
    )

    # ── Overall ──────────────────────────────────────────────────────────────
    overall_cur, overall_pairs = aggregate_human_evaluator_agreement(
        annotations, runs, evaluator_uuid
    )
    overall_series = trend_series_human_evaluator(
        annotations, runs, [evaluator_uuid], bucket=bucket, days=days
    ).get(evaluator_uuid, [])

    # ── Per-version breakdown ─────────────────────────────────────────────────
    all_versions = get_evaluator_versions(evaluator_uuid)
    version_number_by_id = {v["uuid"]: v["version_number"] for v in all_versions}
    live_version_id = evaluator.get("live_version_id")

    seen_versions: List[str] = []
    seen_v_set: set = set()
    for r in runs:
        vid = r.get("evaluator_version_id")
        if vid and vid not in seen_v_set:
            seen_v_set.add(vid)
            seen_versions.append(vid)

    versions_block: List[Dict[str, Any]] = []
    for vid in seen_versions:
        v_runs = [r for r in runs if r.get("evaluator_version_id") == vid]
        cur, pairs = aggregate_human_evaluator_agreement(annotations, v_runs, evaluator_uuid)
        series = trend_series_human_evaluator(
            annotations, v_runs, [evaluator_uuid], bucket=bucket, days=days
        ).get(evaluator_uuid, [])
        versions_block.append(
            {
                "version_id": vid,
                "version_number": version_number_by_id.get(vid),
                "is_live": vid == live_version_id,
                "current": cur,
                "pair_count": pairs,
                "series": series,
            }
        )

    # ── Per-task breakdown ────────────────────────────────────────────────────
    seen_tasks: List[str] = []
    seen_t_set: set = set()
    for r in runs:
        tid = r.get("task_id")
        if tid and tid not in seen_t_set:
            seen_t_set.add(tid)
            seen_tasks.append(tid)

    tasks_block: List[Dict[str, Any]] = []
    for tid in seen_tasks:
        t_runs = [r for r in runs if r.get("task_id") == tid]
        t_annotations = [a for a in annotations if a.get("task_id") == tid]
        cur, pairs = aggregate_human_evaluator_agreement(t_annotations, t_runs, evaluator_uuid)
        series = trend_series_human_evaluator(
            t_annotations, t_runs, [evaluator_uuid], bucket=bucket, days=days
        ).get(evaluator_uuid, [])
        task_meta = get_annotation_task(tid) or {}
        tasks_block.append(
            {
                "task_id": tid,
                "task_name": task_meta.get("name"),
                "current": cur,
                "pair_count": pairs,
                "series": series,
            }
        )

    return {
        "evaluator_id": evaluator_uuid,
        "evaluator_name": evaluator.get("name"),
        "bucket": bucket,
        "days": days,
        "filters": {"task_id": task_id, "version_id": version_id},
        "overall": {
            "current": overall_cur,
            "pair_count": overall_pairs,
            "series": overall_series,
        },
        "versions": versions_block,
        "tasks": tasks_block,
    }
