from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query

from db import (
    get_annotations_for_user,
    get_evaluators_by_uuids,
    get_evaluator_runs_for_user,
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
    user_id: str = Depends(get_current_user_id),
):
    """Account-wide human-vs-human agreement trend across all of the user's
    annotation tasks, plus per-evaluator human-vs-evaluator alignment for
    every evaluator that has produced at least one run on the user's data.

    Returns:
      - `human_human`: `{ current, pair_count, series }`.
      - `evaluators`: list of `{ evaluator_id, name, current, pair_count, series }`,
        one per evaluator that's been run at least once on this account's data.
    """
    annotations = get_annotations_for_user(user_id)
    raw_runs = get_evaluator_runs_for_user(user_id)

    hh_current, hh_pairs = aggregate_agreement(annotations)
    hh_series = trend_series(annotations, bucket=bucket, days=days)

    # Distinct evaluator_ids that have produced runs on the user's data — that's
    # the natural set to include in the account-wide rollup. Names are resolved
    # live from the evaluators table so renames show up. We also resolve each
    # evaluator's live_version_id to filter runs down to live-version only;
    # non-live experimental runs stay in the DB but don't contribute to the
    # account-wide "evaluator agreement" number.
    evaluator_ids = []
    seen = set()
    for r in raw_runs:
        ev_id = r.get("evaluator_id")
        if ev_id and ev_id not in seen:
            seen.add(ev_id)
            evaluator_ids.append(ev_id)

    # One bulk fetch instead of N round-trips through `get_evaluator`.
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
        "human_human": {
            "current": hh_current,
            "pair_count": hh_pairs,
            "series": hh_series,
        },
        "evaluators": evaluators_block,
    }
