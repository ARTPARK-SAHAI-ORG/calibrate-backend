"""Human-vs-human and human-vs-evaluator agreement metrics for annotation tasks.

Pairwise mean agreement on shared rows. Each (item, evaluator_id) is treated as
its own annotation slot; a row-level overall annotation has evaluator_id=None.

For binary judgements: agreement = 1.0 if equal else 0.0.
For numeric (rating) judgements: agreement = 1 - |a - b| / divider, where the
divider is the rubric's width (scale_max - scale_min): the width of the version
a judge score came from, or of the evaluator's live version for human labels,
which record no version. It is never narrower than the values compared, so a
label given on an older, wider scale cannot push agreement below 0.

For mixed types within a slot, only same-type pairs contribute.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Rubric bounds as `(scale_min, scale_max)`, keyed by evaluator uuid (its live
# version) and by version uuid (the version a judge score came from).
ScaleBounds = Tuple[Optional[float], Optional[float]]
Scales = Dict[str, ScaleBounds]


# All agreement floats returned to callers are rounded to this many decimal
# digits so the API surface is stable and free of float-formatting noise.
_AGREEMENT_DECIMALS = 4


def _round_agreement(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), _AGREEMENT_DECIMALS)


def _scalar(value: Any) -> Any:
    """Extract the comparable payload from an annotation `value` JSON.

    Conventions accepted: a bare bool/number/string, or a dict with one of
    {value, score, rating, label, binary}. Comments and other keys are ignored.
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        for key in ("value", "score", "rating", "label", "binary"):
            if key in value and value[key] is not None:
                return value[key]
    return None


def _classify(scalar: Any) -> Optional[str]:
    if scalar is None:
        return None
    if isinstance(scalar, bool):
        return "binary"
    if isinstance(scalar, (int, float)):
        return "numeric"
    if isinstance(scalar, str):
        return "categorical"
    return None


def _divider(values: Iterable[Any], scale: Optional[ScaleBounds] = None) -> float:
    """What a rating difference is divided by: the rubric's width, widened to
    the range of `values` when they fall outside it, and that range alone when
    no usable bounds are known."""
    numerics = [v for v in values if _classify(v) == "numeric"]
    span = max(numerics) - min(numerics) if numerics else 0
    if scale:
        low, high = scale
        if low is not None and high is not None and high > low:
            span = max(span, high - low)
    return float(span) if span > 0 else 1.0


def _pair_score(value_a: Any, value_b: Any, span: float) -> Optional[float]:
    """Agreement for one pair of judgements, or None when the two are not
    comparable: different kinds, or either one unclassifiable."""
    kind = _classify(value_a)
    if kind is None or kind != _classify(value_b):
        return None
    if kind == "numeric":
        return 1.0 - abs(value_a - value_b) / span
    return 1.0 if value_a == value_b else 0.0


def _pairwise_agreement(
    values: List[Any],
    scale: Optional[ScaleBounds] = None,
) -> Tuple[float, int]:
    """Mean pairwise agreement and pair count over a single annotation slot.

    `scale` is the slot evaluator's rubric bounds, used to divide rating
    differences. Returns (mean_agreement, pair_count). pair_count == 0 ⇒ no
    comparable pairs.
    """
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return (0.0, 0)

    span = _divider(present, scale)
    total = 0.0
    pairs = 0
    for value_a, value_b in combinations(present, 2):
        score = _pair_score(value_a, value_b, span)
        if score is None:
            continue
        total += score
        pairs += 1
    if pairs == 0:
        return (0.0, 0)
    return (total / pairs, pairs)


def _slot_key(annotation: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    return (annotation["item_id"], annotation.get("evaluator_id"))


def scales_for(
    annotations: Iterable[Dict[str, Any]],
    runs: Iterable[Dict[str, Any]] = (),
) -> Scales:
    """Rubric bounds for every evaluator, and every judge version, these rows
    mention. Two queries however many rows."""
    from db import get_evaluator_versions_by_uuids, get_evaluators_by_uuids
    from llm_judge import _scale_bounds

    runs = list(runs)
    evaluator_ids = {r.get("evaluator_id") for r in [*annotations, *runs]} - {None}
    if not evaluator_ids:
        return {}
    live = {
        uuid: ev["live_version_id"]
        for uuid, ev in get_evaluators_by_uuids(list(evaluator_ids)).items()
        if ev.get("live_version_id")
    }
    version_ids = set(live.values()) | (
        {r.get("evaluator_version_id") for r in runs} - {None}
    )
    bounds = {
        uuid: _scale_bounds(version.get("output_config"))
        for uuid, version in get_evaluator_versions_by_uuids(list(version_ids)).items()
    }
    return {**bounds, **{ev: bounds[v] for ev, v in live.items() if v in bounds}}


def filter_runs_to_live_versions(
    runs: Iterable[Dict[str, Any]],
    live_version_by_evaluator: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:
    """Drop evaluator_runs whose `evaluator_version_id` doesn't match the
    evaluator's current live version.

    Used so that experimental/non-live runs don't silently contaminate the
    "evaluator agreement" surfaced anywhere in the API. When a user says
    "evaluator agreement" without qualifying a version, they mean the live
    version's agreement.

    Runs for evaluators with no live version (or evaluator_id missing in the
    map) are dropped — without a live version there's no contribution to
    "live agreement" by definition.
    """
    return [
        r
        for r in runs
        if (live_v := live_version_by_evaluator.get(r.get("evaluator_id") or ""))
        and r.get("evaluator_version_id") == live_v
    ]


def runs_at_live_versions(
    runs: Iterable[Dict[str, Any]],
    evaluators: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep only runs at each given evaluator's live version, so aggregated
    agreement never mixes in an experimental version's judgements."""
    return filter_runs_to_live_versions(
        runs, {e["uuid"]: e.get("live_version_id") for e in evaluators}
    )


def aggregate_agreement(
    annotations: Iterable[Dict[str, Any]],
    scales: Optional[Scales] = None,
) -> Tuple[Optional[float], int]:
    """Mean pairwise agreement across every (item, evaluator) slot that has ≥2
    distinct annotators in the supplied annotation list.

    Rating differences are divided by each evaluator's live rubric width,
    looked up when `scales` is not given. Returns (mean_agreement | None if no
    comparable pairs, total_pair_count).
    """
    annotations = list(annotations)
    if scales is None:
        scales = scales_for(annotations)
    # Group by (item_id, evaluator_id), dedup per annotator (latest wins — but
    # input is already the upserted row, so one annotator contributes once).
    by_slot: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
    for a in annotations:
        slot = _slot_key(a)
        annotator = a.get("annotator_id")
        if annotator is None:
            continue
        scalar = _scalar(a.get("value"))
        if scalar is None:
            continue
        # Pick the latest per (slot, annotator) — annotations are sorted by
        # updated_at ascending, so we just overwrite.
        by_slot.setdefault(slot, {})[annotator] = scalar

    total_weighted = 0.0
    total_pairs = 0
    for (_item_id, ev_id), per_annotator in by_slot.items():
        if len(per_annotator) < 2:
            continue
        mean, pairs = _pairwise_agreement(
            list(per_annotator.values()), scales.get(ev_id)
        )
        if pairs == 0:
            continue
        total_weighted += mean * pairs
        total_pairs += pairs
    if total_pairs == 0:
        return (None, 0)
    return (_round_agreement(total_weighted / total_pairs), total_pairs)


# ============ Bucketing for trend series ============


_BUCKET_DELTAS = {
    "week": timedelta(days=7),
    "month": timedelta(days=30),       # calendar-month-ish; keeps math simple
    "year": timedelta(days=365),
}


def _parse_ts(ts: str) -> datetime:
    """Parse a SQLite CURRENT_TIMESTAMP string ('YYYY-MM-DD HH:MM:SS', UTC)."""
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _fmt_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def build_buckets(
    bucket: str, days: int, now: Optional[datetime] = None
) -> List[Tuple[datetime, datetime]]:
    """Build a list of (start, end) UTC bucket boundaries spanning the last
    `days` ending now, in ascending order. Each bucket is [start, end)."""
    if bucket not in _BUCKET_DELTAS:
        raise ValueError(
            f"bucket must be one of {list(_BUCKET_DELTAS)}, got {bucket!r}"
        )
    if days <= 0:
        raise ValueError("days must be > 0")
    delta = _BUCKET_DELTAS[bucket]
    now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    window_start = now - timedelta(days=days)
    buckets: List[Tuple[datetime, datetime]] = []
    cur_end = now
    while cur_end > window_start:
        cur_start = cur_end - delta
        if cur_start < window_start:
            cur_start = window_start
        buckets.append((cur_start, cur_end))
        cur_end = cur_start
    return list(reversed(buckets))


# Timestamp fields to read, in order of preference, per kind of row.
_ANNOTATION_TS = ("updated_at", "created_at")
_RUN_TS = ("completed_at", "created_at")


def _bucketed_streams(
    bucket: str,
    days: int,
    now: Optional[datetime],
    streams: Sequence[Tuple[Iterable[Dict[str, Any]], Tuple[str, ...]]],
) -> Iterator[
    Tuple[Dict[str, str], List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]
]:
    """Walk the buckets once, yielding per bucket: the bucket's start/end meta
    dict, every stream's rows timestamped up to the bucket end, and the rows
    that bucket added.

    Each stream is `(rows, timestamp_fields)`; the first field a row carries is
    its timestamp, and a row with none or an unparseable one is dropped. The
    cumulative lists are reused across buckets, so read them before advancing.
    """
    parsed: List[List[Tuple[datetime, Dict[str, Any]]]] = []
    for rows, ts_fields in streams:
        dated: List[Tuple[datetime, Dict[str, Any]]] = []
        for row in rows:
            ts = next((row.get(f) for f in ts_fields if row.get(f)), None)
            if not ts:
                continue
            try:
                dated.append((_parse_ts(ts), row))
            except ValueError:
                continue
        dated.sort(key=lambda pair: pair[0])
        parsed.append(dated)

    next_idx = [0] * len(parsed)
    visible: List[List[Dict[str, Any]]] = [[] for _ in parsed]
    for start, end in build_buckets(bucket, days, now=now):
        added: List[List[Dict[str, Any]]] = []
        for stream_no, dated in enumerate(parsed):
            new_rows: List[Dict[str, Any]] = []
            while next_idx[stream_no] < len(dated) and dated[next_idx[stream_no]][0] <= end:
                new_rows.append(dated[next_idx[stream_no]][1])
                next_idx[stream_no] += 1
            visible[stream_no].extend(new_rows)
            added.append(new_rows)
        yield (
            {"bucket_start": _fmt_ts(start), "bucket_end": _fmt_ts(end)},
            visible,
            added,
        )


def _latest_evaluator_value_per_slot(
    evaluator_runs: Iterable[Dict[str, Any]],
    evaluator_id: Optional[str] = None,
) -> Dict[Tuple[str, str], Tuple[Any, str, Optional[str]]]:
    """Pick the most recent evaluator_run per (item_id, evaluator_id) pair.
    Returns `{(item_id, evaluator_id): (scalar_value, completed_at_or_created_at,
    evaluator_version_id)}`. Filters to a single evaluator_id when supplied."""
    out: Dict[Tuple[str, str], Tuple[Any, str, Optional[str]]] = {}
    for r in evaluator_runs:
        ev_id = r.get("evaluator_id")
        if not ev_id:
            continue
        if evaluator_id is not None and ev_id != evaluator_id:
            continue
        item_id = r.get("item_id")
        if not item_id:
            continue
        scalar = _scalar(r.get("value"))
        if scalar is None:
            continue
        ts = r.get("completed_at") or r.get("created_at") or ""
        slot = (item_id, ev_id)
        existing = out.get(slot)
        if existing is None or ts > existing[1]:
            out[slot] = (scalar, ts, r.get("evaluator_version_id"))
    return out


def _value_summary(values: List[Any]) -> Dict[str, Any]:
    """Roll judgement values up into `{count, true_count, mean}`. `true_count`
    is set for binary values and `mean` for numeric ones; the other stays None."""
    bools = [v for v in values if isinstance(v, bool)]
    numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return {
        "count": len(values),
        "true_count": sum(1 for v in bools if v) if bools else None,
        "mean": _round_agreement(sum(numbers) / len(numbers)) if numbers else None,
    }


def evaluator_result_summary(
    evaluator_runs: Iterable[Dict[str, Any]],
    evaluator_id: str,
) -> Dict[str, Any]:
    """Roll one evaluator's own judgements up into `{count, true_count, mean}`.

    Same per-slot value the summary table shows (latest run per item), so the
    two screens can't disagree. `true_count` is set for binary values and
    `mean` for numeric ones; the other stays None.
    """
    return _value_summary(
        [
            value
            for (value, _ts, _version) in _latest_evaluator_value_per_slot(
                evaluator_runs, evaluator_id
            ).values()
        ]
    )


def human_result_summary(
    annotations: Iterable[Dict[str, Any]],
    evaluator_id: str,
) -> Optional[Dict[str, Any]]:
    """Roll the human labels for one evaluator up into `{count, true_count, mean}`.

    Same shape as `evaluator_result_summary`, so the two numbers sit side by
    side. Every label counts once, so an item labelled by three annotators
    weighs three times as much as one labelled by one. Returns None when
    nobody has labelled for this evaluator.
    """
    by_pair: Dict[Tuple[str, Any], Any] = {}
    for a in annotations:
        if a.get("evaluator_id") != evaluator_id:
            continue
        annotator = a.get("annotator_id")
        if annotator is None:
            continue
        scalar = _scalar(a.get("value"))
        if scalar is None:
            continue
        by_pair[(a["item_id"], annotator)] = scalar

    values = list(by_pair.values())
    if not values:
        return None
    return _value_summary(values)


def evaluator_human_pair_agreement(
    eval_value: Any,
    human_values: Iterable[Any],
    scale: Optional[ScaleBounds] = None,
) -> Tuple[float, int]:
    """One-slot evaluator-vs-human agreement: pair the evaluator's value with
    each human's value on the same slot and return (sum_of_scores, pair_count).

    Same scoring rules as `_pairwise_agreement`:
      - binary / categorical: 1.0 if equal, 0.0 otherwise
      - numeric: 1 - |a - b| / the rubric width in `scale`

    Pairs of mismatched type are skipped (don't contribute to the count).

    Returns the SUM (not mean) so callers aggregating across slots can weight
    correctly: account-level mean = sum_of_sums / sum_of_counts. Single-slot
    callers that want the mean compute `sum / count` themselves (and treat
    count==0 as `None`).
    """
    eval_kind = _classify(eval_value)
    if eval_kind is None:
        return (0.0, 0)
    humans = list(human_values)
    if not humans:
        return (0.0, 0)
    span = _divider([eval_value, *humans], scale)
    total = 0.0
    pairs = 0
    for hv in humans:
        score = _pair_score(eval_value, hv, span)
        if score is None:
            continue
        total += score
        pairs += 1
    return (total, pairs)


def aggregate_human_evaluator_agreement(
    annotations: Iterable[Dict[str, Any]],
    evaluator_runs: Iterable[Dict[str, Any]],
    evaluator_id: str,
    scales: Optional[Scales] = None,
) -> Tuple[Optional[float], int]:
    """Mean pairwise agreement between one evaluator's machine judgements and
    every human annotation on the matching slot.

    Slot = (item_id, evaluator_id). For each slot where the evaluator ran AND
    at least one human annotated, pair the evaluator value with each human's
    latest value and score:
      - bool / categorical: 1.0 if equal else 0.0
      - numeric: 1 - |a - b| / the rubric width of the version the judge score
        came from, looked up when `scales` is not given
    Mean weighted by pair count. (None, 0) if no comparable pairs."""
    annotations = list(annotations)
    evaluator_runs = list(evaluator_runs)
    eval_by_slot = _latest_evaluator_value_per_slot(evaluator_runs, evaluator_id)
    if not eval_by_slot:
        return (None, 0)
    if scales is None:
        scales = scales_for(annotations, evaluator_runs)

    # Group human annotations by (item, evaluator) slot, keeping latest per
    # annotator. Annotations are sorted ascending by updated_at so overwriting
    # gives us "latest wins" without an explicit timestamp comparison.
    human_by_slot: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for a in annotations:
        ev_id = a.get("evaluator_id")
        if ev_id != evaluator_id:
            continue
        slot = (a["item_id"], ev_id)
        annotator = a.get("annotator_id")
        if annotator is None:
            continue
        scalar = _scalar(a.get("value"))
        if scalar is None:
            continue
        human_by_slot.setdefault(slot, {})[annotator] = scalar

    total_weighted = 0.0
    total_pairs = 0
    for slot, (eval_val, _ts, version_id) in eval_by_slot.items():
        humans = human_by_slot.get(slot)
        if not humans:
            continue
        slot_total, slot_pairs = evaluator_human_pair_agreement(
            eval_val,
            humans.values(),
            scales.get(version_id, scales.get(evaluator_id)),
        )
        total_weighted += slot_total
        total_pairs += slot_pairs
    if total_pairs == 0:
        return (None, 0)
    return (_round_agreement(total_weighted / total_pairs), total_pairs)


def has_any_comparable_pair(
    annotations: Iterable[Dict[str, Any]],
    evaluator_runs: Iterable[Dict[str, Any]],
    evaluator_ids: Iterable[str],
) -> bool:
    """True if any agreement pair exists: a shared slot with ≥2 human
    annotators (human-vs-human), or a slot where a human and one of the given
    evaluators both judged (human-vs-evaluator). False otherwise."""
    annotations_list = list(annotations)
    runs_list = list(evaluator_runs)
    # The divider cannot change a pair count, so no rubric lookup.
    if aggregate_agreement(annotations_list, {})[1] > 0:
        return True
    for ev_id in evaluator_ids:
        if aggregate_human_evaluator_agreement(
            annotations_list, runs_list, ev_id, {}
        )[1] > 0:
            return True
    return False


def per_item_agreement(
    annotations_for_item: Iterable[Dict[str, Any]],
    evaluator_runs_for_item: Iterable[Dict[str, Any]],
    evaluator_ids: Iterable[str],
    scales: Optional[Scales] = None,
) -> Dict[str, Any]:
    """Compute, for one item: human-vs-human agreement across all evaluators
    annotated, plus per-evaluator human-vs-evaluator agreement. Returns:

        {
          "human_human": {"agreement": float|None, "pair_count": int},
          "evaluators": [
            {"evaluator_id": str, "agreement": float|None, "pair_count": int}
          ]
        }
    """
    annotations_list = list(annotations_for_item)
    runs_list = list(evaluator_runs_for_item)
    if scales is None:
        scales = scales_for(annotations_list, runs_list)
    hh, hh_pairs = aggregate_agreement(annotations_list, scales)
    out_evaluators: List[Dict[str, Any]] = []
    for ev_id in evaluator_ids:
        agree, pairs = aggregate_human_evaluator_agreement(
            annotations_list, runs_list, ev_id, scales
        )
        out_evaluators.append(
            {
                "evaluator_id": ev_id,
                "agreement": agree,
                "pair_count": pairs,
            }
        )
    return {
        "human_human": {"agreement": hh, "pair_count": hh_pairs},
        "evaluators": out_evaluators,
    }


def trend_series_human_evaluator(
    annotations: List[Dict[str, Any]],
    evaluator_runs: List[Dict[str, Any]],
    evaluator_ids: List[str],
    bucket: str,
    days: int,
    now: Optional[datetime] = None,
    scales: Optional[Scales] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Per-evaluator cumulative agreement trend over time, bucketed.

    For each bucket end T and each evaluator E, compute alignment using:
      - every annotation with `updated_at <= T`
      - every evaluator_run with `completed_at <= T` (falls back to created_at)

    Returns `{evaluator_id: [{bucket_start, bucket_end, agreement, pair_count}, ...]}`.
    """
    if not evaluator_ids:
        return {}
    if scales is None:
        scales = scales_for(annotations, evaluator_runs)

    out: Dict[str, List[Dict[str, Any]]] = {ev_id: [] for ev_id in evaluator_ids}
    for meta, (visible_ann, visible_runs), _added in _bucketed_streams(
        bucket,
        days,
        now,
        [(annotations, _ANNOTATION_TS), (evaluator_runs, _RUN_TS)],
    ):
        for ev_id in evaluator_ids:
            agreement, pair_count = aggregate_human_evaluator_agreement(
                visible_ann, visible_runs, ev_id, scales
            )
            out[ev_id].append(
                {**meta, "agreement": agreement, "pair_count": pair_count}
            )
    return out


def trend_series_evaluator_breakdown(
    annotations: List[Dict[str, Any]],
    runs: List[Dict[str, Any]],
    evaluator_id: str,
    version_ids: List[str],
    task_ids: List[str],
    bucket: str,
    days: int,
    now: Optional[datetime] = None,
    scales: Optional[Scales] = None,
) -> Dict[str, Any]:
    """Single-pass cumulative agreement trend for one evaluator, broken down by
    version and task simultaneously.

    `runs` must carry a `task_id` field (as returned by
    `get_evaluator_runs_for_evaluator_org_scoped`).

    Returns:
        {
            "overall":    [{bucket_start, bucket_end, agreement, pair_count}, ...],
            "by_version": {version_id: [series ...]},
            "by_task":    {task_id:    [series ...]},
        }
    """
    if scales is None:
        scales = scales_for(annotations, runs)
    version_id_set = set(version_ids)
    task_id_set = set(task_ids)

    # Pre-separated visible accumulators — populated as we advance through
    # buckets so each aggregate call works on an already-filtered slice.
    visible_ann_by_task: Dict[str, List[Dict[str, Any]]] = {tid: [] for tid in task_ids}
    visible_runs_by_version: Dict[str, List[Dict[str, Any]]] = {vid: [] for vid in version_ids}
    visible_runs_by_task: Dict[str, List[Dict[str, Any]]] = {tid: [] for tid in task_ids}

    out_overall: List[Dict[str, Any]] = []
    out_by_version: Dict[str, List[Dict[str, Any]]] = {vid: [] for vid in version_ids}
    out_by_task: Dict[str, List[Dict[str, Any]]] = {tid: [] for tid in task_ids}

    for (
        meta,
        (visible_ann_all, visible_runs_all),
        (new_ann, new_runs),
    ) in _bucketed_streams(
        bucket, days, now, [(annotations, _ANNOTATION_TS), (runs, _RUN_TS)]
    ):
        for a in new_ann:
            tid = a.get("task_id")
            if tid in task_id_set:
                visible_ann_by_task[tid].append(a)

        for r in new_runs:
            vid = r.get("evaluator_version_id")
            if vid in version_id_set:
                visible_runs_by_version[vid].append(r)
            tid = r.get("task_id")
            if tid in task_id_set:
                visible_runs_by_task[tid].append(r)

        agree, pairs = aggregate_human_evaluator_agreement(
            visible_ann_all, visible_runs_all, evaluator_id, scales
        )
        out_overall.append({**meta, "agreement": agree, "pair_count": pairs})

        for vid in version_ids:
            agree, pairs = aggregate_human_evaluator_agreement(
                visible_ann_all, visible_runs_by_version[vid], evaluator_id, scales
            )
            out_by_version[vid].append({**meta, "agreement": agree, "pair_count": pairs})

        for tid in task_ids:
            agree, pairs = aggregate_human_evaluator_agreement(
                visible_ann_by_task[tid], visible_runs_by_task[tid], evaluator_id, scales
            )
            out_by_task[tid].append({**meta, "agreement": agree, "pair_count": pairs})

    return {
        "overall": out_overall,
        "by_version": out_by_version,
        "by_task": out_by_task,
    }


def aggregate_agreement_for_annotator(
    annotations: Iterable[Dict[str, Any]],
    annotator_id: str,
    scales: Optional[Scales] = None,
) -> Tuple[Optional[float], int]:
    """Mean pairwise agreement between `annotator_id` and every other annotator
    on slots where both participated.

    Slots = (item_id, evaluator_id). Only slots that include `annotator_id`
    AND at least one other annotator contribute. Each pair is `annotator_id`
    vs one other annotator.
    """
    annotations = list(annotations)
    if scales is None:
        scales = scales_for(annotations)
    by_slot: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
    for a in annotations:
        slot = _slot_key(a)
        annotator = a.get("annotator_id")
        if annotator is None:
            continue
        scalar = _scalar(a.get("value"))
        if scalar is None:
            continue
        by_slot.setdefault(slot, {})[annotator] = scalar

    total_weighted = 0.0
    total_pairs = 0
    for (_item_id, ev_id), per_annotator in by_slot.items():
        if annotator_id not in per_annotator or len(per_annotator) < 2:
            continue
        my_val = per_annotator[annotator_id]
        span = _divider(per_annotator.values(), scales.get(ev_id))
        for other_id, other_val in per_annotator.items():
            if other_id == annotator_id:
                continue
            score = _pair_score(my_val, other_val, span)
            if score is None:
                continue
            total_weighted += score
            total_pairs += 1
    if total_pairs == 0:
        return (None, 0)
    return (_round_agreement(total_weighted / total_pairs), total_pairs)


def trend_series(
    annotations: List[Dict[str, Any]],
    bucket: str,
    days: int,
    now: Optional[datetime] = None,
    scales: Optional[Scales] = None,
) -> List[Dict[str, Any]]:
    """Group annotations into buckets by `updated_at`, compute per-bucket
    pairwise agreement using ALL annotations whose updated_at <= bucket end.

    The "as-of" semantics make the trend monotone in coverage: once a row has
    enough annotators, every subsequent bucket still benefits from those
    judgements. This matches the intuition "average alignment so far."
    """
    if scales is None:
        scales = scales_for(annotations)
    out: List[Dict[str, Any]] = []
    for meta, (visible,), _added in _bucketed_streams(
        bucket, days, now, [(annotations, _ANNOTATION_TS)]
    ):
        agreement, pair_count = aggregate_agreement(visible, scales)
        out.append({**meta, "agreement": agreement, "pair_count": pair_count})
    return out


def trend_series_for_annotator(
    annotations: List[Dict[str, Any]],
    annotator_id: str,
    bucket: str,
    days: int,
    now: Optional[datetime] = None,
    scales: Optional[Scales] = None,
) -> List[Dict[str, Any]]:
    """Cumulative trend of `annotator_id`'s pairwise agreement with other
    annotators, bucketed by updated_at."""
    if scales is None:
        scales = scales_for(annotations)
    out: List[Dict[str, Any]] = []
    for meta, (visible,), _added in _bucketed_streams(
        bucket, days, now, [(annotations, _ANNOTATION_TS)]
    ):
        agreement, pair_count = aggregate_agreement_for_annotator(
            visible, annotator_id, scales
        )
        out.append({**meta, "agreement": agreement, "pair_count": pair_count})
    return out
