"""Item filters for an annotation task: by score and by whether it is labelled.

Shared by `GET /annotation-tasks/{uuid}/summary` and the bulk actions that take
`select_all`, so the list and a select-all over it always act on the same items.
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import HTTPException

from annotation_metrics import _scalar, filter_runs_to_live_versions

# Each filter is one pass over the task's items, so an unbounded repeat is an
# unbounded amount of work. Matches the cap on the trace list's `score`.
MAX_SCORE_FILTERS = 20

SCORE_SOURCES = ("evaluator", "human", "either", "both")

SCORE_FILTER_DESCRIPTION = (
    "Return only items with this score, written as "
    "`<evaluator ID>:<values>:<source>`. Values are `true`, `false` or whole "
    "numbers, joined by `.` to accept any of them: `1.2` means 1 or 2. The "
    "source says whose answer is checked:\n"
    "- `evaluator`: the evaluator's latest score\n"
    "- `human`: at least one annotator's latest label\n"
    "- `either`: the evaluator or an annotator, used when the source is left out\n"
    "- `both`: the evaluator and at least one annotator\n"
    "Repeat the parameter to require every filter at once"
)

LABELLED_FILTER_DESCRIPTION = (
    "When true, return only items at least one annotator has scored. When "
    "false, return only items nobody has scored yet"
)

# (evaluator ID, accepted values, source)
ScoreFilter = Tuple[str, Set[float], str]


def parse_score_filters(
    values: Optional[List[str]], linked_evaluator_ids: Set[str]
) -> List[ScoreFilter]:
    """Turn each `<evaluator ID>:<values>:<source>` into a ScoreFilter."""
    raw_values = values or []
    if len(raw_values) > MAX_SCORE_FILTERS:
        raise HTTPException(
            status_code=422,
            detail=f"score accepts at most {MAX_SCORE_FILTERS} filters",
        )
    parsed = []
    for raw in raw_values:
        parts = [p.strip() for p in raw.split(":")]
        if len(parts) == 2:
            parts.append("either")
        if len(parts) != 3 or not parts[0] or parts[2] not in SCORE_SOURCES:
            raise _bad_score_filter(raw)
        evaluator_uuid, value_text, source = parts
        accepted = set()
        for token in value_text.split("."):
            number = _filter_number(token.strip())
            if number is None:
                raise _bad_score_filter(raw)
            accepted.add(number)
        if evaluator_uuid not in linked_evaluator_ids:
            raise HTTPException(
                status_code=422,
                detail=f"{raw!r} names an evaluator that is not linked to this task",
            )
        parsed.append((evaluator_uuid, accepted, source))
    return parsed


def _filter_number(token: str) -> Optional[float]:
    if token == "true":
        return 1.0
    if token == "false":
        return 0.0
    try:
        number = float(token)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _bad_score_filter(raw: str) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail=(
            f"{raw!r} is not a score filter. Write an evaluator ID, a colon, "
            "values such as true, false or 1.2, then optionally a colon and "
            "evaluator, human, either or both"
        ),
    )


def _comparable(value: Any) -> Optional[float]:
    """A stored value as a number, so true/1 and false/0 compare equal."""
    scalar = _scalar(value)
    if isinstance(scalar, (bool, int, float)):
        return float(scalar)
    # Labels uploaded as text read the way the Items tab reads them.
    if isinstance(scalar, str):
        return _TEXT_VERDICTS.get(scalar.strip().lower())
    return None


_TEXT_VERDICTS = {"true": 1.0, "yes": 1.0, "1": 1.0, "false": 0.0, "no": 0.0, "0": 0.0}


def filter_items(
    items: List[Dict[str, Any]],
    *,
    scores: List[ScoreFilter],
    labelled: Optional[bool],
    evaluators: List[Dict[str, Any]],
    runs: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    annotator_ids: Set[str],
) -> List[Dict[str, Any]]:
    """Keep the items that pass every score filter and the labelled filter.

    `annotator_ids` are the annotators that still exist. The summary leaves a
    deleted annotator's labels out of its rows, so they are left out here too.
    """
    if not scores and labelled is None:
        return items
    live_version = {ev["uuid"]: ev.get("live_version_id") for ev in evaluators}

    # Latest run per (item, evaluator) at the evaluator's live version, picked
    # by the same timestamp rule as the summary rows.
    evaluator_value: Dict[tuple, Optional[float]] = {}
    latest_ts: Dict[tuple, str] = {}
    for r in filter_runs_to_live_versions(runs, live_version):
        slot = (r.get("item_id"), r.get("evaluator_id"))
        ts = r.get("completed_at") or r.get("created_at") or ""
        if slot not in latest_ts or ts > latest_ts[slot]:
            latest_ts[slot] = ts
            evaluator_value[slot] = _comparable(r.get("value"))

    # Each annotator's latest annotation per (item, evaluator). Annotations
    # arrive oldest first, so a later clear overwrites an earlier value.
    latest_ann: Dict[tuple, Any] = {}
    for a in annotations:
        ev_id = a.get("evaluator_id")
        if ev_id not in live_version or a.get("annotator_id") not in annotator_ids:
            continue
        latest_ann[(a.get("item_id"), ev_id, a["annotator_id"])] = _scalar(
            a.get("value")
        )
    human_values: Dict[tuple, Set[float]] = {}
    labelled_items: Set[str] = set()
    for (item_id, ev_id, _), scalar in latest_ann.items():
        if scalar is None:
            continue
        labelled_items.add(item_id)
        value = _comparable(scalar)
        if value is not None:
            human_values.setdefault((item_id, ev_id), set()).add(value)

    def _keep(item_id: str) -> bool:
        if labelled is not None and (item_id in labelled_items) != labelled:
            return False
        for ev_id, accepted, source in scores:
            by_evaluator = evaluator_value.get((item_id, ev_id)) in accepted
            by_human = bool(human_values.get((item_id, ev_id), set()) & accepted)
            ok = {
                "evaluator": by_evaluator,
                "human": by_human,
                "both": by_evaluator and by_human,
            }.get(source, by_evaluator or by_human)
            if not ok:
                return False
        return True

    return [it for it in items if _keep(it["uuid"])]
