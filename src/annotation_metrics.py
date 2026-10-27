"""Agreement between annotators who label the same items, and between those
annotators and an evaluator's judge.

Every comparison is between two answers given to the same item for the same
evaluator. A row-level annotation, which has no evaluator, is grouped on its
own.

Two true/false answers agree 1.0 when equal and 0.0 when not. Two ratings agree
1 - |first_rating - second_rating| / largest_possible_rating_gap, where the
largest possible gap is scale_max - scale_min of the rating scale: the scale of
the version a judge score came from, or of the evaluator's live version for an
annotator's label, which records no version. It is never smaller than the gap
between the ratings compared, so a label given on an older, wider scale cannot
push agreement below 0.

Answers of different kinds, such as a true/false and a rating, are not compared.

Kappa corrects an agreement number for the agreement expected by chance:
(agreement - chance_agreement) / (1 - chance_agreement), over the same pairs
that agreement number compared. Chance agreement is how much two answers drawn
at random would agree, weighed the same way as a real pair. For each evaluator
the answers are drawn from its compared pairs on every item:

- Annotators with each other use Fleiss' kappa. Both answers come from one pool
  in which an annotator's answer counts once for every other annotator it was
  compared with on that item, so an item counts as much as its pairs, as in the
  agreement number. When every item has the same number of annotators this is
  exactly Fleiss' kappa.
- A judge against annotators, and one annotator against the others, use Cohen's
  kappa. Each answer of a pair is drawn from its own side.

Chance agreement between ratings uses the largest possible rating gap each
pair was compared over, gives no credit to two ratings further apart than that
gap, and is averaged over the gaps weighted by pair count.

Kinds of answer are never combined into one kappa: binary answers, which
either match or do not, and ratings each get their own, from their own pairs.
An evaluator counts towards a kind's kappa only with pairs on two items or
more.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

# A rating scale as `(scale_min, scale_max)`. `RatingScales` holds one per
# evaluator uuid (its live version) and one per version uuid (the version a
# judge score came from).
RatingScale = Tuple[Optional[float], Optional[float]]
RatingScales = Dict[str, RatingScale]
# Default for callers that pass no rating scales: the largest possible rating
# gap is taken as the gap between the highest and lowest rating compared. Read
# only, never mutated.
_NO_RATING_SCALES: RatingScales = {}

KappaFamily = Literal["fleiss", "cohen"]
KappaType = Literal["fleiss", "linear_weighted_fleiss", "cohen", "linear_weighted_cohen"]
KappaBlankReason = Literal["too_few_items", "everyone_agreed"]
AnswerKind = Literal["binary", "rating"]
# Agreement over every compared pair, how many pairs that is, and the kappa of
# each kind of answer compared (see `_KappaTally.kappa_by_kind`).
AgreementAndKappa = Tuple[Optional[float], int, List[Dict[str, Any]]]
# The largest possible rating gap one item's answers were compared over, and
# each pair compared as (first_answer, second_answer, agreement).
AnswerPairsCompared = Tuple[float, List[Tuple[Any, Any, float]]]

KAPPA_FIELDS = ("kappa", "kappa_type", "kappa_blank_reason", "chance_agreement")

ANSWER_KIND_DESCRIPTION = (
    "Which answers this entry covers:\n\n"
    "- `binary`: answers that either match or do not, such as pass or fail\n"
    "- `rating`: ratings on a scale\n"
)
KAPPA_BY_KIND_DESCRIPTION = (
    "Agreement with the other annotators and its kappa, one entry for each kind of answer "
    "compared, binary answers first. The kinds are never combined into one kappa"
)


class KappaFields(BaseModel):
    kappa: Optional[float] = Field(
        description=(
            "Agreement corrected for the agreement expected by chance, where 1 is perfect "
            "agreement, 0 is chance level, and below 0 is worse than chance"
        )
    )
    kappa_type: Optional[KappaType] = Field(
        description=(
            "The kind of kappa reported in `kappa`:\n\n"
            "- `fleiss`: Fleiss' kappa between annotators\n"
            "- `linear_weighted_fleiss`: Fleiss' kappa between annotators, giving partial credit to close ratings\n"
            "- `cohen`: Cohen's kappa between a judge and the annotators, or between one annotator and the others\n"
            "- `linear_weighted_cohen`: Cohen's kappa between a judge and the annotators, or between one annotator and the others, giving partial credit to close ratings\n"
        )
    )
    kappa_blank_reason: Optional[KappaBlankReason] = Field(
        description=(
            "Why `kappa` is blank when answers were compared:\n\n"
            "- `too_few_items`: every evaluator compared these answers on only one item, and one item cannot show the agreement expected by chance\n"
            "- `everyone_agreed`: every compared answer is the same, so agreement cannot be told apart from chance\n"
        )
    )
    chance_agreement: Optional[float] = Field(
        description="Agreement expected by chance alone, which `kappa` corrects for, on the same scale as the agreement"
    )


# All agreement floats returned to callers are rounded to this many decimal
# digits so the API surface is stable and free of float-formatting noise.
_AGREEMENT_DECIMALS = 4


def _round_agreement(agreement: Optional[float]) -> Optional[float]:
    if agreement is None:
        return None
    return round(float(agreement), _AGREEMENT_DECIMALS)


def _extract_answer(stored_value: Any) -> Any:
    """The answer inside a stored annotation or judge score `value`: a bare
    bool, number or string, or from a dict the first of {value, score, rating,
    label, binary} that is set. Comments and other keys are ignored."""
    if stored_value is None:
        return None
    if isinstance(stored_value, (bool, int, float, str)):
        return stored_value
    if isinstance(stored_value, dict):
        for key in ("value", "score", "rating", "label", "binary"):
            if key in stored_value and stored_value[key] is not None:
                return stored_value[key]
    return None


def _answer_kind(answer: Any) -> Optional[str]:
    if answer is None:
        return None
    if isinstance(answer, bool):
        return "binary"
    if isinstance(answer, (int, float)):
        return "numeric"
    if isinstance(answer, str):
        return "categorical"
    return None


def _largest_possible_rating_gap(
    answers: Iterable[Any], rating_scale: Optional[RatingScale] = None
) -> float:
    """The largest gap two ratings could have: scale_max - scale_min of
    `rating_scale`, raised to the gap between the highest and lowest of the
    ratings in `answers` when they fall outside the scale, and that gap alone
    when no usable scale is known."""
    ratings = [answer for answer in answers if _answer_kind(answer) == "numeric"]
    largest_possible_rating_gap = max(ratings) - min(ratings) if ratings else 0
    if rating_scale:
        scale_min, scale_max = rating_scale
        if scale_min is not None and scale_max is not None and scale_max > scale_min:
            largest_possible_rating_gap = max(
                largest_possible_rating_gap, scale_max - scale_min
            )
    if largest_possible_rating_gap <= 0:
        return 1.0
    return float(largest_possible_rating_gap)


def _agreement_between_two_answers(
    first_answer: Any, second_answer: Any, largest_possible_rating_gap: float
) -> Optional[float]:
    """Agreement between two answers, or None when they cannot be compared:
    different kinds, or either one of no known kind."""
    kind = _answer_kind(first_answer)
    if kind is None or kind != _answer_kind(second_answer):
        return None
    if kind == "numeric":
        return 1.0 - abs(first_answer - second_answer) / largest_possible_rating_gap
    return 1.0 if first_answer == second_answer else 0.0


def _answer_pairs_compared(
    answer_pairs: Iterable[Tuple[Any, Any]], largest_possible_rating_gap: float
) -> List[Tuple[Any, Any, float]]:
    """The pairs whose two answers can be compared, each with its agreement."""
    return [
        (first_answer, second_answer, agreement)
        for first_answer, second_answer in answer_pairs
        if (
            agreement := _agreement_between_two_answers(
                first_answer, second_answer, largest_possible_rating_gap
            )
        )
        is not None
    ]


def _agreements(answer_pairs_compared: Iterable[Tuple[Any, Any, float]]) -> List[float]:
    return [agreement for _first_answer, _second_answer, agreement in answer_pairs_compared]


def _mean_agreement(
    agreements: Iterable[float], rounded: bool = True
) -> Tuple[Optional[float], int]:
    """Mean agreement over every compared pair, given each pair's agreement,
    and how many pairs that is, or (None, 0) when there are none. Every
    agreement average goes through here, summed with `math.fsum` so the order
    the pairs arrive in never changes the result."""
    agreement_list = list(agreements)
    if not agreement_list:
        return (None, 0)
    mean = math.fsum(agreement_list) / len(agreement_list)
    return (_round_agreement(mean) if rounded else mean, len(agreement_list))


def _answer_pairs_compared_between_annotators(
    answers: List[Any],
    rating_scale: Optional[RatingScale] = None,
) -> AnswerPairsCompared:
    """Every pair of `answers` given to one item for one evaluator that can be
    compared, and the largest possible rating gap they were compared over."""
    given_answers = [answer for answer in answers if answer is not None]
    largest_possible_rating_gap = _largest_possible_rating_gap(
        given_answers, rating_scale
    )
    return (
        largest_possible_rating_gap,
        _answer_pairs_compared(
            combinations(given_answers, 2), largest_possible_rating_gap
        ),
    )


def _mean_agreement_between_answers(
    answers: List[Any],
    rating_scale: Optional[RatingScale] = None,
) -> Tuple[float, int]:
    """Mean agreement over every pair of `answers` given to one item for one
    evaluator, and how many pairs that mean covers. `rating_scale` is that
    evaluator's rating scale. Returns (0.0, 0) when no two answers can be
    compared."""
    mean, pair_count = _mean_agreement(
        _agreements(_answer_pairs_compared_between_annotators(answers, rating_scale)[1]),
        rounded=False,
    )
    return (mean if pair_count else 0.0, pair_count)


def _record_annotator_answers(
    item_and_evaluator_to_annotator_answers: Dict[
        Tuple[str, Optional[str]], Dict[str, Any]
    ],
    annotations: Iterable[Dict[str, Any]],
    only_evaluator_id: Optional[str] = None,
) -> Dict[Tuple[str, Optional[str]], None]:
    """Add each annotator's answer to `{(item_id, evaluator_id): {annotator_id:
    answer}}`, a later annotation overwriting an earlier one, and return the
    `(item_id, evaluator_id)` pairs that got an answer, in the order first
    reached. Only annotations for `only_evaluator_id` are kept when it is
    given."""
    item_and_evaluator_pairs_answered: Dict[Tuple[str, Optional[str]], None] = {}
    for annotation in annotations:
        evaluator_id = annotation.get("evaluator_id")
        if only_evaluator_id is not None and evaluator_id != only_evaluator_id:
            continue
        annotator_id = annotation.get("annotator_id")
        if annotator_id is None:
            continue
        answer = _extract_answer(annotation.get("value"))
        if answer is None:
            continue
        item_and_evaluator = (annotation["item_id"], evaluator_id)
        item_and_evaluator_to_annotator_answers.setdefault(item_and_evaluator, {})[
            annotator_id
        ] = answer
        item_and_evaluator_pairs_answered[item_and_evaluator] = None
    return item_and_evaluator_pairs_answered


def _annotator_answers_per_item_and_evaluator(
    annotations: Iterable[Dict[str, Any]],
    only_evaluator_id: Optional[str] = None,
) -> Dict[Tuple[str, Optional[str]], Dict[str, Any]]:
    """`{(item_id, evaluator_id): {annotator_id: answer}}`, keeping each
    annotator's latest answer. Annotations arrive sorted by `updated_at`
    ascending, so a later one simply overwrites. Only annotations for
    `only_evaluator_id` are kept when it is given."""
    item_and_evaluator_to_annotator_answers: Dict[
        Tuple[str, Optional[str]], Dict[str, Any]
    ] = {}
    _record_annotator_answers(
        item_and_evaluator_to_annotator_answers, annotations, only_evaluator_id
    )
    return item_and_evaluator_to_annotator_answers


def rating_scales_for(
    annotations: Iterable[Dict[str, Any]],
    evaluator_runs: Iterable[Dict[str, Any]] = (),
) -> RatingScales:
    """The rating scale of every evaluator, and of every version a judge score
    came from, that these rows mention. Two queries however many rows."""
    from db import get_evaluator_versions_by_uuids, get_evaluators_by_uuids
    from llm_judge import _scale_bounds

    evaluator_runs = list(evaluator_runs)
    evaluator_ids = {
        row.get("evaluator_id") for row in [*annotations, *evaluator_runs]
    } - {None}
    if not evaluator_ids:
        return {}
    evaluator_id_to_live_version_id = {
        evaluator_id: evaluator["live_version_id"]
        for evaluator_id, evaluator in get_evaluators_by_uuids(
            list(evaluator_ids)
        ).items()
        if evaluator.get("live_version_id")
    }
    version_ids = set(evaluator_id_to_live_version_id.values()) | (
        {run.get("evaluator_version_id") for run in evaluator_runs} - {None}
    )
    version_id_to_rating_scale = {
        version_id: _scale_bounds(version.get("output_config"))
        for version_id, version in get_evaluator_versions_by_uuids(
            list(version_ids)
        ).items()
    }
    evaluator_id_to_rating_scale = {
        evaluator_id: version_id_to_rating_scale[live_version_id]
        for evaluator_id, live_version_id in evaluator_id_to_live_version_id.items()
        if live_version_id in version_id_to_rating_scale
    }
    return {**version_id_to_rating_scale, **evaluator_id_to_rating_scale}


def keep_runs_from_live_versions(
    evaluator_runs: Iterable[Dict[str, Any]],
    evaluator_id_to_live_version_id: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:
    """Drop judge scores that did not come from their evaluator's current live
    version, so an experimental version's scores never count towards the
    agreement the API reports. An evaluator with no live version, or missing
    from the map, keeps nothing."""
    return [
        run
        for run in evaluator_runs
        if (
            live_version_id := evaluator_id_to_live_version_id.get(
                run.get("evaluator_id") or ""
            )
        )
        and run.get("evaluator_version_id") == live_version_id
    ]


def keep_runs_from_live_versions_of(
    evaluator_runs: Iterable[Dict[str, Any]],
    evaluators: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """`keep_runs_from_live_versions`, reading each live version off the given
    evaluator records."""
    return keep_runs_from_live_versions(
        evaluator_runs,
        {evaluator["uuid"]: evaluator.get("live_version_id") for evaluator in evaluators},
    )


# ============ Time buckets for agreement over time ============


_TIME_BUCKET_LENGTHS = {
    "week": timedelta(days=7),
    "month": timedelta(days=30),       # calendar-month-ish; keeps math simple
    "year": timedelta(days=365),
}


def _parse_timestamp(timestamp: str) -> datetime:
    """Parse a SQLite CURRENT_TIMESTAMP string ('YYYY-MM-DD HH:MM:SS', UTC)."""
    return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )


def _format_timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def build_time_buckets(
    bucket: str, days: int, now: Optional[datetime] = None
) -> List[Tuple[datetime, datetime]]:
    """Build a list of (start, end) UTC bucket boundaries spanning the last
    `days` ending now, in ascending order. Each bucket is [start, end)."""
    if bucket not in _TIME_BUCKET_LENGTHS:
        raise ValueError(
            f"bucket must be one of {list(_TIME_BUCKET_LENGTHS)}, got {bucket!r}"
        )
    if days <= 0:
        raise ValueError("days must be > 0")
    bucket_length = _TIME_BUCKET_LENGTHS[bucket]
    now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    window_start = now - timedelta(days=days)
    buckets: List[Tuple[datetime, datetime]] = []
    current_bucket_end = now
    while current_bucket_end > window_start:
        current_bucket_start = current_bucket_end - bucket_length
        if current_bucket_start < window_start:
            current_bucket_start = window_start
        buckets.append((current_bucket_start, current_bucket_end))
        current_bucket_end = current_bucket_start
    return list(reversed(buckets))


# Timestamp fields to read, in order of preference, per kind of row.
_ANNOTATION_TIMESTAMP_FIELDS = ("updated_at", "created_at")
_EVALUATOR_RUN_TIMESTAMP_FIELDS = ("completed_at", "created_at")


def _rows_new_in_each_time_bucket(
    bucket: str,
    days: int,
    now: Optional[datetime],
    row_streams: Sequence[Tuple[Iterable[Dict[str, Any]], Tuple[str, ...]]],
) -> Iterator[Tuple[Dict[str, str], List[List[Dict[str, Any]]]]]:
    """Walk the time buckets once. For each bucket, yield its start and end,
    and every stream's rows timestamped after the previous bucket's end and up
    to this bucket's end, oldest first. The first bucket also takes every row
    older than it.

    Each stream is `(rows, timestamp_fields)`; a row's timestamp is the first of
    those fields it carries, and a row with none, or one that will not parse,
    is dropped.
    """
    rows_sorted_by_time_per_stream: List[List[Tuple[datetime, Dict[str, Any]]]] = []
    for rows, timestamp_fields in row_streams:
        timestamped_rows: List[Tuple[datetime, Dict[str, Any]]] = []
        for row in rows:
            timestamp = next(
                (row.get(field) for field in timestamp_fields if row.get(field)), None
            )
            if not timestamp:
                continue
            try:
                timestamped_rows.append((_parse_timestamp(timestamp), row))
            except ValueError:
                continue
        timestamped_rows.sort(key=lambda timestamp_and_row: timestamp_and_row[0])
        rows_sorted_by_time_per_stream.append(timestamped_rows)

    next_row_index_per_stream = [0] * len(rows_sorted_by_time_per_stream)
    for bucket_start, bucket_end in build_time_buckets(bucket, days, now=now):
        rows_new_in_bucket_per_stream: List[List[Dict[str, Any]]] = []
        for stream_index, timestamped_rows in enumerate(rows_sorted_by_time_per_stream):
            rows_new_in_bucket: List[Dict[str, Any]] = []
            while (
                next_row_index_per_stream[stream_index] < len(timestamped_rows)
                and timestamped_rows[next_row_index_per_stream[stream_index]][0]
                <= bucket_end
            ):
                rows_new_in_bucket.append(
                    timestamped_rows[next_row_index_per_stream[stream_index]][1]
                )
                next_row_index_per_stream[stream_index] += 1
            rows_new_in_bucket_per_stream.append(rows_new_in_bucket)
        yield (
            {
                "bucket_start": _format_timestamp(bucket_start),
                "bucket_end": _format_timestamp(bucket_end),
            },
            rows_new_in_bucket_per_stream,
        )


def _record_latest_judge_answers(
    item_and_evaluator_to_latest_judge_answer: Dict[
        Tuple[str, str], Tuple[Any, str, Optional[str]]
    ],
    evaluator_runs: Iterable[Dict[str, Any]],
    only_evaluator_id: Optional[str] = None,
) -> Dict[Tuple[str, str], None]:
    """Add judge scores to `{(item_id, evaluator_id): (answer, scored_at,
    evaluator_version_id)}`, a score replacing the kept one only when scored
    strictly later, and return the `(item_id, evaluator_id)` pairs whose kept
    score changed. Only scores from `only_evaluator_id` are kept when it is
    given."""
    item_and_evaluator_pairs_rescored: Dict[Tuple[str, str], None] = {}
    for run in evaluator_runs:
        evaluator_id = run.get("evaluator_id")
        if not evaluator_id:
            continue
        if only_evaluator_id is not None and evaluator_id != only_evaluator_id:
            continue
        item_id = run.get("item_id")
        if not item_id:
            continue
        answer = _extract_answer(run.get("value"))
        if answer is None:
            continue
        scored_at = run.get("completed_at") or run.get("created_at") or ""
        item_and_evaluator = (item_id, evaluator_id)
        previous = item_and_evaluator_to_latest_judge_answer.get(item_and_evaluator)
        if previous is None or scored_at > previous[1]:
            item_and_evaluator_to_latest_judge_answer[item_and_evaluator] = (
                answer,
                scored_at,
                run.get("evaluator_version_id"),
            )
            item_and_evaluator_pairs_rescored[item_and_evaluator] = None
    return item_and_evaluator_pairs_rescored


def _latest_judge_answer_per_item_and_evaluator(
    evaluator_runs: Iterable[Dict[str, Any]],
    only_evaluator_id: Optional[str] = None,
) -> Dict[Tuple[str, str], Tuple[Any, str, Optional[str]]]:
    """The most recent judge score per (item_id, evaluator_id), as
    `{(item_id, evaluator_id): (answer, scored_at, evaluator_version_id)}`.
    Only scores from `only_evaluator_id` are kept when it is given."""
    item_and_evaluator_to_latest_judge_answer: Dict[
        Tuple[str, str], Tuple[Any, str, Optional[str]]
    ] = {}
    _record_latest_judge_answers(
        item_and_evaluator_to_latest_judge_answer, evaluator_runs, only_evaluator_id
    )
    return item_and_evaluator_to_latest_judge_answer


def _answer_summary(answers: List[Any]) -> Dict[str, Any]:
    """Roll answers up into `{count, true_count, mean}`. `true_count` is set
    for true/false answers and `mean` for ratings; the other stays None."""
    true_false_answers = [answer for answer in answers if isinstance(answer, bool)]
    ratings = [
        answer
        for answer in answers
        if isinstance(answer, (int, float)) and not isinstance(answer, bool)
    ]
    return {
        "count": len(answers),
        "true_count": (
            sum(1 for answer in true_false_answers if answer)
            if true_false_answers
            else None
        ),
        "mean": _round_agreement(sum(ratings) / len(ratings)) if ratings else None,
    }


def judge_answer_summary(
    evaluator_runs: Iterable[Dict[str, Any]],
    evaluator_id: str,
) -> Dict[str, Any]:
    """Roll one evaluator's judge scores up into `{count, true_count, mean}`,
    using the latest score per item, the same one the summary table shows, so
    the two screens cannot disagree."""
    return _answer_summary(
        [
            answer
            for (
                answer,
                _scored_at,
                _evaluator_version_id,
            ) in _latest_judge_answer_per_item_and_evaluator(
                evaluator_runs, evaluator_id
            ).values()
        ]
    )


def annotator_answer_summary(
    annotations: Iterable[Dict[str, Any]],
    evaluator_id: str,
) -> Optional[Dict[str, Any]]:
    """Roll annotators' answers for one evaluator up into
    `{count, true_count, mean}`, the same shape as `judge_answer_summary`, so
    the two numbers sit side by side. Every answer counts once, so an item
    answered by three annotators weighs three times as much as one answered by
    one. Returns None when nobody has answered for this evaluator."""
    answers = [
        answer
        for annotator_id_to_answer in _annotator_answers_per_item_and_evaluator(
            annotations, only_evaluator_id=evaluator_id
        ).values()
        for answer in annotator_id_to_answer.values()
    ]
    if not answers:
        return None
    return _answer_summary(answers)


def _answer_pairs_compared_between_one_answer_and_others(
    answer: Any,
    other_answers: Iterable[Any],
    rating_scale: Optional[RatingScale] = None,
) -> AnswerPairsCompared:
    """One answer on one item, a judge's or an annotator's, paired with each of
    `other_answers` that can be compared with it, and the largest possible
    rating gap they were compared over."""
    other_answer_list = list(other_answers)
    largest_possible_rating_gap = _largest_possible_rating_gap(
        [answer, *other_answer_list], rating_scale
    )
    return (
        largest_possible_rating_gap,
        _answer_pairs_compared(
            ((answer, other_answer) for other_answer in other_answer_list),
            largest_possible_rating_gap,
        ),
    )


def judge_and_annotators_agreement_on_item(
    judge_answer: Any,
    annotator_answers: Iterable[Any],
    rating_scale: Optional[RatingScale] = None,
) -> Tuple[float, int]:
    """Agreement between the judge's answer and each annotator's answer on one
    item for one evaluator, as (sum of agreements, number of pairs). Answers of
    a different kind from the judge's are skipped. A rating gap is divided by
    the largest possible gap on `rating_scale`.

    Returns the sum, not the mean, so callers adding up many items can weight
    each pair equally: overall mean = sum of sums / sum of counts. A caller
    wanting one item's mean divides the two itself, treating a count of 0 as
    None.
    """
    agreements = _agreements(
        _answer_pairs_compared_between_one_answer_and_others(
            judge_answer, annotator_answers, rating_scale
        )[1]
    )
    return (math.fsum(agreements), len(agreements))


def has_answers_to_compare(
    annotations: Iterable[Dict[str, Any]],
    evaluator_runs: Iterable[Dict[str, Any]],
    evaluator_ids: Iterable[str],
) -> bool:
    """True if any item has two answers that can be compared: from two
    annotators, or from an annotator and one of the given evaluators' judges.
    Two answers can be compared when they are of the same known kind."""
    item_and_evaluator_to_annotator_answer_kinds: Dict[
        Tuple[str, Optional[str]], set
    ] = {}
    for (
        item_and_evaluator,
        annotator_id_to_answer,
    ) in _annotator_answers_per_item_and_evaluator(annotations).items():
        annotator_answer_kinds = set()
        for answer in annotator_id_to_answer.values():
            answer_kind = _answer_kind(answer)
            if answer_kind is None:
                continue
            if answer_kind in annotator_answer_kinds:
                return True
            annotator_answer_kinds.add(answer_kind)
        item_and_evaluator_to_annotator_answer_kinds[item_and_evaluator] = (
            annotator_answer_kinds
        )

    evaluator_run_list = list(evaluator_runs)
    for evaluator_id in evaluator_ids:
        for item_and_evaluator, (
            judge_answer,
            _scored_at,
            _judge_version_id,
        ) in _latest_judge_answer_per_item_and_evaluator(
            evaluator_run_list, evaluator_id
        ).items():
            if _answer_kind(
                judge_answer
            ) in item_and_evaluator_to_annotator_answer_kinds.get(
                item_and_evaluator, ()
            ):
                return True
    return False


def agreement_on_item(
    annotations_for_item: Iterable[Dict[str, Any]],
    evaluator_runs_for_item: Iterable[Dict[str, Any]],
    evaluator_ids: Iterable[str],
    rating_scales: RatingScales = _NO_RATING_SCALES,
) -> Dict[str, Any]:
    """For one item: agreement between annotators across every evaluator they
    answered, plus agreement between each evaluator's judge and the
    annotators. Returns:

        {
          "human_human": {"agreement": float|None, "pair_count": int},
          "evaluators": [
            {"evaluator_id": str, "agreement": float|None, "pair_count": int}
          ]
        }
    """
    annotation_list = list(annotations_for_item)
    evaluator_run_list = list(evaluator_runs_for_item)
    # One item can never show kappa (it is always `too_few_items`), so skip it.
    annotator_agreement, annotator_pair_count = agreement_and_kappa_between_annotators(
        annotation_list, rating_scales, with_kappa=False
    )[:2]
    agreement_per_evaluator: List[Dict[str, Any]] = []
    for evaluator_id in evaluator_ids:
        evaluator_agreement, evaluator_pair_count = agreement_and_kappa_between_judge_and_annotators(
            annotation_list, evaluator_run_list, evaluator_id, rating_scales, with_kappa=False
        )[:2]
        agreement_per_evaluator.append(
            {
                "evaluator_id": evaluator_id,
                "agreement": evaluator_agreement,
                "pair_count": evaluator_pair_count,
            }
        )
    return {
        "human_human": {
            "agreement": annotator_agreement,
            "pair_count": annotator_pair_count,
        },
        "evaluators": agreement_per_evaluator,
    }


def _chance_agreement(
    first_answer_counts: Counter,
    second_answer_counts: Counter,
    largest_possible_rating_gap: float,
) -> float:
    """Expected agreement between one answer drawn from each pool, each pair
    weighted the way a real pair compared over `largest_possible_rating_gap`
    would be, and a pair further apart than that gap given no credit."""
    return math.fsum(
        first_count
        * second_count
        * max(
            0.0,
            _agreement_between_two_answers(
                first_answer, second_answer, largest_possible_rating_gap
            ),
        )
        for first_answer, first_count in first_answer_counts.items()
        for second_answer, second_count in second_answer_counts.items()
    ) / (sum(first_answer_counts.values()) * sum(second_answer_counts.values()))


def _add_to_count(counts: Counter, key: Any, change: int) -> None:
    counts[key] += change
    if not counts[key]:
        del counts[key]


class _AnswerKindTally:
    """One evaluator's compared pairs of one `_answer_kind`: the answers on each
    side, the pairs at each largest possible rating gap, and each item's pair
    agreements."""

    def __init__(self, kappa_family: KappaFamily) -> None:
        self.first_answer_counts: Counter = Counter()
        # Fleiss draws both answers from one pool.
        self.second_answer_counts: Counter = (
            self.first_answer_counts if kappa_family == "fleiss" else Counter()
        )
        self.rating_gap_to_pair_count: Counter = Counter()
        self.item_id_to_agreements: Dict[str, List[float]] = {}

    def count(
        self,
        item_id: str,
        largest_possible_rating_gap: float,
        answer_pairs_compared: List[Tuple[Any, Any, float]],
        change: int,
    ) -> None:
        if change > 0:
            self.item_id_to_agreements[item_id] = _agreements(answer_pairs_compared)
        else:
            del self.item_id_to_agreements[item_id]
        _add_to_count(
            self.rating_gap_to_pair_count,
            largest_possible_rating_gap,
            change * len(answer_pairs_compared),
        )
        for first_answer, second_answer, _agreement in answer_pairs_compared:
            _add_to_count(self.first_answer_counts, first_answer, change)
            _add_to_count(self.second_answer_counts, second_answer, change)


class _KappaTally:
    """The answer counts behind each kind's kappa, kept current as one (item,
    evaluator)'s compared pairs are replaced, so a trend does not recount every
    pair at every bucket."""

    def __init__(self, kappa_family: KappaFamily) -> None:
        self.kappa_family = kappa_family
        self.item_and_evaluator_to_answer_pairs_compared: Dict[
            Tuple[str, Optional[str]], AnswerPairsCompared
        ] = {}
        self.answer_kind_and_evaluator_to_tally: Dict[
            Tuple[str, Optional[str]], _AnswerKindTally
        ] = {}

    def replace(
        self,
        item_and_evaluator: Tuple[str, Optional[str]],
        answer_pairs_compared: AnswerPairsCompared,
    ) -> None:
        previous = self.item_and_evaluator_to_answer_pairs_compared.get(item_and_evaluator)
        if previous is not None:
            self._count(item_and_evaluator, previous, -1)
        self.item_and_evaluator_to_answer_pairs_compared[item_and_evaluator] = (
            answer_pairs_compared
        )
        self._count(item_and_evaluator, answer_pairs_compared, 1)

    def _count(
        self,
        item_and_evaluator: Tuple[str, Optional[str]],
        answer_pairs_compared: AnswerPairsCompared,
        change: int,
    ) -> None:
        item_id, evaluator_id = item_and_evaluator
        largest_possible_rating_gap, pairs = answer_pairs_compared
        answer_kind_to_pairs: Dict[str, List[Tuple[Any, Any, float]]] = defaultdict(list)
        for pair in pairs:
            answer_kind_to_pairs[_answer_kind(pair[0])].append(pair)
        for answer_kind, pairs_of_kind in answer_kind_to_pairs.items():
            tally = self.answer_kind_and_evaluator_to_tally.get((answer_kind, evaluator_id))
            if tally is None:
                tally = _AnswerKindTally(self.kappa_family)
                self.answer_kind_and_evaluator_to_tally[(answer_kind, evaluator_id)] = tally
            tally.count(item_id, largest_possible_rating_gap, pairs_of_kind, change)

    def kappa_by_kind(self) -> List[Dict[str, Any]]:
        """Kappa once for binary answers (any answer that either matches or does
        not) and once for ratings, each kind only when it was compared, binary
        first. Answers are pooled by their own `_answer_kind`, so a pass/fail
        answer is never drawn against a text one. Chance agreement is worked out
        for each evaluator, answer kind and largest possible rating gap from
        that evaluator's answers on every item, over the same gap the agreement
        used, then averaged weighted by pair count. An evaluator counts only
        when its pairs of the kind are on two items or more: one item cannot
        show chance, and a Cohen's kappa on it would always read 0. Each entry's
        `agreement` and `pair_count` cover the pairs its kappa uses, or every
        pair of the kind when no evaluator counts, in which case the kappa is
        blank with `too_few_items` and no chance agreement. It is blank with
        `everyone_agreed` when chance agreement is 1."""
        reported_kind_to_evaluator_to_tallies: Dict[
            str, Dict[Optional[str], List[_AnswerKindTally]]
        ] = defaultdict(lambda: defaultdict(list))
        for (answer_kind, evaluator_id), tally in self.answer_kind_and_evaluator_to_tally.items():
            if tally.item_id_to_agreements:
                reported_kind_to_evaluator_to_tallies[_reported_answer_kind(answer_kind)][
                    evaluator_id
                ].append(tally)
        kappa_by_kind: List[Dict[str, Any]] = []
        for reported_kind in ("binary", "rating"):
            evaluator_to_tallies = reported_kind_to_evaluator_to_tallies.get(reported_kind)
            if not evaluator_to_tallies:
                continue
            tallies_counted = [
                tally
                for tallies in evaluator_to_tallies.values()
                if _compared_on_two_items_or_more(tallies)
                for tally in tallies
            ]
            tallies_shown = tallies_counted or [
                tally for tallies in evaluator_to_tallies.values() for tally in tallies
            ]
            kind_agreement, kind_pair_count = _mean_agreement(
                (
                    agreement
                    for tally in tallies_shown
                    for agreements in tally.item_id_to_agreements.values()
                    for agreement in agreements
                ),
                rounded=False,
            )
            kind_kappa: Dict[str, Any] = {
                "answer_kind": reported_kind,
                "agreement": _round_agreement(kind_agreement),
                "pair_count": kind_pair_count,
                "kappa": None,
                "kappa_type": (
                    f"linear_weighted_{self.kappa_family}"
                    if reported_kind == "rating"
                    else self.kappa_family
                ),
                "kappa_blank_reason": None,
                "chance_agreement": None,
            }
            kappa_by_kind.append(kind_kappa)
            if not tallies_counted:
                kind_kappa["kappa_blank_reason"] = "too_few_items"
                continue
            chance_agreement = (
                math.fsum(
                    _chance_agreement(
                        tally.first_answer_counts,
                        tally.second_answer_counts,
                        largest_possible_rating_gap,
                    )
                    * pair_count_at_gap
                    for tally in tallies_counted
                    for largest_possible_rating_gap, pair_count_at_gap in (
                        tally.rating_gap_to_pair_count.items()
                    )
                )
                / kind_pair_count
            )
            kind_kappa["chance_agreement"] = _round_agreement(chance_agreement)
            if 1 - chance_agreement < 1e-12:
                kind_kappa["kappa_blank_reason"] = "everyone_agreed"
                continue
            kind_kappa["kappa"] = _round_agreement(
                (kind_agreement - chance_agreement) / (1 - chance_agreement)
            )
        return kappa_by_kind


def _reported_answer_kind(answer_kind: Optional[str]) -> str:
    return "rating" if answer_kind == "numeric" else "binary"


def _compared_on_two_items_or_more(tallies: List[_AnswerKindTally]) -> bool:
    item_ids: set = set()
    for tally in tallies:
        item_ids.update(tally.item_id_to_agreements)
        if len(item_ids) >= 2:
            return True
    return False


_NO_KAPPA: Dict[str, Any] = dict.fromkeys(KAPPA_FIELDS)


def kappa_of_one_kind(kappa_by_kind: List[Dict[str, Any]]) -> Dict[str, Any]:
    """`kappa`, `kappa_type`, `kappa_blank_reason` and `chance_agreement` of a
    number over one evaluator, whose answers are all of one kind. All blank
    when nothing was compared, or when the evaluator was answered with both
    kinds."""
    if len(kappa_by_kind) != 1:
        return dict(_NO_KAPPA)
    return {key: kappa_by_kind[0][key] for key in _NO_KAPPA}


def _trend_point(
    bucket_start_and_end: Dict[str, str],
    agreement: Optional[float],
    pair_count: int,
    kappa_by_kind: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """A trend point for a number over one evaluator, with its kappa fields."""
    return {
        **bucket_start_and_end,
        "agreement": agreement,
        "pair_count": pair_count,
        **kappa_of_one_kind(kappa_by_kind),
    }


def agreement_and_kappa_between_annotators(
    annotations: Iterable[Dict[str, Any]],
    rating_scales: RatingScales = _NO_RATING_SCALES,
    with_kappa: bool = True,
) -> AgreementAndKappa:
    """Mean agreement between annotators, over every item and evaluator that
    two or more of them answered, how many pairs of answers that covers, and
    the Fleiss' kappa of each kind of answer. Ratings use each evaluator's live
    rating scale from `rating_scales` (see `rating_scales_for`). Returns
    (None, 0, []) when nothing can be compared."""
    kappa_tally = _KappaTally("fleiss")
    agreements: List[float] = []
    for item_and_evaluator, annotator_id_to_answer in _annotator_answers_per_item_and_evaluator(
        annotations
    ).items():
        answer_pairs_compared = _answer_pairs_compared_between_annotators(
            list(annotator_id_to_answer.values()), rating_scales.get(item_and_evaluator[1])
        )
        agreements.extend(_agreements(answer_pairs_compared[1]))
        if with_kappa:
            kappa_tally.replace(item_and_evaluator, answer_pairs_compared)
    return (*_mean_agreement(agreements), kappa_tally.kappa_by_kind())


def agreement_and_kappa_of_annotator_with_others(
    annotations: Iterable[Dict[str, Any]],
    annotator_id: str,
    rating_scales: RatingScales = _NO_RATING_SCALES,
) -> AgreementAndKappa:
    """Mean agreement between `annotator_id` and every other annotator, over
    the items and evaluators they both answered, how many pairs that covers,
    and the Cohen's kappa of each kind of answer. Each pair is `annotator_id`
    against one other annotator."""
    kappa_tally = _KappaTally("cohen")
    agreements_with_others: List[float] = []
    for item_and_evaluator, annotator_id_to_answer in _annotator_answers_per_item_and_evaluator(
        annotations
    ).items():
        if annotator_id not in annotator_id_to_answer:
            continue
        answer_pairs_compared = _answer_pairs_compared_between_one_answer_and_others(
            annotator_id_to_answer[annotator_id],
            [
                other_answer
                for other_annotator_id, other_answer in annotator_id_to_answer.items()
                if other_annotator_id != annotator_id
            ],
            rating_scales.get(item_and_evaluator[1]),
        )
        agreements_with_others.extend(_agreements(answer_pairs_compared[1]))
        kappa_tally.replace(item_and_evaluator, answer_pairs_compared)
    return (
        *_mean_agreement(agreements_with_others),
        kappa_tally.kappa_by_kind(),
    )


class _AgreementBetweenJudgeAndAnnotatorsSoFar:
    """`agreement_and_kappa_between_judge_and_annotators` over every row added
    so far, rescoring only the (item, evaluator) pairs that new rows answer or
    rescore."""

    def __init__(
        self, evaluator_id: str, rating_scales: RatingScales, with_kappa: bool = True
    ) -> None:
        self.evaluator_id = evaluator_id
        self.with_kappa = with_kappa
        self.rating_scales = rating_scales
        self.item_and_evaluator_to_annotator_answers: Dict[
            Tuple[str, Optional[str]], Dict[str, Any]
        ] = {}
        self.item_and_evaluator_to_judge_answer: Dict[
            Tuple[str, str], Tuple[Any, str, Optional[str]]
        ] = {}
        self.item_and_evaluator_to_agreements: Dict[Tuple[str, str], List[float]] = {}
        self.kappa_tally = _KappaTally("cohen")

    def add_rows(
        self,
        annotations: Iterable[Dict[str, Any]],
        evaluator_runs: Iterable[Dict[str, Any]],
    ) -> None:
        item_and_evaluator_pairs_changed = {
            **_record_annotator_answers(
                self.item_and_evaluator_to_annotator_answers,
                annotations,
                only_evaluator_id=self.evaluator_id,
            ),
            **_record_latest_judge_answers(
                self.item_and_evaluator_to_judge_answer,
                evaluator_runs,
                only_evaluator_id=self.evaluator_id,
            ),
        }
        for item_and_evaluator in item_and_evaluator_pairs_changed:
            judge_answer_scored_at_and_version_id = (
                self.item_and_evaluator_to_judge_answer.get(item_and_evaluator)
            )
            annotator_id_to_answer = self.item_and_evaluator_to_annotator_answers.get(
                item_and_evaluator
            )
            if judge_answer_scored_at_and_version_id is None or not annotator_id_to_answer:
                continue
            judge_answer, _scored_at, judge_version_id = (
                judge_answer_scored_at_and_version_id
            )
            answer_pairs_compared = _answer_pairs_compared_between_one_answer_and_others(
                judge_answer,
                annotator_id_to_answer.values(),
                self.rating_scales.get(
                    judge_version_id, self.rating_scales.get(self.evaluator_id)
                ),
            )
            self.item_and_evaluator_to_agreements[item_and_evaluator] = _agreements(
                answer_pairs_compared[1]
            )
            if self.with_kappa:
                self.kappa_tally.replace(item_and_evaluator, answer_pairs_compared)

    def agreements(self) -> Iterator[float]:
        for agreements in self.item_and_evaluator_to_agreements.values():
            yield from agreements

    def agreement_and_kappa(self) -> AgreementAndKappa:
        return (*_mean_agreement(self.agreements()), self.kappa_tally.kappa_by_kind())


def agreement_and_kappa_between_judge_and_annotators(
    annotations: Iterable[Dict[str, Any]],
    evaluator_runs: Iterable[Dict[str, Any]],
    evaluator_id: str,
    rating_scales: RatingScales = _NO_RATING_SCALES,
    with_kappa: bool = True,
) -> AgreementAndKappa:
    """Mean agreement between one evaluator's judge and the annotators, over
    every item both answered, how many pairs of answers that covers, and the
    Cohen's kappa of each kind of answer. On each item the judge's latest
    score is compared with each annotator's latest answer. A rating gap is
    divided by the largest possible gap on the scale, from `rating_scales`, of
    the version the judge score came from. Returns (None, 0, []) when nothing
    can be compared."""
    agreement_so_far = _AgreementBetweenJudgeAndAnnotatorsSoFar(
        evaluator_id, rating_scales, with_kappa
    )
    agreement_so_far.add_rows(annotations, evaluator_runs)
    return agreement_so_far.agreement_and_kappa()


def agreement_between_judge_and_annotators_over_time_by_version_and_task(
    annotations: List[Dict[str, Any]],
    evaluator_runs: List[Dict[str, Any]],
    evaluator_id: str,
    version_ids: List[str],
    task_ids: List[str],
    bucket: str,
    days: int,
    now: Optional[datetime] = None,
    rating_scales: RatingScales = _NO_RATING_SCALES,
) -> Dict[str, Any]:
    """For one evaluator, judge-vs-annotator agreement at the end of each time
    bucket: overall, for each of `version_ids`, and for each of `task_ids`, in
    one walk through the buckets.

    `evaluator_runs` must carry a `task_id` field (as returned by
    `get_evaluator_runs_for_evaluator_org_scoped`).

    Returns:
        {
            "overall":    [{bucket_start, bucket_end, agreement, pair_count, kappa, kappa_type, kappa_blank_reason, chance_agreement}, ...],
            "by_version": {version_id: [...]},
            "by_task":    {task_id:    [...]},
        }
    """
    overall_agreement_so_far = _AgreementBetweenJudgeAndAnnotatorsSoFar(
        evaluator_id, rating_scales
    )
    version_id_to_agreement_so_far = {
        version_id: _AgreementBetweenJudgeAndAnnotatorsSoFar(evaluator_id, rating_scales)
        for version_id in version_ids
    }
    task_id_to_agreement_so_far = {
        task_id: _AgreementBetweenJudgeAndAnnotatorsSoFar(evaluator_id, rating_scales)
        for task_id in task_ids
    }

    overall_agreement_over_time: List[Dict[str, Any]] = []
    version_id_to_agreement_over_time: Dict[str, List[Dict[str, Any]]] = {
        version_id: [] for version_id in version_ids
    }
    task_id_to_agreement_over_time: Dict[str, List[Dict[str, Any]]] = {
        task_id: [] for task_id in task_ids
    }

    for (
        bucket_start_and_end,
        (annotations_new_in_bucket, evaluator_runs_new_in_bucket),
    ) in _rows_new_in_each_time_bucket(
        bucket,
        days,
        now,
        [
            (annotations, _ANNOTATION_TIMESTAMP_FIELDS),
            (evaluator_runs, _EVALUATOR_RUN_TIMESTAMP_FIELDS),
        ],
    ):
        overall_agreement_so_far.add_rows(
            annotations_new_in_bucket, evaluator_runs_new_in_bucket
        )
        for version_id, agreement_so_far in version_id_to_agreement_so_far.items():
            agreement_so_far.add_rows(
                annotations_new_in_bucket,
                [
                    run
                    for run in evaluator_runs_new_in_bucket
                    if run.get("evaluator_version_id") == version_id
                ],
            )
        for task_id, agreement_so_far in task_id_to_agreement_so_far.items():
            agreement_so_far.add_rows(
                [
                    annotation
                    for annotation in annotations_new_in_bucket
                    if annotation.get("task_id") == task_id
                ],
                [run for run in evaluator_runs_new_in_bucket if run.get("task_id") == task_id],
            )

        overall_agreement_over_time.append(
            _trend_point(bucket_start_and_end, *overall_agreement_so_far.agreement_and_kappa())
        )
        for version_id in version_ids:
            version_id_to_agreement_over_time[version_id].append(
                _trend_point(
                    bucket_start_and_end,
                    *version_id_to_agreement_so_far[version_id].agreement_and_kappa(),
                )
            )
        for task_id in task_ids:
            task_id_to_agreement_over_time[task_id].append(
                _trend_point(
                    bucket_start_and_end, *task_id_to_agreement_so_far[task_id].agreement_and_kappa()
                )
            )

    return {
        "overall": overall_agreement_over_time,
        "by_version": version_id_to_agreement_over_time,
        "by_task": task_id_to_agreement_over_time,
    }


def _rows_per_evaluator(rows: Iterable[Dict[str, Any]]) -> Dict[Optional[str], List[Dict[str, Any]]]:
    evaluator_id_to_rows: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        evaluator_id_to_rows[row.get("evaluator_id")].append(row)
    return evaluator_id_to_rows


def agreement_response(
    annotations: List[Dict[str, Any]],
    evaluator_runs: List[Dict[str, Any]],
    evaluators: List[Dict[str, Any]],
    rating_scales: RatingScales = _NO_RATING_SCALES,
) -> Dict[str, Any]:
    """What both agreement endpoints return for `evaluators` (rows with `uuid`
    and `name`): `human_human`, the annotators with each other over every
    evaluator; `evaluators_with_humans`, every evaluator's judge against the
    annotators, weighted by pair count; and `evaluators`, for each one its
    judge against the annotators and the annotators with each other on it
    (`inter_annotator`), each with its kappa."""
    human_agreement, human_pair_count = agreement_and_kappa_between_annotators(
        annotations, rating_scales
    )[:2]
    evaluator_id_to_annotations = _rows_per_evaluator(annotations)
    evaluator_id_to_evaluator_runs = _rows_per_evaluator(evaluator_runs)
    agreement_per_evaluator = []
    judge_agreement_so_far_per_evaluator = []
    for evaluator in evaluators:
        evaluator_id = evaluator["uuid"]
        judge_agreement_so_far = _AgreementBetweenJudgeAndAnnotatorsSoFar(
            evaluator_id, rating_scales
        )
        judge_agreement_so_far.add_rows(
            evaluator_id_to_annotations.get(evaluator_id, ()),
            evaluator_id_to_evaluator_runs.get(evaluator_id, ()),
        )
        judge_agreement_so_far_per_evaluator.append(judge_agreement_so_far)
        judge_agreement, judge_pair_count, judge_kappa_by_kind = (
            judge_agreement_so_far.agreement_and_kappa()
        )
        annotator_agreement, annotator_pair_count, annotator_kappa_by_kind = (
            agreement_and_kappa_between_annotators(
                evaluator_id_to_annotations.get(evaluator_id, ()), rating_scales
            )
        )
        agreement_per_evaluator.append(
            {
                "evaluator_id": evaluator_id,
                "name": evaluator.get("name"),
                "current": judge_agreement,
                "pair_count": judge_pair_count,
                **kappa_of_one_kind(judge_kappa_by_kind),
                "inter_annotator": {
                    "current": annotator_agreement,
                    "pair_count": annotator_pair_count,
                    **kappa_of_one_kind(annotator_kappa_by_kind),
                },
            }
        )
    judges_agreement, judges_pair_count = _mean_agreement(
        agreement
        for judge_agreement_so_far in judge_agreement_so_far_per_evaluator
        for agreement in judge_agreement_so_far.agreements()
    )
    return {
        "human_human": {"current": human_agreement, "pair_count": human_pair_count},
        "evaluators_with_humans": {"current": judges_agreement, "pair_count": judges_pair_count},
        "evaluators": agreement_per_evaluator,
    }
