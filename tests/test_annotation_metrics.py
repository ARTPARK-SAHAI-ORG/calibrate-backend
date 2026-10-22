"""Tests for src/annotation_metrics.py — pure pairwise agreement math."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import random
import uuid

import pytest

from annotation_metrics import (
    _answer_kind,
    rating_scales_for,
    _mean_agreement_between_answers,
    _extract_answer,
    agreement_between_annotators,
    agreement_of_annotator_with_others,
    agreement_between_judge_and_annotators,
    build_time_buckets,
    judge_and_annotators_agreement_on_item,
    judge_answer_summary,
    keep_runs_from_live_versions,
    has_answers_to_compare,
    annotator_answer_summary,
    agreement_on_item,
    keep_runs_from_live_versions_of,
    agreement_between_annotators_over_time,
    agreement_between_judge_and_annotators_over_time_by_version_and_task,
    agreement_of_annotator_with_others_over_time,
    agreement_between_judge_and_annotators_over_time,
)


def test_scalar_extracts_value_variants():
    assert _extract_answer(None) is None
    assert _extract_answer(1) == 1
    assert _extract_answer(True) is True
    assert _extract_answer(3.14) == 3.14
    assert _extract_answer("foo") == "foo"
    assert _extract_answer({"value": 5, "reasoning": "x"}) == 5
    assert _extract_answer({"score": 4}) == 4
    assert _extract_answer({"label": "good"}) == "good"
    assert _extract_answer({"binary": True}) is True
    assert _extract_answer({"other": "x"}) is None
    assert _extract_answer([1, 2]) is None


def test_classify_kinds():
    assert _answer_kind(None) is None
    assert _answer_kind(True) == "binary"
    assert _answer_kind(1) == "numeric"
    assert _answer_kind(3.0) == "numeric"
    assert _answer_kind("x") == "categorical"
    assert _answer_kind([1, 2]) is None


def test_pairwise_agreement_branches():
    # < 2 → 0
    assert _mean_agreement_between_answers([]) == (0.0, 0)
    assert _mean_agreement_between_answers([1]) == (0.0, 0)
    # binary match
    mean, pairs = _mean_agreement_between_answers([True, True])
    assert mean == 1.0 and pairs == 1
    # binary mismatch
    mean, pairs = _mean_agreement_between_answers([True, False])
    assert mean == 0.0 and pairs == 1
    # numeric with span = max - min
    mean, pairs = _mean_agreement_between_answers([1, 5])
    assert pairs == 1 and mean == 0.0
    # numeric all equal -> span 1, agreement = 1
    mean, pairs = _mean_agreement_between_answers([3, 3, 3])
    assert pairs == 3 and mean == 1.0
    # categorical
    mean, pairs = _mean_agreement_between_answers(["a", "a", "b"])
    assert pairs == 3 and 0 < mean < 1
    # mixed types skip mismatched pairs
    mean, pairs = _mean_agreement_between_answers([True, 1])
    # bool/int don't match in _answer_kind (bool is binary, int is numeric)
    assert pairs == 0
    # all None → 0
    assert _mean_agreement_between_answers([None, None]) == (0.0, 0)


def test_pairwise_agreement_skips_unclassifiable_values():
    # The public labelling route stores whatever value it is handed, so a slot
    # can hold a list. Two of those are not a comparable pair.
    assert _mean_agreement_between_answers([[1], [1]]) == (0.0, 0)
    assert _mean_agreement_between_answers([[1], "a"]) == (0.0, 0)
    # A real pair alongside a list still scores on its own.
    assert _mean_agreement_between_answers([[1], True, True]) == (1.0, 1)


def test_filter_runs_to_live_versions():
    runs = [
        {"evaluator_id": "e1", "evaluator_version_id": "v-live"},
        {"evaluator_id": "e1", "evaluator_version_id": "v-stale"},
        {"evaluator_id": "e2", "evaluator_version_id": "vx"},  # not in map
        {"evaluator_id": None, "evaluator_version_id": "v"},  # no eval id
    ]
    out = keep_runs_from_live_versions(runs, {"e1": "v-live", "e2": None})
    assert len(out) == 1
    assert out[0]["evaluator_id"] == "e1"


def test_runs_at_live_versions():
    runs = [
        {"evaluator_id": "e1", "evaluator_version_id": "v-live"},
        {"evaluator_id": "e1", "evaluator_version_id": "v-stale"},
        {"evaluator_id": "e2", "evaluator_version_id": "v2"},
    ]
    evaluators = [
        {"uuid": "e1", "live_version_id": "v-live"},
        {"uuid": "e2", "live_version_id": None},
    ]
    out = keep_runs_from_live_versions_of(runs, evaluators)
    assert out == [{"evaluator_id": "e1", "evaluator_version_id": "v-live"}]


def test_aggregate_agreement_empty_and_full():
    assert agreement_between_annotators([]) == (None, 0)
    # Two annotators agreeing on one slot
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "b", "value": {"value": True}},
    ]
    agree, pairs = agreement_between_annotators(annotations)
    assert agree == 1.0 and pairs == 1
    # Skip rows missing scalar / annotator
    annotations.append({"item_id": "i2", "evaluator_id": "e1", "annotator_id": None, "value": {"value": True}})
    annotations.append({"item_id": "i3", "evaluator_id": "e1", "annotator_id": "a", "value": {"other": 1}})
    agree, pairs = agreement_between_annotators(annotations)
    assert pairs == 1


def test_build_buckets_validation():
    with pytest.raises(ValueError):
        build_time_buckets("decade", 7)
    with pytest.raises(ValueError):
        build_time_buckets("week", 0)
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    buckets = build_time_buckets("week", 14, now=now)
    assert len(buckets) == 2
    # Buckets are returned chronologically
    assert buckets[0][1] <= buckets[1][1]


def test_evaluator_human_pair_agreement():
    # eval=binary, humans=binary
    total, pairs = judge_and_annotators_agreement_on_item(True, [True, False])
    assert pairs == 2 and total == 1.0
    # eval=None classification
    assert judge_and_annotators_agreement_on_item(None, [True]) == (0.0, 0)
    # eval present but no humans
    assert judge_and_annotators_agreement_on_item(True, []) == (0.0, 0)
    # eval = numeric; mismatched-type humans are skipped
    total, pairs = judge_and_annotators_agreement_on_item(3, [3, "foo"])
    assert pairs == 1


def test_evaluator_result_summary():
    assert judge_answer_summary([], "e1") == {
        "count": 0,
        "true_count": None,
        "mean": None,
    }

    binary_runs = [
        {"item_id": "i1", "evaluator_id": "e1", "value": {"value": True}},
        {"item_id": "i2", "evaluator_id": "e1", "value": {"value": False}},
        # Latest run wins for the same item
        {"item_id": "i1", "evaluator_id": "e1", "value": {"value": False}, "completed_at": "2024-02-01 00:00:00"},
        # Other evaluator, and a failed run with no value
        {"item_id": "i3", "evaluator_id": "e2", "value": {"value": True}},
        {"item_id": "i4", "evaluator_id": "e1", "value": None},
    ]
    assert judge_answer_summary(binary_runs, "e1") == {
        "count": 2,
        "true_count": 0,
        "mean": None,
    }

    rating_runs = [
        {"item_id": "i1", "evaluator_id": "e1", "value": {"value": 4}},
        {"item_id": "i2", "evaluator_id": "e1", "value": {"value": 3}},
    ]
    assert judge_answer_summary(rating_runs, "e1") == {
        "count": 2,
        "true_count": None,
        "mean": 3.5,
    }


def test_aggregate_human_evaluator_agreement_empty():
    agree, pairs = agreement_between_judge_and_annotators([], [], "e1")
    assert agree is None and pairs == 0

    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
    ]
    # Latest-per-slot picks the most recent evaluator run for the slot
    runs = [
        {"item_id": "i1", "evaluator_id": "e1", "evaluator_version_id": "v1", "value": {"value": True}, "created_at": "2024-01-01 00:00:00"},
        {"item_id": "i1", "evaluator_id": "e1", "evaluator_version_id": "v1", "value": {"value": False}, "completed_at": "2024-02-01 00:00:00"},
        # Different evaluator — filtered out
        {"item_id": "i1", "evaluator_id": "e2", "evaluator_version_id": "v", "value": {"value": True}},
        # Missing fields — filtered
        {"evaluator_id": "e1", "value": {"value": True}},
    ]
    agree, pairs = agreement_between_judge_and_annotators(annotations, runs, "e1")
    # Latest evaluator value is False; human said True → disagreement
    assert agree == 0.0 and pairs == 1


def test_has_any_comparable_pair_human_human():
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "b", "value": {"value": False}},
    ]
    assert has_answers_to_compare(annotations, [], ["e1"]) is True


def test_has_any_comparable_pair_human_evaluator():
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
    ]
    runs = [
        {"item_id": "i1", "evaluator_id": "e1", "value": {"value": True}, "created_at": "2024-01-01 00:00:00"},
    ]
    assert has_answers_to_compare(annotations, runs, ["e1"]) is True


def test_has_any_comparable_pair_none():
    # Single annotation, no evaluator run → no pair
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
    ]
    assert has_answers_to_compare(annotations, [], ["e1"]) is False
    # Empty inputs
    assert has_answers_to_compare([], [], []) is False
    assert has_answers_to_compare([], [], ["e1"]) is False


def test_has_any_comparable_pair_accepts_iterables():
    # Generators are consumed multiple times internally — must be materialized.
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
    ]
    runs = [
        {"item_id": "i1", "evaluator_id": "e1", "value": {"value": True}, "created_at": "2024-01-01 00:00:00"},
    ]
    ann_gen = (a for a in annotations)
    run_gen = (r for r in runs)
    ev_gen = (e for e in ["e1"])
    assert has_answers_to_compare(ann_gen, run_gen, ev_gen) is True



def test_has_answers_to_compare_matches_scoring_every_pair():
    random_generator = random.Random(749)
    possible_stored_values = [
        True, False, 0, 1, 3, 2.5, "yes", "no", [1, 2], None,
        {"value": [1, 2]}, {"value": 4}, {"value": False}, {"score": True},
        {"label": "a"}, {"rating": 2}, {"comment": "only a comment"},
        {"value": None, "score": 5}, {"value": {"nested": 1}},
    ]
    possible_timestamps = ["2024-01-01 00:00:00", "2024-01-02 00:00:00", None]

    def random_annotation():
        return {
            "item_id": random_generator.choice(["i1", "i2", "i3"]),
            "evaluator_id": random_generator.choice(["e1", "e2", "e3", None]),
            "annotator_id": random_generator.choice(["a1", "a2", "a3", None]),
            "value": random_generator.choice(possible_stored_values),
        }

    def random_evaluator_run():
        return {
            "item_id": random_generator.choice(["i1", "i2", "i3", None]),
            "evaluator_id": random_generator.choice(["e1", "e2", "e3", None, ""]),
            "value": random_generator.choice(possible_stored_values),
            "completed_at": random_generator.choice(possible_timestamps),
            "created_at": random_generator.choice(possible_timestamps),
            "evaluator_version_id": random_generator.choice(["v1", "v2", None]),
        }

    def compare_by_scoring_every_pair(annotations, evaluator_runs, evaluator_ids):
        return agreement_between_annotators(annotations)[1] > 0 or any(
            agreement_between_judge_and_annotators(
                annotations, evaluator_runs, evaluator_id
            )[1] > 0
            for evaluator_id in evaluator_ids
        )

    results_seen = set()
    for _ in range(3000):
        annotations = [
            random_annotation() for _ in range(random_generator.randint(0, 6))
        ]
        evaluator_runs = [
            random_evaluator_run() for _ in range(random_generator.randint(0, 6))
        ]
        evaluator_ids = random_generator.sample(
            ["e1", "e2", "e3", "evaluator_with_no_runs", None],
            random_generator.randint(0, 3),
        )
        expected = compare_by_scoring_every_pair(
            annotations, evaluator_runs, evaluator_ids
        )
        if random_generator.random() < 0.5:
            actual = has_answers_to_compare(
                iter(annotations), iter(evaluator_runs), iter(evaluator_ids)
            )
        else:
            actual = has_answers_to_compare(annotations, evaluator_runs, evaluator_ids)
        assert actual is expected, (annotations, evaluator_runs, evaluator_ids)
        results_seen.add(actual)
    assert results_seen == {True, False}

def test_per_item_agreement_shape():
    ann = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "b", "value": {"value": False}},
    ]
    runs = [
        {"item_id": "i1", "evaluator_id": "e1", "value": {"value": True}, "created_at": "2024-01-01 00:00:00"},
    ]
    out = agreement_on_item(ann, runs, ["e1"])
    assert "human_human" in out
    assert out["evaluators"][0]["evaluator_id"] == "e1"


def test_aggregate_agreement_for_annotator():
    # No data
    assert agreement_of_annotator_with_others([], "a") == (None, 0)
    annotations = [
        # Slot with annotator a + b
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "b", "value": {"value": True}},
        # Slot without target annotator — skipped
        {"item_id": "i2", "evaluator_id": "e1", "annotator_id": "c", "value": {"value": True}},
        # Numeric type
        {"item_id": "i3", "evaluator_id": "e1", "annotator_id": "a", "value": {"score": 3}},
        {"item_id": "i3", "evaluator_id": "e1", "annotator_id": "b", "value": {"score": 5}},
        # Mismatched type — skipped pair
        {"item_id": "i4", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
        {"item_id": "i4", "evaluator_id": "e1", "annotator_id": "b", "value": {"value": "yes"}},
        # No scalar — skipped
        {"item_id": "i5", "evaluator_id": "e1", "annotator_id": None, "value": None},
    ]
    agree, pairs = agreement_of_annotator_with_others(annotations, "a")
    assert pairs >= 2


def test_trend_series_empty_returns_buckets():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    series = agreement_between_annotators_over_time([], "week", 14, now=now)
    assert all(b["agreement"] is None for b in series)
    assert all(b["pair_count"] == 0 for b in series)


def test_trend_series_populated():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    annotations = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "annotator_id": "a",
            "value": {"value": True},
            "updated_at": "2024-01-25 12:00:00",
        },
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "annotator_id": "b",
            "value": {"value": True},
            "updated_at": "2024-01-26 12:00:00",
        },
        # Malformed timestamp → skipped
        {
            "item_id": "i2",
            "annotator_id": "a",
            "value": {"value": True},
            "updated_at": "garbage",
        },
        # No timestamp → skipped
        {"item_id": "i3", "annotator_id": "a", "value": {"value": True}},
    ]
    series = agreement_between_annotators_over_time(annotations, "week", 14, now=now)
    assert any(b["pair_count"] > 0 for b in series)


def test_trend_series_for_annotator():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    annotations = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "annotator_id": "a",
            "value": {"value": True},
            "updated_at": "2024-01-26 12:00:00",
        },
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "annotator_id": "b",
            "value": {"value": False},
            "updated_at": "2024-01-26 12:01:00",
        },
        {
            "item_id": "i2",
            "annotator_id": "a",
            "value": "garbage-ts",
            "updated_at": "not a date",
        },
    ]
    series = agreement_of_annotator_with_others_over_time(annotations, "a", "week", 14, now=now)
    assert len(series) == 2

    # Empty inputs go through the empty-series branch
    empty = agreement_of_annotator_with_others_over_time([], "a", "week", 14, now=now)
    assert all(b["agreement"] is None for b in empty)


def test_trend_series_human_evaluator_branches():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    # Empty evaluator list
    assert agreement_between_judge_and_annotators_over_time([], [], [], "week", 14, now=now) == {}

    # No data path
    empty = agreement_between_judge_and_annotators_over_time([], [], ["e1"], "week", 14, now=now)
    assert all(b["agreement"] is None for b in empty["e1"])

    annotations = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "annotator_id": "a",
            "value": {"value": True},
            "updated_at": "2024-01-26 12:00:00",
        }
    ]
    runs = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "evaluator_version_id": "v1",
            "value": {"value": True},
            "completed_at": "2024-01-26 13:00:00",
        },
        # malformed ts → skipped
        {
            "item_id": "i2",
            "evaluator_id": "e1",
            "evaluator_version_id": "v1",
            "value": {"value": True},
            "completed_at": "garbage",
        },
        # no ts → skipped
        {
            "item_id": "i3",
            "evaluator_id": "e1",
            "value": {"value": True},
        },
    ]
    series = agreement_between_judge_and_annotators_over_time(
        annotations, runs, ["e1"], "week", 14, now=now
    )
    assert "e1" in series


def test_trend_series_evaluator_breakdown():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    # Empty data path
    out = agreement_between_judge_and_annotators_over_time_by_version_and_task(
        [], [], "e1", ["v1"], ["t1"], "week", 14, now=now
    )
    assert "overall" in out
    assert all(b["agreement"] is None for b in out["overall"])

    annotations = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "annotator_id": "a",
            "value": {"value": True},
            "task_id": "t1",
            "updated_at": "2024-01-26 12:00:00",
        }
    ]
    runs = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "evaluator_version_id": "v1",
            "task_id": "t1",
            "value": {"value": True},
            "completed_at": "2024-01-26 12:00:00",
        },
        # malformed ts annotation
        {
            "item_id": "i2",
            "evaluator_id": "e1",
            "evaluator_version_id": "v1",
            "task_id": "t1",
            "value": {"value": True},
            "completed_at": "bad",
        },
        # no ts at all → skipped
        {
            "item_id": "i3",
            "evaluator_id": "e1",
            "evaluator_version_id": "v1",
            "task_id": "t1",
            "value": {"value": True},
        },
    ]
    out = agreement_between_judge_and_annotators_over_time_by_version_and_task(
        annotations, runs, "e1", ["v1"], ["t1"], "week", 14, now=now
    )
    assert "v1" in out["by_version"]
    assert "t1" in out["by_task"]


def test_trend_series_evaluator_breakdown_ignores_other_versions_and_tasks():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    annotations = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "annotator_id": "a",
            "value": {"value": True},
            "task_id": "t1",
            "updated_at": "2024-01-26 12:00:00",
        },
        # Another task, left out of the requested list
        {
            "item_id": "i2",
            "evaluator_id": "e1",
            "annotator_id": "a",
            "value": {"value": True},
            "task_id": "t2",
            "updated_at": "2024-01-26 12:00:00",
        },
    ]
    runs = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "evaluator_version_id": "v1",
            "task_id": "t1",
            "value": {"value": True},
            "completed_at": "2024-01-26 12:00:00",
        },
        # Another version and another task, both left out
        {
            "item_id": "i2",
            "evaluator_id": "e1",
            "evaluator_version_id": "v2",
            "task_id": "t2",
            "value": {"value": False},
            "completed_at": "2024-01-26 12:00:00",
        },
    ]
    out = agreement_between_judge_and_annotators_over_time_by_version_and_task(
        annotations, runs, "e1", ["v1"], ["t1"], "week", 14, now=now
    )
    # The overall series sees both items; the scoped series see only t1/v1.
    assert out["overall"][-1]["pair_count"] == 2
    assert out["by_version"]["v1"][-1] == {
        "bucket_start": out["overall"][-1]["bucket_start"],
        "bucket_end": out["overall"][-1]["bucket_end"],
        "agreement": 1.0,
        "pair_count": 1,
    }
    assert out["by_task"]["t1"][-1]["pair_count"] == 1
    assert out["by_task"]["t1"][-1]["agreement"] == 1.0


def test_human_result_summary():
    assert annotator_answer_summary([], "e1") is None

    binary = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a1", "value": {"value": True}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a2", "value": {"value": False}},
        # Later label from the same annotator on the same item replaces the first
        {"item_id": "i2", "evaluator_id": "e1", "annotator_id": "a1", "value": {"value": False}},
        {"item_id": "i2", "evaluator_id": "e1", "annotator_id": "a1", "value": {"value": True}},
        # Other evaluator, no annotator, and an empty value are all ignored
        {"item_id": "i3", "evaluator_id": "e2", "annotator_id": "a1", "value": {"value": True}},
        {"item_id": "i4", "evaluator_id": "e1", "annotator_id": None, "value": {"value": True}},
        {"item_id": "i5", "evaluator_id": "e1", "annotator_id": "a1", "value": None},
    ]
    assert annotator_answer_summary(binary, "e1") == {
        "count": 3,
        "true_count": 2,
        "mean": None,
    }

    rating = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a1", "value": {"value": 2}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a2", "value": {"value": 5}},
    ]
    assert annotator_answer_summary(rating, "e1") == {
        "count": 2,
        "true_count": None,
        "mean": 3.5,
    }


# ============ Rating divider: the evaluator's rubric width ============

_ONE_TO_FIVE = (1, 5)


def _rating_annotations(values_by_annotator, evaluator_id="e1"):
    return [
        {
            "item_id": "i1",
            "evaluator_id": evaluator_id,
            "annotator_id": annotator,
            "value": {"value": value},
        }
        for annotator, value in values_by_annotator.items()
    ]


def test_rating_pair_one_point_apart_scores_by_rubric_width():
    # 3 vs 4 on a 1-to-5 rubric: one step out of a width of 4 → 1 - 1/4.
    assert _mean_agreement_between_answers([3, 4], _ONE_TO_FIVE) == (0.75, 1)
    # The two ends of the same rubric agree on nothing.
    assert _mean_agreement_between_answers([1, 5], _ONE_TO_FIVE) == (0.0, 1)


def test_third_annotator_cannot_raise_the_first_pair():
    # Adding a 1 keeps 3-vs-4 at 0.75; mean = (0.75 + 0.5 + 0.25) / 3.
    mean, pairs = _mean_agreement_between_answers([3, 4, 1], _ONE_TO_FIVE)
    assert pairs == 3 and mean == 0.5
    # Without the rubric the outlier widens the divider and lifts 3-vs-4
    # from 0.0 to 2/3 — the behaviour the rubric width replaces.
    loose_mean, loose_pairs = _mean_agreement_between_answers([3, 4, 1])
    assert loose_pairs == 3 and round(loose_mean, 4) == 0.3333
    assert _mean_agreement_between_answers([3, 4]) == (0.0, 1)


def test_unusable_rubric_falls_back_to_the_observed_range():
    # A rating evaluator whose scale will not parse, and a zero-width one.
    assert _mean_agreement_between_answers([3, 4], (None, None)) == (0.0, 1)
    assert _mean_agreement_between_answers([3, 4], (3, 3)) == (0.0, 1)
    assert _mean_agreement_between_answers([3, 4], (5, 1)) == (0.0, 1)


def test_rating_scales_map_is_looked_up_per_slot():
    annotations = _rating_annotations({"a": 3, "b": 4}) + [
        # Row-level overall annotation: no evaluator, so no rubric to find.
        {"item_id": "i1", "evaluator_id": None, "annotator_id": "a", "value": {"value": 3}},
        {"item_id": "i1", "evaluator_id": None, "annotator_id": "b", "value": {"value": 4}},
    ]
    rating_scales = {"e1": _ONE_TO_FIVE}
    # e1's slot scores 0.75, the evaluator-less slot falls back to 0.0.
    assert agreement_between_annotators(annotations, rating_scales) == (0.375, 2)
    # An evaluator missing from the map falls back for its own slot only.
    assert agreement_between_annotators(annotations, {"other": _ONE_TO_FIVE}) == (0.0, 2)


def test_rating_agreement_unchanged_without_a_map():
    annotations = _rating_annotations({"a": 3, "b": 4})
    runs = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "value": {"value": 5},
            "created_at": "2024-01-01 00:00:00",
        }
    ]
    assert agreement_between_annotators(annotations) == (0.0, 1)
    assert agreement_of_annotator_with_others(annotations, "a") == (0.0, 1)
    # Observed range over the evaluator's 5 and the humans' 3 and 4 is 2.
    assert agreement_between_judge_and_annotators(annotations, runs, "e1") == (0.25, 2)
    assert judge_and_annotators_agreement_on_item(5, [3, 4]) == (0.5, 2)
    item = agreement_on_item(annotations, runs, ["e1"])
    assert item["human_human"]["agreement"] == 0.0
    assert item["evaluators"][0]["agreement"] == 0.25


def test_rating_agreement_with_a_map():
    annotations = _rating_annotations({"a": 3, "b": 4})
    runs = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "value": {"value": 5},
            "created_at": "2024-01-01 00:00:00",
        }
    ]
    rating_scales = {"e1": _ONE_TO_FIVE}
    assert agreement_between_annotators(annotations, rating_scales) == (0.75, 1)
    assert agreement_of_annotator_with_others(annotations, "a", rating_scales) == (0.75, 1)
    # Evaluator said 5: 1 - 2/4 against the 3, 1 - 1/4 against the 4.
    assert agreement_between_judge_and_annotators(annotations, runs, "e1", rating_scales) == (
        0.625,
        2,
    )
    assert judge_and_annotators_agreement_on_item(5, [3, 4], _ONE_TO_FIVE) == (1.25, 2)
    item = agreement_on_item(annotations, runs, ["e1"], rating_scales)
    assert item["human_human"]["agreement"] == 0.75
    assert item["evaluators"][0]["agreement"] == 0.625


def test_trend_series_carry_the_map_down():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    annotations = [
        {**a, "task_id": "t1", "updated_at": "2024-01-30 00:00:00"}
        for a in _rating_annotations({"a": 3, "b": 4})
    ]
    runs = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "evaluator_version_id": "v1",
            "task_id": "t1",
            "value": {"value": 5},
            "completed_at": "2024-01-30 00:00:00",
        }
    ]
    rating_scales = {"e1": _ONE_TO_FIVE}
    assert agreement_between_annotators_over_time(annotations, "week", 7, now=now)[-1]["agreement"] == 0.0
    assert (
        agreement_between_annotators_over_time(annotations, "week", 7, now=now, rating_scales=rating_scales)[-1]["agreement"]
        == 0.75
    )
    assert (
        agreement_of_annotator_with_others_over_time(
            annotations, "a", "week", 7, now=now, rating_scales=rating_scales
        )[-1]["agreement"]
        == 0.75
    )
    assert (
        agreement_between_judge_and_annotators_over_time(
            annotations, runs, ["e1"], "week", 7, now=now, rating_scales=rating_scales
        )["e1"][-1]["agreement"]
        == 0.625
    )
    breakdown = agreement_between_judge_and_annotators_over_time_by_version_and_task(
        annotations,
        runs,
        "e1",
        version_ids=["v1"],
        task_ids=["t1"],
        bucket="week",
        days=7,
        now=now,
        rating_scales=rating_scales,
    )
    assert breakdown["overall"][-1]["agreement"] == 0.625
    assert breakdown["by_version"]["v1"][-1]["agreement"] == 0.625
    assert breakdown["by_task"]["t1"][-1]["agreement"] == 0.625


def test_rating_scales_for_reads_the_live_version_and_each_judge_version():
    import db as db_mod

    user_uuid = db_mod.create_user(
        "Scale", "Bounds", f"scale-{uuid.uuid4().hex[:8]}@example.com"
    )
    org = db_mod.get_personal_org_for_user(user_uuid)
    rating = db_mod.create_evaluator(
        name=f"rating-{uuid.uuid4().hex[:8]}",
        output_type="rating",
        owner_user_id=user_uuid,
        org_uuid=org["uuid"],
    )
    stale = db_mod.create_evaluator_version(
        rating,
        judge_model="m",
        system_prompt="p",
        output_config={"scale": [{"value": i, "name": str(i)} for i in (1, 2, 3)]},
    )
    live = db_mod.create_evaluator_version(
        rating,
        judge_model="m",
        system_prompt="p",
        output_config={"scale": [{"value": i, "name": str(i)} for i in (1, 3, 5)]},
    )
    db_mod.set_evaluator_live_version(rating, live["uuid"])
    binary = db_mod.create_evaluator(
        name=f"binary-{uuid.uuid4().hex[:8]}",
        owner_user_id=user_uuid,
        org_uuid=org["uuid"],
    )

    annotations = [
        {"evaluator_id": e} for e in (rating, binary, None, "missing-evaluator")
    ]
    runs = [{"evaluator_id": rating, "evaluator_version_id": stale["uuid"]}]
    rating_scales = rating_scales_for(annotations, runs)
    # Human labels record no version, so they get the live rubric.
    assert rating_scales[rating] == (1, 5)
    # A judge score gets the rubric of the version that gave it.
    assert rating_scales[stale["uuid"]] == (1, 3)
    # A binary evaluator has no live version here, and an unknown id is absent.
    assert binary not in rating_scales and "missing-evaluator" not in rating_scales
    assert rating_scales_for([]) == {}


def test_judge_score_is_divided_by_its_own_versions_width():
    # The judge scored on an older 1-to-4 version; the live version rates 1 to 2.
    rating_scales = {"e1": (1, 2), "v-old": (1, 4)}
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": 1}}
    ]
    runs = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "evaluator_version_id": "v-old",
            "value": {"value": 2},
            "completed_at": "2024-01-01 00:00:00",
        }
    ]
    assert agreement_between_judge_and_annotators(annotations, runs, "e1", rating_scales) == (
        round(1 - 1 / 3, 4),
        1,
    )


def test_agreement_never_goes_below_zero():
    # Labels 3 apart, given before the rubric was narrowed to 1 to 2.
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": a, "value": {"value": v}}
        for a, v in (("a", 1), ("b", 4))
    ]
    assert agreement_between_annotators(annotations, {"e1": (1, 2)}) == (0.0, 1)


def _rows_timestamped_up_to(rows, timestamp_fields, bucket_end):
    """Rows whose first set timestamp field parses and falls on or before
    `bucket_end`, oldest first, ties in their given order."""
    timestamped_rows = []
    for row in rows:
        timestamp = next((row.get(field) for field in timestamp_fields if row.get(field)), None)
        try:
            parsed_timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except (TypeError, ValueError):
            continue
        if parsed_timestamp <= bucket_end:
            timestamped_rows.append((parsed_timestamp, row))
    timestamped_rows.sort(key=lambda timestamp_and_row: timestamp_and_row[0])
    return [row for _timestamp, row in timestamped_rows]


def _assert_over_time_agreement_equals_agreement_on_rows_up_to_each_bucket_end(
    annotations, evaluator_runs, rating_scales, annotator_id, evaluator_id,
    requested_evaluator_ids, requested_version_ids, requested_task_ids, bucket, days, now,
):
    expected_annotator_series = []
    expected_annotator_with_others_series = []
    expected_evaluator_id_to_series = {identifier: [] for identifier in requested_evaluator_ids}
    expected_breakdown = {
        "overall": [],
        "by_version": {identifier: [] for identifier in requested_version_ids},
        "by_task": {identifier: [] for identifier in requested_task_ids},
    }
    for bucket_start, bucket_end in build_time_buckets(bucket, days, now=now):
        bucket_start_and_end = {
            "bucket_start": bucket_start.strftime("%Y-%m-%d %H:%M:%S"),
            "bucket_end": bucket_end.strftime("%Y-%m-%d %H:%M:%S"),
        }

        def point(agreement_and_pair_count):
            agreement, pair_count = agreement_and_pair_count
            return {**bucket_start_and_end, "agreement": agreement, "pair_count": pair_count}

        annotations_so_far = _rows_timestamped_up_to(annotations, ("updated_at", "created_at"), bucket_end)
        evaluator_runs_so_far = _rows_timestamped_up_to(evaluator_runs, ("completed_at", "created_at"), bucket_end)
        expected_annotator_series.append(
            point(agreement_between_annotators(annotations_so_far, rating_scales))
        )
        expected_annotator_with_others_series.append(
            point(agreement_of_annotator_with_others(annotations_so_far, annotator_id, rating_scales))
        )
        for identifier in requested_evaluator_ids:
            expected_evaluator_id_to_series[identifier].append(point(
                agreement_between_judge_and_annotators(annotations_so_far, evaluator_runs_so_far, identifier, rating_scales)
            ))
        expected_breakdown["overall"].append(point(
            agreement_between_judge_and_annotators(annotations_so_far, evaluator_runs_so_far, evaluator_id, rating_scales)
        ))
        for version_id in requested_version_ids:
            expected_breakdown["by_version"][version_id].append(point(
                agreement_between_judge_and_annotators(
                    annotations_so_far,
                    [run for run in evaluator_runs_so_far if run.get("evaluator_version_id") == version_id],
                    evaluator_id,
                    rating_scales,
                )
            ))
        for task_id in requested_task_ids:
            expected_breakdown["by_task"][task_id].append(point(
                agreement_between_judge_and_annotators(
                    [annotation for annotation in annotations_so_far if annotation.get("task_id") == task_id],
                    [run for run in evaluator_runs_so_far if run.get("task_id") == task_id],
                    evaluator_id,
                    rating_scales,
                )
            ))

    assert agreement_between_annotators_over_time(
        annotations, bucket, days, now=now, rating_scales=rating_scales
    ) == expected_annotator_series
    assert agreement_of_annotator_with_others_over_time(
        annotations, annotator_id, bucket, days, now=now, rating_scales=rating_scales
    ) == expected_annotator_with_others_series
    assert agreement_between_judge_and_annotators_over_time(
        annotations, evaluator_runs, requested_evaluator_ids, bucket, days, now=now, rating_scales=rating_scales
    ) == expected_evaluator_id_to_series
    assert agreement_between_judge_and_annotators_over_time_by_version_and_task(
        annotations, evaluator_runs, evaluator_id, requested_version_ids, requested_task_ids,
        bucket, days, now=now, rating_scales=rating_scales,
    ) == expected_breakdown
    return expected_annotator_series


def test_over_time_agreement_equals_agreement_on_rows_up_to_each_bucket_end():
    now = datetime(2024, 6, 30, 12, tzinfo=timezone.utc)

    # These 16 agreements, added one at a time, total 11.499999999999998, so the
    # mean rounds to 0.7187; a correctly rounded sum gives 11.5 and 0.7188.
    rating_gap_and_scale_per_item = [
        (1, "e5"), (0, "e5"), (0, "e5"), (0, "e5"), (3, "e5"), (2, "e5"), (0, "e5"), (0, "e5"),
        (2, "e4"), (0, "e5"), (0, "e5"), (0, "e5"), (1, "e4"), (2, "e5"), (2, "e5"), (4, "e5"),
    ]
    order_sensitive_annotations = []
    for item_index, (rating_gap, evaluator_id) in enumerate(rating_gap_and_scale_per_item):
        updated_at = (now - timedelta(days=30 - item_index)).strftime("%Y-%m-%d %H:%M:%S")
        for annotator_id, rating in (("a", 1), ("b", 1 + rating_gap)):
            order_sensitive_annotations.append({
                "item_id": f"i{item_index}", "evaluator_id": evaluator_id, "annotator_id": annotator_id,
                "value": {"value": rating}, "updated_at": updated_at,
            })
    order_sensitive_series = _assert_over_time_agreement_equals_agreement_on_rows_up_to_each_bucket_end(
        order_sensitive_annotations, [], {"e5": (1, 5), "e4": (1, 4)}, "a", "e5",
        ["e5"], [], [], "week", 35, now,
    )
    assert order_sensitive_series[-1]["agreement"] == 0.7187

    randomizer = random.Random(20261022)
    bucket_and_days_choices = [("week", 35), ("month", 120), ("year", 800)]
    answer_choices = [
        True, False, {"value": True}, {"value": False}, 1, 2, 3, 5, 2.5,
        {"value": 4}, {"score": 1}, "good", "bad", None, {"value": None},
        {"comment": "no answer"},
    ]
    for _case in range(400):
        bucket, days = randomizer.choice(bucket_and_days_choices)
        item_ids = [f"i{index}" for index in range(randomizer.randint(1, 6))]
        evaluator_ids = [f"e{index}" for index in range(randomizer.randint(1, 3))]
        annotator_ids = [f"a{index}" for index in range(randomizer.randint(1, 4))]
        version_ids = [f"v{index}" for index in range(randomizer.randint(1, 3))]
        task_ids = [f"t{index}" for index in range(randomizer.randint(1, 3))]
        # A small pool of moments makes ties; some fall before the window or after now.
        moments = [
            (now - timedelta(days=randomizer.uniform(-3, days + 40))).replace(microsecond=0)
            for _ in range(randomizer.randint(1, 8))
        ]

        def random_timestamp():
            roll = randomizer.random()
            if roll < 0.05:
                return None
            if roll < 0.08:
                return "not a timestamp"
            moment = randomizer.choice(moments)
            if roll < 0.11:
                # Parses, but sorts differently as text.
                return f"{moment.year}-{moment.month}-{moment.day} {moment.hour}:{moment.minute}:{moment.second}"
            return moment.strftime("%Y-%m-%d %H:%M:%S")

        def mostly(choices, rare_choice):
            return rare_choice if randomizer.random() < 0.1 else randomizer.choice(choices)

        annotations = []
        for _ in range(randomizer.randint(0, 25)):
            annotation = {
                "item_id": randomizer.choice(item_ids),
                "evaluator_id": randomizer.choice([*evaluator_ids, None]),
                "annotator_id": mostly(annotator_ids, None),
                "task_id": randomizer.choice(task_ids),
                "value": randomizer.choice(answer_choices),
                "created_at": random_timestamp(),
            }
            if randomizer.random() < 0.8:
                annotation["updated_at"] = random_timestamp()
            annotations.append(annotation)
        evaluator_runs = []
        for _ in range(randomizer.randint(0, 15)):
            run = {
                "item_id": mostly(item_ids, None),
                "evaluator_id": mostly(evaluator_ids, None),
                "evaluator_version_id": randomizer.choice([*version_ids, None]),
                "task_id": randomizer.choice(task_ids),
                "value": randomizer.choice(answer_choices),
                "created_at": random_timestamp(),
            }
            if randomizer.random() < 0.8:
                run["completed_at"] = random_timestamp()
            evaluator_runs.append(run)
        rating_scales = randomizer.choice([
            {},
            {
                identifier: randomizer.choice([(1, 5), (1, 3), (0, 10), (3, 3), (None, None)])
                for identifier in [*evaluator_ids, *version_ids]
                if randomizer.random() < 0.7
            },
        ])
        _assert_over_time_agreement_equals_agreement_on_rows_up_to_each_bucket_end(
            annotations,
            evaluator_runs,
            rating_scales,
            annotator_id=randomizer.choice([*annotator_ids, "never-annotated"]),
            evaluator_id=randomizer.choice(evaluator_ids),
            requested_evaluator_ids=randomizer.choices([*evaluator_ids, "unknown"], k=randomizer.randint(1, 3)),
            requested_version_ids=randomizer.choices([*version_ids, "unknown"], k=randomizer.randint(0, 3)),
            requested_task_ids=randomizer.choices([*task_ids, "unknown"], k=randomizer.randint(0, 3)),
            bucket=bucket,
            days=days,
            now=now,
        )
