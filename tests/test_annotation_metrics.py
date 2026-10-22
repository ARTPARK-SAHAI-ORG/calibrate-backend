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
    agreement_and_kappa_between_annotators,
    agreement_and_kappa_between_judge_and_annotators,
    agreement_and_kappa_of_annotator_with_others,
    agreement_and_kappa_of_each_judge_with_annotators,
    agreement_of_each_evaluator_with_annotators,
    kappa_of_one_kind,
    trend_without_kappa,
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
    assert agreement_and_kappa_between_annotators([])[:2] == (None, 0)
    # Two annotators agreeing on one slot
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "b", "value": {"value": True}},
    ]
    agree, pairs = agreement_and_kappa_between_annotators(annotations)[:2]
    assert agree == 1.0 and pairs == 1
    # Skip rows missing scalar / annotator
    annotations.append({"item_id": "i2", "evaluator_id": "e1", "annotator_id": None, "value": {"value": True}})
    annotations.append({"item_id": "i3", "evaluator_id": "e1", "annotator_id": "a", "value": {"other": 1}})
    agree, pairs = agreement_and_kappa_between_annotators(annotations)[:2]
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
    agree, pairs = agreement_and_kappa_between_judge_and_annotators([], [], "e1")[:2]
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
    agree, pairs = agreement_and_kappa_between_judge_and_annotators(annotations, runs, "e1")[:2]
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
        return agreement_and_kappa_between_annotators(annotations)[1] > 0 or any(
            agreement_and_kappa_between_judge_and_annotators(
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
    assert agreement_and_kappa_of_annotator_with_others([], "a")[:2] == (None, 0)
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
    agree, pairs = agreement_and_kappa_of_annotator_with_others(annotations, "a")[:2]
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
    no_evaluators, no_evaluators_overall = agreement_between_judge_and_annotators_over_time(
        [], [], [], "week", 14, now=now
    )
    assert no_evaluators == {}
    assert no_evaluators_overall and all(b["agreement"] is None for b in no_evaluators_overall)

    # No data path
    empty, _ = agreement_between_judge_and_annotators_over_time([], [], ["e1"], "week", 14, now=now)
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
    series, _ = agreement_between_judge_and_annotators_over_time(
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
        "kappa": None,
        "kappa_type": "cohen",
        "kappa_blank_reason": "too_few_items",
        "chance_agreement": None,
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
    assert agreement_and_kappa_between_annotators(annotations, rating_scales)[:2] == (0.375, 2)
    # An evaluator missing from the map falls back for its own slot only.
    assert agreement_and_kappa_between_annotators(annotations, {"other": _ONE_TO_FIVE})[:2] == (0.0, 2)


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
    assert agreement_and_kappa_between_annotators(annotations)[:2] == (0.0, 1)
    assert agreement_and_kappa_of_annotator_with_others(annotations, "a")[:2] == (0.0, 1)
    # Observed range over the evaluator's 5 and the humans' 3 and 4 is 2.
    assert agreement_and_kappa_between_judge_and_annotators(annotations, runs, "e1")[:2] == (0.25, 2)
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
    assert agreement_and_kappa_between_annotators(annotations, rating_scales)[:2] == (0.75, 1)
    assert agreement_and_kappa_of_annotator_with_others(annotations, "a", rating_scales)[:2] == (0.75, 1)
    # Evaluator said 5: 1 - 2/4 against the 3, 1 - 1/4 against the 4.
    assert agreement_and_kappa_between_judge_and_annotators(annotations, runs, "e1", rating_scales)[:2] == (
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
        )[0]["e1"][-1]["agreement"]
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
    assert agreement_and_kappa_between_judge_and_annotators(annotations, runs, "e1", rating_scales)[:2] == (
        round(1 - 1 / 3, 4),
        1,
    )


def test_agreement_never_goes_below_zero():
    # Labels 3 apart, given before the rubric was narrowed to 1 to 2.
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": a, "value": {"value": v}}
        for a, v in (("a", 1), ("b", 4))
    ]
    assert agreement_and_kappa_between_annotators(annotations, {"e1": (1, 2)})[:2] == (0.0, 1)


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
    expected_points = []
    expected_annotator_series = []
    expected_annotator_with_others_series = []
    expected_evaluator_id_to_series = {identifier: [] for identifier in requested_evaluator_ids}
    expected_all_judges_series = []
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

        def point(agreement_and_kappa):
            agreement, pair_count, kappa_by_kind = agreement_and_kappa
            expected_points.append({
                **bucket_start_and_end, "agreement": agreement, "pair_count": pair_count,
                **kappa_of_one_kind(kappa_by_kind),
            })
            return expected_points[-1]

        def point_by_kind(agreement_and_kappa):
            agreement, pair_count, kappa_by_kind = agreement_and_kappa
            expected_points.append({
                **bucket_start_and_end, "agreement": agreement, "pair_count": pair_count,
                "kappa_by_kind": kappa_by_kind,
            })
            return expected_points[-1]

        annotations_so_far = _rows_timestamped_up_to(annotations, ("updated_at", "created_at"), bucket_end)
        evaluator_runs_so_far = _rows_timestamped_up_to(evaluator_runs, ("completed_at", "created_at"), bucket_end)
        expected_annotator_series.append(
            point(agreement_and_kappa_between_annotators(annotations_so_far, rating_scales))
        )
        expected_annotator_with_others_series.append(
            point_by_kind(agreement_and_kappa_of_annotator_with_others(annotations_so_far, annotator_id, rating_scales))
        )
        _, (agreement_of_all_judges, pair_count_of_all_judges) = (
            agreement_and_kappa_of_each_judge_with_annotators(
                annotations_so_far, evaluator_runs_so_far, requested_evaluator_ids, rating_scales
            )
        )
        expected_all_judges_series.append({
            **bucket_start_and_end,
            "agreement": agreement_of_all_judges,
            "pair_count": pair_count_of_all_judges,
        })
        for identifier in requested_evaluator_ids:
            expected_evaluator_id_to_series[identifier].append(point(
                agreement_and_kappa_between_judge_and_annotators(annotations_so_far, evaluator_runs_so_far, identifier, rating_scales)
            ))
        expected_breakdown["overall"].append(point(
            agreement_and_kappa_between_judge_and_annotators(annotations_so_far, evaluator_runs_so_far, evaluator_id, rating_scales)
        ))
        for version_id in requested_version_ids:
            expected_breakdown["by_version"][version_id].append(point(
                agreement_and_kappa_between_judge_and_annotators(
                    annotations_so_far,
                    [run for run in evaluator_runs_so_far if run.get("evaluator_version_id") == version_id],
                    evaluator_id,
                    rating_scales,
                )
            ))
        for task_id in requested_task_ids:
            expected_breakdown["by_task"][task_id].append(point(
                agreement_and_kappa_between_judge_and_annotators(
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
    ) == (expected_evaluator_id_to_series, expected_all_judges_series)
    assert agreement_between_judge_and_annotators_over_time_by_version_and_task(
        annotations, evaluator_runs, evaluator_id, requested_version_ids, requested_task_ids,
        bucket, days, now=now, rating_scales=rating_scales,
    ) == expected_breakdown
    return expected_annotator_series, expected_points


def _check_a_random_trend_case(randomizer, answer_choices, now):
    """Build one random set of rows and check every trend against fresh
    results at each bucket end. Returns every expected point."""
    bucket_and_days_choices = [("week", 35), ("month", 120), ("year", 800)]
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
    _annotator_series, expected_points = _assert_over_time_agreement_equals_agreement_on_rows_up_to_each_bucket_end(
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
    return expected_points


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
    order_sensitive_series, _order_sensitive_points = _assert_over_time_agreement_equals_agreement_on_rows_up_to_each_bucket_end(
        order_sensitive_annotations, [], {"e5": (1, 5), "e4": (1, 4)}, "a", "e5",
        ["e5"], [], [], "week", 35, now,
    )
    assert order_sensitive_series[-1]["agreement"] == 0.7187

    mixed_answer_choices = [
        True, False, {"value": True}, {"value": False}, 1, 2, 3, 5, 2.5,
        {"value": 4}, {"score": 1}, "good", "bad", None, {"value": None},
        {"comment": "no answer"},
    ]
    randomizer = random.Random(20261022)
    for _case in range(400):
        _check_a_random_trend_case(randomizer, mixed_answer_choices, now)

    # One kind of answer per case, plus an answer of no known kind that is
    # never compared, so most points have a kappa to check.
    single_kind_answer_choices = [
        [True, False, {"value": True}, {"value": False}, None, {"value": [1]}],
        [1, 2, 3, 5, 2.5, {"value": 4}, {"score": 1}, {"value": None}, {"value": [1]}],
    ]
    single_kind_randomizer = random.Random(20261023)
    kappa_point_count = 0
    for _case in range(400):
        expected_points = _check_a_random_trend_case(
            single_kind_randomizer,
            single_kind_randomizer.choice(single_kind_answer_choices),
            now,
        )
        kappa_point_count += sum(1 for point in expected_points if point.get("kappa") is not None)
    assert kappa_point_count > 500


def _annotations_from_answers(answers_per_item, evaluator_id="e1", item_prefix="i"):
    return [
        {
            "item_id": f"{item_prefix}{item_index}",
            "evaluator_id": evaluator_id,
            "annotator_id": f"a{annotator_index}",
            "value": {"value": answer},
        }
        for item_index, answers in enumerate(answers_per_item)
        for annotator_index, answer in enumerate(answers)
    ]


def _judge_runs_from_answers(answers, evaluator_id="e1"):
    return [
        {
            "item_id": f"i{item_index}",
            "evaluator_id": evaluator_id,
            "value": {"value": answer},
            "created_at": "2024-01-01 00:00:00",
        }
        for item_index, answer in enumerate(answers)
    ]


def _kappa(answer_kind, kappa_type, kappa, kappa_blank_reason=None, chance_agreement=None):
    return {
        "answer_kind": answer_kind,
        "kappa": kappa,
        "kappa_type": kappa_type,
        "kappa_blank_reason": kappa_blank_reason,
        "chance_agreement": chance_agreement,
    }


def test_fleiss_kappa_on_pass_fail_answers():
    annotations = _annotations_from_answers(
        [(True, True), (True, True), (True, False), (False, False)]
    )
    # Chance: 5 of 8 answers are true, so (5/8)^2 + (3/8)^2 = 0.53125.
    assert agreement_and_kappa_between_annotators(annotations) == (
        0.75, 4, [_kappa("binary", "fleiss", 0.4667, chance_agreement=0.5312)]
    )
    assert round((0.75 - 0.53125) / (1 - 0.53125), 4) == 0.4667


def test_cohen_kappa_between_judge_and_annotator():
    annotations = _annotations_from_answers([(True,), (False,), (False,), (False,)])
    runs = _judge_runs_from_answers([True, True, False, False])
    # Chance: 0.5 * 0.25 + 0.5 * 0.75 = 0.5.
    assert agreement_and_kappa_between_judge_and_annotators(annotations, runs, "e1") == (
        0.75, 4, [_kappa("binary", "cohen", 0.5, chance_agreement=0.5)]
    )


def test_cohen_kappa_of_one_annotator_with_the_others():
    annotations = _annotations_from_answers(
        [(True, True), (True, True), (True, False), (False, False)]
    )
    # a0 said true 3 of 4 times and a1 2 of 4: chance 0.75 * 0.5 + 0.25 * 0.5.
    assert agreement_and_kappa_of_annotator_with_others(annotations, "a0") == (
        0.75, 4, [_kappa("binary", "cohen", 0.5, chance_agreement=0.5)]
    )


def test_linear_weighted_cohen_kappa_on_a_one_to_five_scale():
    annotations = _annotations_from_answers([(4,), (4,), (1,), (1,)])
    runs = _judge_runs_from_answers([5, 4, 2, 1])
    # Agreement: 1 - 1/4, 1, 1 - 1/4, 1, so 0.875.
    # Chance: each judge answer 1/4 of the time, each annotator answer 1/2,
    # and the weights of the 8 combinations add up to 4.5, so 4.5 / 8 = 0.5625.
    assert agreement_and_kappa_between_judge_and_annotators(
        annotations, runs, "e1", {"e1": _ONE_TO_FIVE}
    ) == (0.875, 4, [_kappa("rating", "linear_weighted_cohen", 0.7143, chance_agreement=0.5625)])
    assert round((0.875 - 0.5625) / (1 - 0.5625), 4) == 0.7143


def test_chance_agreement_widens_the_gap_to_the_pooled_ratings():
    # No scale: each item divides by its own gap of 1, so both disagree fully.
    # Pooled, the ratings 1, 2, 2, 3 span 2, and chance agreement over that gap
    # is 0.625.
    annotations = _annotations_from_answers([(1, 2), (2, 3)])
    assert agreement_and_kappa_between_annotators(annotations) == (
        0.0,
        2,
        [_kappa("rating", "linear_weighted_fleiss", round(-0.625 / 0.375, 4), chance_agreement=0.625)],
    )


def test_two_kinds_of_evaluator_each_get_their_own_kappa():
    annotations = _annotations_from_answers([(True, True), (False, False)]) + (
        _annotations_from_answers([(1, 5), (5, 5)], evaluator_id="e2", item_prefix="r")
    )
    # Binary: both pairs agree, chance 0.5. Rating: agreement 0 and 1, so 0.5,
    # chance 1/16 + 9/16, so (0.5 - 0.625) / 0.375. The agreement covers all
    # four pairs.
    assert agreement_and_kappa_between_annotators(annotations, {"e2": _ONE_TO_FIVE}) == (
        0.75,
        4,
        [
            _kappa("binary", "fleiss", 1.0, chance_agreement=0.5),
            _kappa("rating", "linear_weighted_fleiss", -0.3333, chance_agreement=0.625),
        ],
    )


def test_kappa_is_empty_when_nothing_was_compared():
    assert agreement_and_kappa_between_annotators([]) == (None, 0, [])
    assert agreement_and_kappa_between_judge_and_annotators([], [], "e1") == (None, 0, [])
    assert agreement_and_kappa_of_annotator_with_others([], "a0") == (None, 0, [])
    assert kappa_of_one_kind([]) == {
        "kappa": None, "kappa_type": None, "kappa_blank_reason": None, "chance_agreement": None
    }


def test_kappa_is_blank_because_everyone_agreed_when_every_answer_is_the_same():
    every_answer_the_same = _annotations_from_answers([(True, True), (True, True)])
    assert agreement_and_kappa_between_annotators(every_answer_the_same) == (
        1.0, 2, [_kappa("binary", "fleiss", None, "everyone_agreed", 1.0)]
    )
    assert agreement_and_kappa_between_judge_and_annotators(
        _annotations_from_answers([(True,), (True,)]), _judge_runs_from_answers([True, True]), "e1"
    ) == (1.0, 2, [_kappa("binary", "cohen", None, "everyone_agreed", 1.0)])


def test_one_evaluator_answered_with_both_kinds_has_no_single_kappa():
    one_evaluator_with_two_kinds = _annotations_from_answers([(True, True), (3, 4)])
    agreement, pair_count, kappa_by_kind = agreement_and_kappa_between_annotators(
        one_evaluator_with_two_kinds, {"e1": _ONE_TO_FIVE}
    )
    assert (agreement, pair_count) == (0.875, 2)
    assert kappa_by_kind == [
        _kappa("binary", "fleiss", None, "too_few_items"),
        _kappa("rating", "linear_weighted_fleiss", None, "too_few_items"),
    ]
    assert kappa_of_one_kind(kappa_by_kind) == kappa_of_one_kind([])


def test_a_single_item_where_everyone_agreed_is_blank_for_too_few_items():
    assert agreement_and_kappa_between_annotators(
        _annotations_from_answers([[True, True, True]])
    ) == (1.0, 3, [_kappa("binary", "fleiss", None, "too_few_items")])
    assert agreement_and_kappa_of_annotator_with_others(
        _annotations_from_answers([[True, True]]), "a0"
    ) == (1.0, 1, [_kappa("binary", "cohen", None, "too_few_items")])


def test_fleiss_pool_counts_each_answer_once_per_pair_it_is_in():
    # Item A: 4 annotators say true, 6 pairs, each answer in 3 of them.
    # Items B and C: true against false, 1 pair each.
    # Agreement: 6 of 8 pairs agree. Pool: 12 + 1 + 1 = 14 true, 2 false, so
    # chance is (14/16)^2 + (2/16)^2 = 0.78125.
    annotations = _annotations_from_answers(
        [(True, True, True, True), (True, False), (True, False)]
    )
    assert agreement_and_kappa_between_annotators(annotations) == (
        0.75, 8, [_kappa("binary", "fleiss", -0.1429, chance_agreement=0.7812)]
    )
    assert round((0.75 - 0.78125) / (1 - 0.78125), 4) == -0.1429


def test_unknown_scale_pools_the_whole_evaluator():
    # Ratings 3, 3, 3, 3, 1, 5: pooled shares 4/6, 1/6, 1/6 over the spread of
    # 4, so chance is 16/36 + 2 * 4/36 * 0.5 * 2 + 1/36 + 1/36 = 26/36.
    # Agreement: 1, 1 and 0, so 2/3. Kappa: (2/3 - 26/36) / (10/36) = -0.2,
    # the same with or without the 1-to-5 scale.
    annotations = _annotations_from_answers([(3, 3), (3, 3), (1, 5)])
    expected = (0.6667, 3, [_kappa("rating", "linear_weighted_fleiss", -0.2, chance_agreement=0.7222)])
    assert agreement_and_kappa_between_annotators(annotations) == expected
    assert agreement_and_kappa_between_annotators(annotations, {"e1": _ONE_TO_FIVE}) == expected


def test_kappa_uses_only_the_answers_compared():
    # The rating 3 on the first item and the list on the second are never
    # compared, so they neither add a rating kappa nor raise.
    annotations = _annotations_from_answers(
        [(True, True, 3), (True, False, [1]), (False, False)]
    )
    # Pool: true 3 times, false 3 times, so chance 0.5. Agreement 2/3.
    assert agreement_and_kappa_between_annotators(annotations) == (
        0.6667, 3, [_kappa("binary", "fleiss", 0.3333, chance_agreement=0.5)]
    )
    # a0 said true, true, false and the others true, false, false: chance
    # 2/3 * 1/3 + 1/3 * 2/3 = 4/9, so (2/3 - 4/9) / (5/9) = 0.4.
    assert agreement_and_kappa_of_annotator_with_others(annotations, "a0") == (
        0.6667, 3, [_kappa("binary", "cohen", 0.4, chance_agreement=0.4444)]
    )
    # The judge's list on the second item compares with nothing. On the other
    # two it matches all four answers, half of them true.
    runs = _judge_runs_from_answers([True, [1], False])
    assert agreement_and_kappa_between_judge_and_annotators(annotations, runs, "e1") == (
        1.0, 4, [_kappa("binary", "cohen", 1.0, chance_agreement=0.5)]
    )


def _correctness_and_conciseness_annotations():
    """Correctness: both annotators say true on two items. Conciseness, 1 to
    5: (1, 1) and (1, 2)."""
    return _annotations_from_answers([(True, True), (True, True)], evaluator_id="correctness") + (
        _annotations_from_answers([(1, 1), (1, 2)], evaluator_id="conciseness", item_prefix="r")
    )


_CONCISENESS_SCALE = {"conciseness": _ONE_TO_FIVE}


def test_kinds_are_never_combined_into_one_kappa():
    annotations = _correctness_and_conciseness_annotations()
    # Pairs agree 1, 1, 1 and 1 - 1/4, so 0.9375 over both kinds.
    assert agreement_and_kappa_between_annotators(annotations, _CONCISENESS_SCALE)[:2] == (0.9375, 4)
    # a0 gave 1, 1 and the other 1, 2: agreement 0.875 and chance
    # 1/2 * 1 + 1/2 * 0.75 = 0.875, so kappa 0.
    assert agreement_and_kappa_of_annotator_with_others(annotations, "a0", _CONCISENESS_SCALE) == (
        0.9375,
        4,
        [
            _kappa("binary", "cohen", None, "everyone_agreed", 1.0),
            _kappa("rating", "linear_weighted_cohen", 0.0, chance_agreement=0.875),
        ],
    )


def test_each_evaluator_reports_its_inter_annotator_kappa_and_the_overall_is_weighted_by_pairs():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    annotations = [
        {**annotation, "updated_at": "2024-01-30 00:00:00"}
        for annotation in _correctness_and_conciseness_annotations()
    ]
    conciseness_judge = [
        {**run, "item_id": "r0", "evaluator_id": "conciseness", "completed_at": "2024-01-30 00:00:00"}
        for run in _judge_runs_from_answers([4])
    ]
    evaluator_id_to_agreement, agreement_of_all_evaluators = (
        agreement_of_each_evaluator_with_annotators(
            annotations, conciseness_judge, ["correctness", "conciseness"], "week", 7, now,
            _CONCISENESS_SCALE,
        )
    )
    conciseness = evaluator_id_to_agreement["conciseness"]
    # Annotator answers 1, 1, 1, 2: chance 0.75^2 + 2 * 0.75 * 0.25 * 0.75 +
    # 0.25^2 = 0.90625, so kappa (0.875 - 0.90625) / 0.09375.
    assert {key: value for key, value in conciseness["inter_annotator"].items() if key != "series"} == {
        "current": 0.875, "pair_count": 2, "kappa": -0.3333,
        "kappa_type": "linear_weighted_fleiss", "kappa_blank_reason": None,
        "chance_agreement": round(0.90625, 4),
    }
    assert conciseness["inter_annotator"]["series"][-1]["kappa"] == -0.3333
    assert {key: value for key, value in evaluator_id_to_agreement["correctness"]["inter_annotator"].items() if key != "series"} == {
        "current": 1.0, "pair_count": 2, "kappa": None, "kappa_type": "fleiss",
        "kappa_blank_reason": "everyone_agreed", "chance_agreement": 1.0,
    }
    # The judge's 4 against the two 1s: 1 - 3/4 each, on one item.
    assert {key: value for key, value in conciseness.items() if key not in ("series", "inter_annotator")} == {
        "current": 0.25, "pair_count": 2, "kappa": None, "kappa_type": "linear_weighted_cohen",
        "kappa_blank_reason": "too_few_items", "chance_agreement": None,
    }
    assert evaluator_id_to_agreement["correctness"]["current"] is None
    assert {key: value for key, value in agreement_of_all_evaluators.items() if key != "series"} == {
        "current": 0.25, "pair_count": 2
    }

    # A Correctness judge that agrees on both items adds 4 pairs at 1: the
    # overall is (2 * 0.25 + 4 * 1) / 6, not the mean of 0.25 and 1.
    correctness_judge = [
        {**run, "evaluator_id": "correctness", "completed_at": "2024-01-30 00:00:00"}
        for run in _judge_runs_from_answers([True, True])
    ]
    _, agreement_of_all_evaluators = agreement_of_each_evaluator_with_annotators(
        annotations, conciseness_judge + correctness_judge, ["correctness", "conciseness"],
        "week", 7, now, _CONCISENESS_SCALE,
    )
    assert agreement_of_all_evaluators["current"] == 0.75
    assert agreement_of_all_evaluators["pair_count"] == 6
    assert agreement_of_all_evaluators["series"][-1] == {
        **{key: agreement_of_all_evaluators["series"][-1][key] for key in ("bucket_start", "bucket_end")},
        "agreement": 0.75,
        "pair_count": 6,
    }


def test_trend_without_kappa_keeps_only_the_bucket_agreement_and_pair_count():
    point = {
        "bucket_start": "a", "bucket_end": "b", "agreement": 0.5, "pair_count": 2,
        "kappa": 0.1, "kappa_type": "fleiss", "kappa_blank_reason": None, "chance_agreement": 0.4,
    }
    assert trend_without_kappa([point]) == [
        {"bucket_start": "a", "bucket_end": "b", "agreement": 0.5, "pair_count": 2}
    ]


def _textbook_fleiss_kappa(answers_per_item, categories, weight):
    """Fleiss' kappa for items that all have the same number of raters, with
    an agreement weight between two categories (1 for an exact match)."""
    rater_count = len(answers_per_item[0])
    item_count = len(answers_per_item)
    category_counts_per_item = [
        {category: list(answers).count(category) for category in categories}
        for answers in answers_per_item
    ]
    observed_agreement = sum(
        (
            sum(
                weight(first, second) * counts[first] * counts[second]
                for first in categories
                for second in categories
            )
            - rater_count
        )
        / (rater_count * (rater_count - 1))
        for counts in category_counts_per_item
    ) / item_count
    category_shares = {
        category: sum(counts[category] for counts in category_counts_per_item)
        / (item_count * rater_count)
        for category in categories
    }
    expected_agreement = sum(
        weight(first, second) * category_shares[first] * category_shares[second]
        for first in categories
        for second in categories
    )
    if 1 - expected_agreement < 1e-12:
        return None
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def test_kappa_is_blank_when_only_one_item_was_compared():
    # One judge answer against one annotator answer: chance equals the
    # agreement itself, so kappa would always read 0.
    assert agreement_and_kappa_between_judge_and_annotators(
        _annotations_from_answers([(1,)]),
        _judge_runs_from_answers([0]),
        "e1",
        {"e1": (0, 2)},
    ) == (0.5, 1, [_kappa("rating", "linear_weighted_cohen", None, "too_few_items")])
    assert agreement_and_kappa_between_annotators(
        _annotations_from_answers([[True, False, False]])
    )[2] == [_kappa("binary", "fleiss", None, "too_few_items")]
    two_items = _annotations_from_answers([[True, False], [True, True]])
    assert agreement_and_kappa_between_annotators(two_items)[2] == [
        _kappa("binary", "fleiss", -0.3333, chance_agreement=0.625)
    ]


@pytest.mark.parametrize(
    "categories, weight, rating_scales, answer_kind, kappa_type",
    [
        ([True, False], lambda first, second: float(first == second), {}, "binary", "fleiss"),
        (
            [1, 2, 3, 4, 5],
            lambda first, second: 1 - abs(first - second) / 4,
            {"e1": _ONE_TO_FIVE},
            "rating",
            "linear_weighted_fleiss",
        ),
    ],
)
def test_fleiss_kappa_matches_the_textbook_formula_when_every_item_has_the_same_raters(
    categories, weight, rating_scales, answer_kind, kappa_type
):
    randomizer = random.Random(3141)
    kappa_count = 0
    for _case in range(300):
        rater_count = randomizer.randint(2, 5)
        answers_per_item = [
            [randomizer.choice(categories) for _rater in range(rater_count)]
            for _item in range(randomizer.randint(1, 8))
        ]
        agreement, _pair_count, [kind_kappa] = agreement_and_kappa_between_annotators(
            _annotations_from_answers(answers_per_item), rating_scales
        )
        assert (kind_kappa["answer_kind"], kind_kappa["kappa_type"]) == (answer_kind, kappa_type)
        expected_kappa = _textbook_fleiss_kappa(answers_per_item, categories, weight)
        if len(answers_per_item) == 1:
            assert kind_kappa == _kappa(answer_kind, kappa_type, None, "too_few_items")
            continue
        if expected_kappa is None:
            assert kind_kappa == _kappa(answer_kind, kappa_type, None, "everyone_agreed", 1.0)
            continue
        kappa_count += 1
        assert kind_kappa["kappa"] == pytest.approx(expected_kappa, abs=5.1e-5)
        assert kind_kappa["kappa_blank_reason"] is None
        chance_agreement = kind_kappa["chance_agreement"]
        assert kind_kappa["kappa"] == pytest.approx(
            (agreement - chance_agreement) / (1 - chance_agreement), abs=1e-3
        )
    assert kappa_count > 250
