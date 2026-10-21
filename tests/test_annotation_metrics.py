"""Tests for src/annotation_metrics.py — pure pairwise agreement math."""

from __future__ import annotations

from datetime import datetime, timezone
import uuid

import pytest

from annotation_metrics import (
    _classify,
    scales_for,
    _pairwise_agreement,
    _scalar,
    aggregate_agreement,
    aggregate_agreement_for_annotator,
    aggregate_human_evaluator_agreement,
    build_buckets,
    evaluator_human_pair_agreement,
    evaluator_result_summary,
    filter_runs_to_live_versions,
    has_any_comparable_pair,
    human_result_summary,
    per_item_agreement,
    runs_at_live_versions,
    trend_series,
    trend_series_evaluator_breakdown,
    trend_series_for_annotator,
    trend_series_human_evaluator,
)


def test_scalar_extracts_value_variants():
    assert _scalar(None) is None
    assert _scalar(1) == 1
    assert _scalar(True) is True
    assert _scalar(3.14) == 3.14
    assert _scalar("foo") == "foo"
    assert _scalar({"value": 5, "reasoning": "x"}) == 5
    assert _scalar({"score": 4}) == 4
    assert _scalar({"label": "good"}) == "good"
    assert _scalar({"binary": True}) is True
    assert _scalar({"other": "x"}) is None
    assert _scalar([1, 2]) is None


def test_classify_kinds():
    assert _classify(None) is None
    assert _classify(True) == "binary"
    assert _classify(1) == "numeric"
    assert _classify(3.0) == "numeric"
    assert _classify("x") == "categorical"
    assert _classify([1, 2]) is None


def test_pairwise_agreement_branches():
    # < 2 → 0
    assert _pairwise_agreement([]) == (0.0, 0)
    assert _pairwise_agreement([1]) == (0.0, 0)
    # binary match
    mean, pairs = _pairwise_agreement([True, True])
    assert mean == 1.0 and pairs == 1
    # binary mismatch
    mean, pairs = _pairwise_agreement([True, False])
    assert mean == 0.0 and pairs == 1
    # numeric with span = max - min
    mean, pairs = _pairwise_agreement([1, 5])
    assert pairs == 1 and mean == 0.0
    # numeric all equal -> span 1, agreement = 1
    mean, pairs = _pairwise_agreement([3, 3, 3])
    assert pairs == 3 and mean == 1.0
    # categorical
    mean, pairs = _pairwise_agreement(["a", "a", "b"])
    assert pairs == 3 and 0 < mean < 1
    # mixed types skip mismatched pairs
    mean, pairs = _pairwise_agreement([True, 1])
    # bool/int don't match in _classify (bool is binary, int is numeric)
    assert pairs == 0
    # all None → 0
    assert _pairwise_agreement([None, None]) == (0.0, 0)


def test_pairwise_agreement_skips_unclassifiable_values():
    # The public labelling route stores whatever value it is handed, so a slot
    # can hold a list. Two of those are not a comparable pair.
    assert _pairwise_agreement([[1], [1]]) == (0.0, 0)
    assert _pairwise_agreement([[1], "a"]) == (0.0, 0)
    # A real pair alongside a list still scores on its own.
    assert _pairwise_agreement([[1], True, True]) == (1.0, 1)


def test_filter_runs_to_live_versions():
    runs = [
        {"evaluator_id": "e1", "evaluator_version_id": "v-live"},
        {"evaluator_id": "e1", "evaluator_version_id": "v-stale"},
        {"evaluator_id": "e2", "evaluator_version_id": "vx"},  # not in map
        {"evaluator_id": None, "evaluator_version_id": "v"},  # no eval id
    ]
    out = filter_runs_to_live_versions(runs, {"e1": "v-live", "e2": None})
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
    out = runs_at_live_versions(runs, evaluators)
    assert out == [{"evaluator_id": "e1", "evaluator_version_id": "v-live"}]


def test_aggregate_agreement_empty_and_full():
    assert aggregate_agreement([]) == (None, 0)
    # Two annotators agreeing on one slot
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "b", "value": {"value": True}},
    ]
    agree, pairs = aggregate_agreement(annotations)
    assert agree == 1.0 and pairs == 1
    # Skip rows missing scalar / annotator
    annotations.append({"item_id": "i2", "evaluator_id": "e1", "annotator_id": None, "value": {"value": True}})
    annotations.append({"item_id": "i3", "evaluator_id": "e1", "annotator_id": "a", "value": {"other": 1}})
    agree, pairs = aggregate_agreement(annotations)
    assert pairs == 1


def test_build_buckets_validation():
    with pytest.raises(ValueError):
        build_buckets("decade", 7)
    with pytest.raises(ValueError):
        build_buckets("week", 0)
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    buckets = build_buckets("week", 14, now=now)
    assert len(buckets) == 2
    # Buckets are returned chronologically
    assert buckets[0][1] <= buckets[1][1]


def test_evaluator_human_pair_agreement():
    # eval=binary, humans=binary
    total, pairs = evaluator_human_pair_agreement(True, [True, False])
    assert pairs == 2 and total == 1.0
    # eval=None classification
    assert evaluator_human_pair_agreement(None, [True]) == (0.0, 0)
    # eval present but no humans
    assert evaluator_human_pair_agreement(True, []) == (0.0, 0)
    # eval = numeric; mismatched-type humans are skipped
    total, pairs = evaluator_human_pair_agreement(3, [3, "foo"])
    assert pairs == 1


def test_evaluator_result_summary():
    assert evaluator_result_summary([], "e1") == {
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
    assert evaluator_result_summary(binary_runs, "e1") == {
        "count": 2,
        "true_count": 0,
        "mean": None,
    }

    rating_runs = [
        {"item_id": "i1", "evaluator_id": "e1", "value": {"value": 4}},
        {"item_id": "i2", "evaluator_id": "e1", "value": {"value": 3}},
    ]
    assert evaluator_result_summary(rating_runs, "e1") == {
        "count": 2,
        "true_count": None,
        "mean": 3.5,
    }


def test_aggregate_human_evaluator_agreement_empty():
    agree, pairs = aggregate_human_evaluator_agreement([], [], "e1")
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
    agree, pairs = aggregate_human_evaluator_agreement(annotations, runs, "e1")
    # Latest evaluator value is False; human said True → disagreement
    assert agree == 0.0 and pairs == 1


def test_has_any_comparable_pair_human_human():
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "b", "value": {"value": False}},
    ]
    assert has_any_comparable_pair(annotations, [], ["e1"]) is True


def test_has_any_comparable_pair_human_evaluator():
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
    ]
    runs = [
        {"item_id": "i1", "evaluator_id": "e1", "value": {"value": True}, "created_at": "2024-01-01 00:00:00"},
    ]
    assert has_any_comparable_pair(annotations, runs, ["e1"]) is True


def test_has_any_comparable_pair_none():
    # Single annotation, no evaluator run → no pair
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
    ]
    assert has_any_comparable_pair(annotations, [], ["e1"]) is False
    # Empty inputs
    assert has_any_comparable_pair([], [], []) is False
    assert has_any_comparable_pair([], [], ["e1"]) is False


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
    assert has_any_comparable_pair(ann_gen, run_gen, ev_gen) is True


def test_per_item_agreement_shape():
    ann = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "b", "value": {"value": False}},
    ]
    runs = [
        {"item_id": "i1", "evaluator_id": "e1", "value": {"value": True}, "created_at": "2024-01-01 00:00:00"},
    ]
    out = per_item_agreement(ann, runs, ["e1"])
    assert "human_human" in out
    assert out["evaluators"][0]["evaluator_id"] == "e1"


def test_aggregate_agreement_for_annotator():
    # No data
    assert aggregate_agreement_for_annotator([], "a") == (None, 0)
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
    agree, pairs = aggregate_agreement_for_annotator(annotations, "a")
    assert pairs >= 2


def test_trend_series_empty_returns_buckets():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    series = trend_series([], "week", 14, now=now)
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
    series = trend_series(annotations, "week", 14, now=now)
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
    series = trend_series_for_annotator(annotations, "a", "week", 14, now=now)
    assert len(series) == 2

    # Empty inputs go through the empty-series branch
    empty = trend_series_for_annotator([], "a", "week", 14, now=now)
    assert all(b["agreement"] is None for b in empty)


def test_trend_series_human_evaluator_branches():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    # Empty evaluator list
    assert trend_series_human_evaluator([], [], [], "week", 14, now=now) == {}

    # No data path
    empty = trend_series_human_evaluator([], [], ["e1"], "week", 14, now=now)
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
    series = trend_series_human_evaluator(
        annotations, runs, ["e1"], "week", 14, now=now
    )
    assert "e1" in series


def test_trend_series_evaluator_breakdown():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    # Empty data path
    out = trend_series_evaluator_breakdown(
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
    out = trend_series_evaluator_breakdown(
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
    out = trend_series_evaluator_breakdown(
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
    assert human_result_summary([], "e1") is None

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
    assert human_result_summary(binary, "e1") == {
        "count": 3,
        "true_count": 2,
        "mean": None,
    }

    rating = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a1", "value": {"value": 2}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a2", "value": {"value": 5}},
    ]
    assert human_result_summary(rating, "e1") == {
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
    assert _pairwise_agreement([3, 4], _ONE_TO_FIVE) == (0.75, 1)
    # The two ends of the same rubric agree on nothing.
    assert _pairwise_agreement([1, 5], _ONE_TO_FIVE) == (0.0, 1)


def test_third_annotator_cannot_raise_the_first_pair():
    # Adding a 1 keeps 3-vs-4 at 0.75; mean = (0.75 + 0.5 + 0.25) / 3.
    mean, pairs = _pairwise_agreement([3, 4, 1], _ONE_TO_FIVE)
    assert pairs == 3 and mean == 0.5
    # Without the rubric the outlier widens the divider and lifts 3-vs-4
    # from 0.0 to 2/3 — the behaviour the rubric width replaces.
    loose_mean, loose_pairs = _pairwise_agreement([3, 4, 1])
    assert loose_pairs == 3 and round(loose_mean, 4) == 0.3333
    assert _pairwise_agreement([3, 4]) == (0.0, 1)


def test_unusable_rubric_falls_back_to_the_observed_range():
    # A rating evaluator whose scale will not parse, and a zero-width one.
    assert _pairwise_agreement([3, 4], (None, None)) == (0.0, 1)
    assert _pairwise_agreement([3, 4], (3, 3)) == (0.0, 1)
    assert _pairwise_agreement([3, 4], (5, 1)) == (0.0, 1)


def test_scales_map_is_looked_up_per_slot():
    annotations = _rating_annotations({"a": 3, "b": 4}) + [
        # Row-level overall annotation: no evaluator, so no rubric to find.
        {"item_id": "i1", "evaluator_id": None, "annotator_id": "a", "value": {"value": 3}},
        {"item_id": "i1", "evaluator_id": None, "annotator_id": "b", "value": {"value": 4}},
    ]
    scales = {"e1": _ONE_TO_FIVE}
    # e1's slot scores 0.75, the evaluator-less slot falls back to 0.0.
    assert aggregate_agreement(annotations, scales) == (0.375, 2)
    # An evaluator missing from the map falls back for its own slot only.
    assert aggregate_agreement(annotations, {"other": _ONE_TO_FIVE}) == (0.0, 2)


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
    assert aggregate_agreement(annotations) == (0.0, 1)
    assert aggregate_agreement_for_annotator(annotations, "a") == (0.0, 1)
    # Observed range over the evaluator's 5 and the humans' 3 and 4 is 2.
    assert aggregate_human_evaluator_agreement(annotations, runs, "e1") == (0.25, 2)
    assert evaluator_human_pair_agreement(5, [3, 4]) == (0.5, 2)
    item = per_item_agreement(annotations, runs, ["e1"])
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
    scales = {"e1": _ONE_TO_FIVE}
    assert aggregate_agreement(annotations, scales) == (0.75, 1)
    assert aggregate_agreement_for_annotator(annotations, "a", scales) == (0.75, 1)
    # Evaluator said 5: 1 - 2/4 against the 3, 1 - 1/4 against the 4.
    assert aggregate_human_evaluator_agreement(annotations, runs, "e1", scales) == (
        0.625,
        2,
    )
    assert evaluator_human_pair_agreement(5, [3, 4], _ONE_TO_FIVE) == (1.25, 2)
    item = per_item_agreement(annotations, runs, ["e1"], scales)
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
    scales = {"e1": _ONE_TO_FIVE}
    assert trend_series(annotations, "week", 7, now=now)[-1]["agreement"] == 0.0
    assert (
        trend_series(annotations, "week", 7, now=now, scales=scales)[-1]["agreement"]
        == 0.75
    )
    assert (
        trend_series_for_annotator(
            annotations, "a", "week", 7, now=now, scales=scales
        )[-1]["agreement"]
        == 0.75
    )
    assert (
        trend_series_human_evaluator(
            annotations, runs, ["e1"], "week", 7, now=now, scales=scales
        )["e1"][-1]["agreement"]
        == 0.625
    )
    breakdown = trend_series_evaluator_breakdown(
        annotations,
        runs,
        "e1",
        version_ids=["v1"],
        task_ids=["t1"],
        bucket="week",
        days=7,
        now=now,
        scales=scales,
    )
    assert breakdown["overall"][-1]["agreement"] == 0.625
    assert breakdown["by_version"]["v1"][-1]["agreement"] == 0.625
    assert breakdown["by_task"]["t1"][-1]["agreement"] == 0.625


def test_scales_for_reads_the_live_version_and_each_judge_version():
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
    bounds = scales_for(annotations, runs)
    # Human labels record no version, so they get the live rubric.
    assert bounds[rating] == (1, 5)
    # A judge score gets the rubric of the version that gave it.
    assert bounds[stale["uuid"]] == (1, 3)
    # A binary evaluator has no live version here, and an unknown id is absent.
    assert binary not in bounds and "missing-evaluator" not in bounds
    assert scales_for([]) == {}


def test_judge_score_is_divided_by_its_own_versions_width():
    # The judge scored on an older 1-to-4 version; the live version rates 1 to 2.
    scales = {"e1": (1, 2), "v-old": (1, 4)}
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
    assert aggregate_human_evaluator_agreement(annotations, runs, "e1", scales) == (
        round(1 - 1 / 3, 4),
        1,
    )


def test_agreement_never_goes_below_zero():
    # Labels 3 apart, given before the rubric was narrowed to 1 to 2.
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": a, "value": {"value": v}}
        for a, v in (("a", 1), ("b", 4))
    ]
    assert aggregate_agreement(annotations, {"e1": (1, 2)}) == (0.0, 1)
