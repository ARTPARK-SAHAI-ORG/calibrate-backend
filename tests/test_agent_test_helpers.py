"""Unit tests for pure helpers in routers/agent_tests.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_read_agent_test_results_json_missing(tmp_path):
    from routers.agent_tests import _read_agent_test_results_json

    assert _read_agent_test_results_json(None) is None
    assert _read_agent_test_results_json(tmp_path / "missing") is None


def test_read_agent_test_results_json_found(tmp_path):
    from routers.agent_tests import _read_agent_test_results_json

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "results.json").write_text(json.dumps([{"x": 1}]))
    assert _read_agent_test_results_json(tmp_path) == [{"x": 1}]


def test_read_agent_test_results_json_malformed(tmp_path):
    from routers.agent_tests import _read_agent_test_results_json

    (tmp_path / "results.json").write_text("{not json")
    assert _read_agent_test_results_json(tmp_path) is None


def test_read_agent_test_metrics_json_missing(tmp_path):
    from routers.agent_tests import _read_agent_test_metrics_json

    assert _read_agent_test_metrics_json(None) is None
    assert _read_agent_test_metrics_json(tmp_path / "missing") is None


def test_read_agent_test_metrics_json_found(tmp_path):
    from routers.agent_tests import _read_agent_test_metrics_json

    (tmp_path / "metrics.json").write_text(json.dumps({"a": 1}))
    assert _read_agent_test_metrics_json(tmp_path) == {"a": 1}


def test_parse_agent_test_results():
    from routers.agent_tests import _parse_agent_test_results

    data = [
        {
            "test_case_id": "t1",
            "output": {"response": "hi", "tool_calls": None},
            "metrics": {"passed": True, "reasoning": "ok"},
            "test_case": {"name": "T1", "id": "t1"},
        }
    ]
    out = _parse_agent_test_results(data)
    assert out[0]["passed"] is True

    assert _parse_agent_test_results(None) == []
    assert _parse_agent_test_results("not-a-list") == []


def test_merge_test_results_by_test_names():
    from routers.agent_tests import _merge_test_results_by_test_names

    completed = [{"name": "t1", "passed": True}]
    merged = _merge_test_results_by_test_names(["t1", "t2"], completed)
    assert merged[0]["passed"] is True
    assert merged[1]["name"] == "t2"
    assert merged[1]["passed"] is None

    # No test_names
    assert _merge_test_results_by_test_names([], completed) == []


def test_benchmark_queued_model_results():
    from routers.agent_tests import _benchmark_queued_model_results

    out = _benchmark_queued_model_results(["m1", "m2"], ["t1"])
    assert len(out) == 2
    assert out[0]["model"] == "m1"
    assert out[0]["success"] is None


def test_enrich_test_results_with_evaluators_none():
    from routers.agent_tests import _enrich_test_results_with_evaluators

    # No-op for None / empty
    _enrich_test_results_with_evaluators(None, {})
    _enrich_test_results_with_evaluators([], {})


def test_enrich_test_results_with_evaluators_dict_judge():
    """judge_results is the raw dict shape calibrate emits."""
    from routers.agent_tests import _enrich_test_results_with_evaluators

    test_results = [
        {
            "test_case_id": "t1",
            "judge_results": {
                "Safety": {
                    "evaluator_id": "ev-1",
                    "reasoning": "ok",
                    "match": True,
                }
            },
        }
    ]
    snapshot = {
        "t1": [{"uuid": "ev-1", "name": "Safety", "variable_values": {"x": 1}}]
    }
    with patch(
        "db.get_evaluator",
        return_value={"uuid": "ev-1", "name": "Safety NEW", "description": "d"},
    ):
        _enrich_test_results_with_evaluators(test_results, snapshot)
    assert test_results[0]["judge_results"][0]["name"] == "Safety NEW"


def test_enrich_test_results_with_evaluators_list_judge():
    """Idempotent when judge_results is already a structured list."""
    from routers.agent_tests import _enrich_test_results_with_evaluators

    test_results = [
        {
            "test_case_id": "t1",
            "judge_results": [
                {"evaluator_uuid": "ev-1", "name": "Stale"},
            ],
        }
    ]
    with patch(
        "db.get_evaluator",
        return_value={"uuid": "ev-1", "name": "Refreshed", "description": "d"},
    ):
        _enrich_test_results_with_evaluators(test_results, None)
    assert test_results[0]["judge_results"][0]["name"] == "Refreshed"


def test_enrich_test_results_with_evaluators_value_name_binary():
    """Binary judge_results pick up `value_name` from the snapshot's rubric."""
    from routers.agent_tests import _enrich_test_results_with_evaluators

    test_results = [
        {
            "test_case_id": "t1",
            "judge_results": {
                "Safety": {
                    "evaluator_id": "ev-1",
                    "reasoning": "ok",
                    "match": True,
                }
            },
        }
    ]
    snapshot = {
        "t1": [
            {
                "uuid": "ev-1",
                "name": "Safety",
                "output_type": "binary",
                "output_config": {
                    "scale": [
                        {"value": True, "name": "Safe"},
                        {"value": False, "name": "Unsafe"},
                    ]
                },
            }
        ]
    }
    with patch(
        "db.get_evaluator",
        return_value={"uuid": "ev-1", "name": "Safety", "description": "d"},
    ):
        _enrich_test_results_with_evaluators(test_results, snapshot)
    entry = test_results[0]["judge_results"][0]
    assert entry["value_name"] == "Safe"


def test_enrich_test_results_with_evaluators_value_name_rating():
    """Rating judge_results resolve `value_name` via the numeric scale entry."""
    from routers.agent_tests import _enrich_test_results_with_evaluators

    test_results = [
        {
            "test_case_id": "t1",
            "judge_results": {
                "Helpfulness": {
                    "evaluator_id": "ev-2",
                    "reasoning": "great",
                    "score": 4,
                }
            },
        }
    ]
    snapshot = {
        "t1": [
            {
                "uuid": "ev-2",
                "name": "Helpfulness",
                "output_type": "rating",
                "scale_min": 1,
                "scale_max": 5,
                "output_config": {
                    "scale": [
                        {"value": 1, "name": "Terrible"},
                        {"value": 4, "name": "Good"},
                        {"value": 5, "name": "Excellent"},
                    ]
                },
            }
        ]
    }
    with patch(
        "db.get_evaluator",
        return_value={"uuid": "ev-2", "name": "Helpfulness", "description": "d"},
    ):
        _enrich_test_results_with_evaluators(test_results, snapshot)
    entry = test_results[0]["judge_results"][0]
    assert entry["value_name"] == "Good"


def test_enrich_test_results_with_evaluators_value_name_list_path():
    """List-shape judge_results (idempotent re-enrichment) also resolves
    `value_name` from the snapshot — matches the dict-path behavior so the
    field doesn't disappear on re-read."""
    from routers.agent_tests import _enrich_test_results_with_evaluators

    test_results = [
        {
            "test_case_id": "t1",
            "judge_results": [
                {"evaluator_uuid": "ev-1", "name": "Safety", "match": False},
            ],
        }
    ]
    snapshot = {
        "t1": [
            {
                "uuid": "ev-1",
                "name": "Safety",
                "output_type": "binary",
                "output_config": {
                    "scale": [
                        {"value": True, "name": "Safe"},
                        {"value": False, "name": "Unsafe"},
                    ]
                },
            }
        ]
    }
    with patch(
        "db.get_evaluator",
        return_value={"uuid": "ev-1", "name": "Safety", "description": "d"},
    ):
        _enrich_test_results_with_evaluators(test_results, snapshot)
    assert test_results[0]["judge_results"][0]["value_name"] == "Unsafe"


def test_enrich_test_results_with_evaluators_value_name_legacy_fallback():
    """Legacy snapshot without `output_config` falls back to Correct/Wrong
    for binary so old runs still surface a label."""
    from routers.agent_tests import _enrich_test_results_with_evaluators

    test_results = [
        {
            "test_case_id": "t1",
            "judge_results": {
                "Safety": {
                    "evaluator_id": "ev-1",
                    "match": True,
                }
            },
        }
    ]
    snapshot = {
        "t1": [{"uuid": "ev-1", "name": "Safety", "output_type": "binary"}]
    }
    with patch(
        "db.get_evaluator",
        return_value={"uuid": "ev-1", "name": "Safety", "description": "d"},
    ):
        _enrich_test_results_with_evaluators(test_results, snapshot)
    assert test_results[0]["judge_results"][0]["value_name"] == "Correct"


def test_enrich_model_results_with_evaluators():
    from routers.agent_tests import _enrich_model_results_with_evaluators

    _enrich_model_results_with_evaluators(None, {})
    _enrich_model_results_with_evaluators([], {})
    # Happy path: nested test_results
    mr = [
        {
            "test_results": [
                {
                    "test_case_id": "t1",
                    "judge_results": {
                        "Safety": {
                            "evaluator_id": "ev-1",
                            "match": True,
                        }
                    },
                }
            ]
        }
    ]
    with patch(
        "db.get_evaluator",
        return_value={"uuid": "ev-1", "name": "Safety", "description": "d"},
    ):
        _enrich_model_results_with_evaluators(mr, {})
    assert mr[0]["test_results"][0]["judge_results"][0]["evaluator_uuid"] == "ev-1"


def test_build_evaluator_summary():
    from routers.agent_tests import _build_evaluator_summary

    assert _build_evaluator_summary(None) is None
    assert _build_evaluator_summary({"criteria": "not-a-dict"}) is None

    out = _build_evaluator_summary(
        {
            "criteria": {
                "Safety": {
                    "type": "binary",
                    "passed": 4,
                    "total": 5,
                    "evaluator_id": "ev-1",
                },
                "Quality": {
                    "type": "rating",
                    "mean": 3.5,
                    "evaluator_id": "ev-2",
                },
                "Skipped": {"type": "other"},
                "AlsoSkipped": "not-a-dict",
            }
        }
    )
    assert any(e["type"] == "binary" for e in out)
    assert any(e["type"] == "rating" for e in out)


def test_calibrate_config_from_agent_test_job_stored():
    """If stored calibrate_config is on the job, it's reused."""
    from routers.agent_tests import _calibrate_config_from_agent_test_job

    with patch(
        "routers.agent_tests.get_agent_test_job",
        return_value={"details": {"calibrate_config": {"a": 1}}},
    ):
        out = _calibrate_config_from_agent_test_job("j", None, None)
    assert out == {"a": 1}


def test_pending_test_case_result_placeholder():
    from routers.agent_tests import _pending_test_case_result_placeholder

    out = _pending_test_case_result_placeholder("t1")
    assert out["name"] == "t1"
    assert out["passed"] is None


def test_get_evaluator_cached_for_enrichment():
    from routers.agent_tests import _get_evaluator_cached_for_enrichment

    cache = {}
    with patch("db.get_evaluator", return_value={"uuid": "e", "name": "n"}):
        ev = _get_evaluator_cached_for_enrichment("e", cache)
    assert ev["name"] == "n"
    # second call doesn't refetch
    ev2 = _get_evaluator_cached_for_enrichment("e", cache)
    assert ev2 is ev
