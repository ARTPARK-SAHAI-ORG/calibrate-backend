"""Additional coverage for src/llm_judge.py."""

from __future__ import annotations

from llm_judge import (
    _calibrate_evaluator_def,
    _format_scale_rubric,
    _render_with_rubric,
    _scale_bounds,
    build_evaluator_cli_payload,
    build_evaluator_cli_payload_unrendered,
    build_test_evaluators_payload,
    render_template,
)


def test_render_template_dict_and_list_values():
    out = render_template("data={{x}}", {"x": {"a": 1}})
    assert out == 'data={"a": 1}'
    out = render_template("items={{y}}", {"y": [1, 2, 3]})
    assert out == "items=[1, 2, 3]"
    # Explicit None substitutes empty string
    assert render_template("v={{n}}", {"n": None}) == "v="


def test_format_scale_rubric_empty_and_full():
    assert _format_scale_rubric(None) == ""
    assert _format_scale_rubric({}) == ""
    assert _format_scale_rubric({"scale": "not a list"}) == ""
    # Some entries have descriptions, others don't — only described ones surface
    rubric = _format_scale_rubric(
        {
            "scale": [
                {"value": 1, "name": "Bad", "description": "no good"},
                {"value": 2, "name": "OK"},  # no description
                {"value": 3, "description": "fine"},
            ]
        }
    )
    assert "Rubric:" in rubric
    assert "1 (Bad): no good" in rubric
    assert "2 (OK)" not in rubric  # skipped (no description)
    assert "3: fine" in rubric

    # Scale without any descriptions returns empty
    assert _format_scale_rubric({"scale": [{"value": 1, "name": "A"}]}) == ""


def test_scale_bounds():
    assert _scale_bounds(None) == (None, None)
    assert _scale_bounds({}) == (None, None)
    assert _scale_bounds({"scale": "not a list"}) == (None, None)
    assert _scale_bounds({"scale": [{"value": "non-numeric"}]}) == (None, None)
    assert _scale_bounds({"scale": [{"value": 1}, {"value": 5}, {"value": 3}]}) == (1, 5)


def test_calibrate_evaluator_def_rating_with_bounds_and_uuid():
    ev = {
        "name": "ev",
        "judge_model": "openai/gpt-4",
        "output_type": "rating",
        "output_config": {"scale": [{"value": 1}, {"value": 2}, {"value": 5}]},
        "uuid": "uuid-1",
    }
    out = _calibrate_evaluator_def(ev, rendered_prompt="prompt")
    assert out["type"] == "rating"
    assert out["scale_min"] == 1
    assert out["scale_max"] == 5
    assert out["id"] == "uuid-1"

    # binary: no scale bounds, no id when uuid missing
    bin_ev = {"name": "x", "judge_model": "m", "output_type": "binary"}
    out = _calibrate_evaluator_def(bin_ev, rendered_prompt="p")
    assert "scale_min" not in out and "scale_max" not in out
    assert "id" not in out
    assert "tool_call" not in out


def test_calibrate_evaluator_def_marks_tool_call_evaluator():
    ev = {"name": "t", "judge_model": "m", "output_type": "binary", "evaluator_type": "tool-call"}
    assert _calibrate_evaluator_def(ev, rendered_prompt="p")["tool_call"] is True
    llm_ev = {**ev, "evaluator_type": "llm"}
    assert "tool_call" not in _calibrate_evaluator_def(llm_ev, rendered_prompt="p")


def test_render_with_rubric_substitutes_variables_and_appends_rubric():
    ev = {
        "system_prompt": "Hello {{name}}",
        "variables": [{"name": "name", "default": "World"}],
        "variable_values": {},
        "output_config": {
            "scale": [{"value": 1, "name": "Bad", "description": "nope"}]
        },
    }
    rendered = _render_with_rubric(ev)
    assert "Hello World" in rendered
    assert "Rubric:" in rendered

    # Variable values + extra_vars combine
    ev["variable_values"] = {"name": "Specific"}
    rendered = _render_with_rubric(ev, extra_vars={"name": "Override"})
    assert "Hello Override" in rendered

    # Variables-spec entries missing a name are skipped
    ev2 = {
        "system_prompt": "Hi",
        "variables": [{"default": "x"}],  # no name
        "variable_values": {"k": "v"},
        "output_config": {},
    }
    out = _render_with_rubric(ev2)
    assert out == "Hi"


def test_build_evaluator_cli_payload_unrendered_preserves_placeholders():
    payload = build_evaluator_cli_payload_unrendered(
        [
            {
                "name": "ev",
                "judge_model": "m",
                "output_type": "binary",
                "system_prompt": "Judge: {{x}}",
                "output_config": {
                    "scale": [
                        {"value": True, "description": "ok"},
                        {"value": False, "description": "no"},
                    ]
                },
            }
        ]
    )
    assert "{{x}}" in payload[0]["system_prompt"]
    assert "Rubric:" in payload[0]["system_prompt"]


def test_build_evaluator_cli_payload_with_variables():
    payload = build_evaluator_cli_payload(
        [
            {
                "name": "ev",
                "judge_model": "m",
                "output_type": "binary",
                "system_prompt": "Judge: {{x}}",
                "variables": [{"name": "x", "default": "default-x"}],
            }
        ]
    )
    assert payload[0]["system_prompt"] == "Judge: default-x"


def test_build_test_evaluators_payload_deduping_and_arguments():
    tests = [
        {
            "test_uuid": "t1",
            "evaluators": [
                {
                    "uuid": "e1",
                    "name": "shared",
                    "judge_model": "m",
                    "output_type": "binary",
                    "system_prompt": "Judge: {{c}}",
                    "variable_values": {"c": "first"},
                },
            ],
        },
        {
            "test_uuid": "t2",
            "evaluators": [
                # Same UUID — dedup
                {
                    "uuid": "e1",
                    "name": "shared",
                    "judge_model": "m",
                    "output_type": "binary",
                    "system_prompt": "Judge: {{c}}",
                    "variable_values": {"c": "second"},
                },
                # Different UUID, same name — gets suffixed
                {
                    "uuid": "e2",
                    "name": "shared",
                    "judge_model": "m",
                    "output_type": "binary",
                    "system_prompt": "Judge: {{c}}",
                },
                # Missing uuid — skipped
                {
                    "name": "no-uuid",
                    "judge_model": "m",
                    "output_type": "binary",
                    "system_prompt": "p",
                },
            ],
        },
    ]
    top, per_test = build_test_evaluators_payload(tests)
    # Top-level deduped to two entries
    assert len(top) == 2
    names = {e["name"] for e in top}
    assert "shared" in names
    assert any(n.startswith("shared-") for n in names)
    # Criteria per test
    assert per_test["t1"][0]["arguments"] == {"c": "first"}
    assert per_test["t2"][0]["arguments"] == {"c": "second"}
