"""Unit tests for pure helpers in llm_judge."""

from llm_judge import (
    build_evaluator_cli_payload,
    build_evaluator_cli_payload_unrendered,
    build_test_evaluators_payload,
    render_template,
)


def test_render_template_substitutes_variable():
    assert render_template("Hello {{name}}", {"name": "World"}) == "Hello World"


def test_render_template_missing_variable_is_empty():
    assert render_template("Hello {{name}}", {}) == "Hello "


def test_build_evaluator_cli_payload_minimal_binary():
    payload = build_evaluator_cli_payload(
        [
            {
                "name": "test-ev",
                "system_prompt": "Judge this.",
                "judge_model": "openai/gpt-4",
                "output_type": "binary",
            }
        ]
    )
    assert len(payload) == 1
    assert payload[0]["name"] == "test-ev"
    assert payload[0]["type"] == "binary"
    assert payload[0]["judge_model"] == "openai/gpt-4"
    assert payload[0]["system_prompt"] == "Judge this."


def test_only_tool_call_evaluators_are_flagged_for_calibrate():
    from llm_judge import build_evaluator_cli_payload_unrendered, build_test_evaluators_payload

    evs = [
        {"uuid": "t", "name": "tc", "evaluator_type": "tool-call", "system_prompt": "p"},
        {"uuid": "l", "name": "reply", "evaluator_type": "llm", "system_prompt": "p"},
    ]
    top_level, _ = build_test_evaluators_payload([{"test_uuid": "x", "evaluators": evs}])
    for payload in (
        build_evaluator_cli_payload(evs),
        build_evaluator_cli_payload_unrendered(evs),
        top_level,
    ):
        assert [ev.get("tool_call") for ev in payload] == [True, None]


def test_unrendered_payloads_tidy_spaced_placeholders_for_calibrate():
    ev = {"uuid": "e1", "name": "Judge", "system_prompt": "Use {{ criteria }} and {{criteria}}."}
    tidy = "Use {{criteria}} and {{criteria}}."
    top_level, criteria = build_test_evaluators_payload(
        [{"test_uuid": "t1", "evaluators": [{**ev, "variable_values": {"criteria": "x"}}]}]
    )
    assert top_level[0]["system_prompt"] == tidy
    assert criteria["t1"] == [{"name": "Judge", "arguments": {"criteria": "x"}}]
    assert build_evaluator_cli_payload_unrendered([ev])[0]["system_prompt"] == tidy
