"""Unit tests for pure helpers in llm_judge."""

from llm_judge import build_evaluator_cli_payload, render_template


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
