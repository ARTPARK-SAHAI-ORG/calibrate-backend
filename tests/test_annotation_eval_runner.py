"""Unit tests for the annotation evaluator-run worker.

Covers the pure helpers (`_resolve_evaluator_dicts`, `build_dataset_for_task_type`,
`calibrate_command_for_task_type`, parsers, error extractor) and the
high-level dispatch (`start_annotation_eval_job`, `resume_annotation_eval_job`).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import annotation_eval_runner as runner


# ---------------------------------------------------------------------------
# trivial helpers
# ---------------------------------------------------------------------------


def test_utc_now_format():
    from utils import utc_now

    s = utc_now()
    assert len(s) == len("2024-01-01 00:00:00")
    assert s[4] == "-" and s[7] == "-" and s[10] == " "


def test_output_dir_snapshot_missing_and_present(tmp_path):
    assert runner._output_dir_snapshot(tmp_path / "missing") == (0, 0)
    (tmp_path / "a.txt").write_text("hi")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("world")
    count, size = runner._output_dir_snapshot(tmp_path)
    assert count == 2 and size > 0


# ---------------------------------------------------------------------------
# _resolve_evaluator_dicts
# ---------------------------------------------------------------------------


def _ev_record(**overrides):
    base = {
        "uuid": "ev-1",
        "name": "Safety",
        "live_version_id": "ver-1",
        "output_type": "binary",
        "kind": "single",
        "data_type": "text",
    }
    base.update(overrides)
    return base


def _version_record(**overrides):
    base = {
        "uuid": "ver-1",
        "evaluator_id": "ev-1",
        "judge_model": "gpt",
        "system_prompt": "p",
        "output_config": {},
        "variables": [],
    }
    base.update(overrides)
    return base


def test_resolve_evaluator_dicts_missing_id():
    with pytest.raises(runner.EvaluatorResolutionError):
        runner._resolve_evaluator_dicts([{"evaluator_id": None}], {"ev-1"}, "llm")


def test_resolve_evaluator_dicts_not_linked():
    with pytest.raises(runner.EvaluatorResolutionError):
        runner._resolve_evaluator_dicts(
            [{"evaluator_id": "ev-1"}], set(), "llm"  # not linked
        )


def test_resolve_evaluator_dicts_evaluator_missing():
    with patch("annotation_eval_runner.get_evaluator", return_value=None):
        with pytest.raises(runner.EvaluatorResolutionError):
            runner._resolve_evaluator_dicts(
                [{"evaluator_id": "ev-1"}], {"ev-1"}, "llm"
            )


def test_resolve_evaluator_dicts_no_live_version():
    with patch(
        "annotation_eval_runner.get_evaluator",
        return_value=_ev_record(live_version_id=None),
    ):
        with pytest.raises(runner.EvaluatorResolutionError):
            runner._resolve_evaluator_dicts(
                [{"evaluator_id": "ev-1"}], {"ev-1"}, "llm"
            )


def test_resolve_evaluator_dicts_version_mismatch():
    with patch(
        "annotation_eval_runner.get_evaluator", return_value=_ev_record()
    ), patch(
        "annotation_eval_runner.get_evaluator_version",
        return_value=_version_record(evaluator_id="other"),
    ):
        with pytest.raises(runner.EvaluatorResolutionError):
            runner._resolve_evaluator_dicts(
                [{"evaluator_id": "ev-1"}], {"ev-1"}, "llm"
            )


def test_resolve_evaluator_dicts_missing_identity_field():
    with patch(
        "annotation_eval_runner.get_evaluator",
        return_value=_ev_record(output_type=None),
    ), patch(
        "annotation_eval_runner.get_evaluator_version", return_value=_version_record()
    ):
        with pytest.raises(runner.EvaluatorResolutionError):
            runner._resolve_evaluator_dicts(
                [{"evaluator_id": "ev-1"}], {"ev-1"}, "llm"
            )


def test_resolve_evaluator_dicts_happy_path():
    with patch(
        "annotation_eval_runner.get_evaluator", return_value=_ev_record()
    ), patch(
        "annotation_eval_runner.get_evaluator_version", return_value=_version_record()
    ):
        out = runner._resolve_evaluator_dicts(
            [{"evaluator_id": "ev-1"}], {"ev-1"}, "llm"
        )
    assert out[0]["uuid"] == "ev-1"
    assert out[0]["_evaluator_version_id"] == "ver-1"


def test_dedupe_evaluator_names():
    evs = [
        {"uuid": "aaaaaaaa11", "name": "Safety"},
        {"uuid": "bbbbbbbb22", "name": "Safety"},
        {"uuid": "cccccccc33", "name": "Other"},
    ]
    runner._dedupe_evaluator_names(evs)
    assert evs[0]["name"] == "Safety"
    assert evs[1]["name"].startswith("Safety-")
    assert evs[2]["name"] == "Other"


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def test_build_stt_dataset():
    items = [
        {
            "uuid": "i1",
            "payload": {
                "predicted_transcript": "pred",
                "reference_transcript": "ref",
            },
        }
    ]
    out = runner._build_stt_dataset(items)
    assert out == [{"id": "i1", "gt": "ref", "pred": "pred"}]


def test_build_stt_dataset_bad_payload():
    with pytest.raises(runner.DatasetBuildError):
        runner._build_stt_dataset([{"uuid": "i1", "payload": "not-a-dict"}])
    with pytest.raises(runner.DatasetBuildError):
        runner._build_stt_dataset([{"uuid": "i1", "payload": {}}])


def test_build_llm_dataset_no_evaluators():
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_dataset([], [])


def test_build_llm_dataset_bad_payload():
    evs = [{"uuid": "e1", "name": "judge"}]
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_dataset(
            [{"uuid": "i1", "payload": {"chat_history": "not-a-list"}}],
            evs,
        )
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_dataset(
            [
                {
                    "uuid": "i1",
                    "payload": {
                        "chat_history": [],
                        "agent_response": "r",
                        "tool_calls": "not-a-list",
                    },
                }
            ],
            evs,
        )
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_dataset(
            [
                {
                    "uuid": "i1",
                    "payload": {
                        "chat_history": [],
                        "agent_response": "r",
                        "evaluator_variables": "not-dict",
                    },
                }
            ],
            evs,
        )


def test_build_llm_dataset_happy():
    evs = [{"uuid": "e1", "name": "judge"}]
    out = runner._build_llm_dataset(
        [
            {
                "uuid": "i1",
                "payload": {
                    "chat_history": [{"role": "user", "content": "hi"}],
                    "agent_response": "hi back",
                    "evaluator_variables": {"e1": {"x": 1}},
                },
            }
        ],
        evs,
    )
    assert out[0]["test_case"]["id"] == "i1"
    assert out[0]["test_case"]["evaluation"]["criteria"][0]["arguments"] == {"x": 1}


def test_build_llm_dataset_judge_sees_only_default_text():
    evs = [{"uuid": "e1", "name": "judge"}]
    out = runner._build_llm_dataset(
        [
            {
                "uuid": "i1",
                "payload": {
                    "chat_history": [
                        {
                            "role": "user",
                            "content": "hi",
                            "label": "en",
                            "versions": {"en": "hi", "hi": "namaste"},
                        }
                    ],
                    "agent_response": {
                        "content": "hi back",
                        "label": "en",
                        "versions": {"en": "hi back", "hi": "namaste wapas"},
                    },
                },
            }
        ],
        evs,
    )
    assert out[0]["test_case"]["history"] == [{"role": "user", "content": "hi"}]
    assert out[0]["output"]["response"] == "hi back"


def test_build_llm_dataset_object_response_without_content():
    evs = [{"uuid": "e1", "name": "judge"}]
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_dataset(
            [
                {
                    "uuid": "i1",
                    "payload": {
                        "chat_history": [],
                        "agent_response": {"label": "en", "versions": {"en": "x"}},
                    },
                }
            ],
            evs,
        )


def test_is_tool_call_row():
    # Tool call, no text output → True (both field conventions).
    assert runner.is_tool_call_row(
        {"payload": {"tool_calls": [{"tool": "t", "arguments": {}}]}}
    )
    assert runner.is_tool_call_row(
        {"payload": {"actual_tool_calls": [{"tool": "t", "arguments": {}}]}}
    )
    # Text output present → False, even with tool calls alongside.
    assert not runner.is_tool_call_row(
        {"payload": {"agent_response": "hi", "tool_calls": [{"tool": "t"}]}}
    )
    assert not runner.is_tool_call_row({"payload": {"output": "o"}})
    # No tool calls, no text → False. Non-dict payload → False.
    assert not runner.is_tool_call_row({"payload": {"tool_calls": []}})
    assert not runner.is_tool_call_row({"payload": "nope"})
    # An item built from a tool-call test or trace carries the expected and
    # actual call lists. Either one present makes it a tool-call row whatever
    # the agent produced, including the agent answering in words and leaving
    # the actual list empty.
    assert runner.is_tool_call_row(
        {
            "payload": {
                "agent_response": "I cannot do that",
                "expected_tool_calls": [{"tool": "t", "arguments": {}}],
                "actual_tool_calls": [],
            }
        }
    )
    assert runner.is_tool_call_row({"payload": {"expected_tool_calls": []}})
    assert runner.is_tool_call_row(
        {"payload": {"output": "words", "actual_tool_calls": []}}
    )
    # The key must hold a list — a stray value of another type does not count.
    assert not runner.is_tool_call_row(
        {"payload": {"output": "o", "expected_tool_calls": "t"}}
    )
    # Same rule for the original tool_calls field: a truthy non-list value
    # (e.g. a raw string) must not count either.
    assert not runner.is_tool_call_row({"payload": {"tool_calls": "search(x)"}})


def test_build_llm_dataset_skips_tool_call_rows_without_a_tool_call_evaluator():
    evs = [{"uuid": "e1", "name": "judge"}]
    out = runner._build_llm_dataset(
        [
            {"uuid": "tc", "payload": {"tool_calls": [{"tool": "t", "arguments": {}}]}},
            {
                "uuid": "resp",
                "payload": {
                    "chat_history": [{"role": "user", "content": "hi"}],
                    "agent_response": "hi back",
                },
            },
        ],
        evs,
    )
    # No evaluator pairs with the tool-call row, so only the response row runs.
    assert [row["test_case"]["id"] for row in out] == ["resp"]


# ---------------------------------------------------------------------------
# tool-call evaluator
# ---------------------------------------------------------------------------

TC_ROW = {"uuid": "tc", "payload": {"tool_calls": [{"tool": "t", "arguments": {}}]}}
TEXT_ROW = {"uuid": "resp", "payload": {"agent_response": "hi"}}


def test_required_evaluator_ids_for_item_tool_call_row():
    assert runner.required_evaluator_ids_for_item(
        TC_ROW, ["ev-1", "ev-tc"], {"ev-tc"}
    ) == {"ev-tc"}
    # No tool-call evaluator on the job → nothing is required on the row.
    assert runner.required_evaluator_ids_for_item(TC_ROW, ["ev-1"], set()) == set()


def test_required_evaluator_ids_for_item_normal_row():
    assert runner.required_evaluator_ids_for_item(
        TEXT_ROW, ["ev-1", "ev-tc"], {"ev-tc"}
    ) == {"ev-1"}
    # Job carries only the tool-call evaluator → nothing is required here.
    assert (
        runner.required_evaluator_ids_for_item(TEXT_ROW, ["ev-tc"], {"ev-tc"}) == set()
    )


def _resolve_with_tool_call_evaluator(requested, tc_prompt="", task_type="llm", **tc):
    """Resolve `requested` where ev-tc is a tool-call evaluator and ev-1 is a
    normal LLM judge."""
    evaluators = {
        "ev-1": _ev_record(evaluator_type="llm"),
        "ev-tc": _ev_record(
            **{
                "uuid": "ev-tc",
                "name": "Tool call",
                "live_version_id": "ver-tc",
                "evaluator_type": runner.TOOL_CALL_EVALUATOR_TYPE,
                **tc,
            }
        ),
    }
    versions = {
        "ver-1": _version_record(),
        "ver-tc": _version_record(
            uuid="ver-tc", evaluator_id="ev-tc", system_prompt=tc_prompt
        ),
    }
    with patch(
        "annotation_eval_runner.get_evaluator", side_effect=lambda u: evaluators[u]
    ), patch(
        "annotation_eval_runner.get_evaluator_version",
        side_effect=lambda u: versions[u],
    ):
        return runner._resolve_evaluator_dicts(
            [{"evaluator_id": e} for e in requested], set(evaluators), task_type
        )


NOTHING_TO_RUN = (
    "None of the selected evaluators can run a judge on this task. A tool-call "
    "evaluator needs a prompt, and runs only on llm and llm-general tasks."
)


@pytest.mark.parametrize("prompt", ["", "   "])
def test_blank_prompt_tool_call_evaluator_never_reaches_calibrate_payload(prompt):
    resolved = _resolve_with_tool_call_evaluator(["ev-1", "ev-tc"], tc_prompt=prompt)
    assert [ev["uuid"] for ev in resolved] == ["ev-1"]
    payload = runner.build_evaluator_cli_payload_unrendered(resolved)
    assert [p["name"] for p in payload] == ["Safety"]


def test_resolve_rejects_only_blank_prompt_tool_call_evaluators():
    with pytest.raises(runner.EvaluatorResolutionError) as e:
        _resolve_with_tool_call_evaluator(["ev-tc"])
    assert str(e.value) == NOTHING_TO_RUN


def test_resolve_leaves_out_tool_call_evaluator_with_no_version():
    with pytest.raises(runner.EvaluatorResolutionError) as e:
        _resolve_with_tool_call_evaluator(["ev-tc"], live_version_id=None)
    assert str(e.value) == NOTHING_TO_RUN


@pytest.mark.parametrize("task_type", ["llm", "llm-general"])
def test_resolve_keeps_tool_call_evaluator_with_a_prompt(task_type):
    resolved = _resolve_with_tool_call_evaluator(
        ["ev-1", "ev-tc"], tc_prompt="Judge the calls", task_type=task_type
    )
    assert [ev["uuid"] for ev in resolved] == ["ev-1", "ev-tc"]
    assert [ev["evaluator_type"] for ev in resolved] == ["llm", "tool-call"]
    for build in (
        runner.build_evaluator_cli_payload,
        runner.build_evaluator_cli_payload_unrendered,
    ):
        assert all("evaluator_type" not in p for p in build(resolved))


@pytest.mark.parametrize("task_type", ["stt", "tts", "conversation"])
def test_resolve_leaves_out_tool_call_evaluator_on_other_task_types(task_type):
    resolved = _resolve_with_tool_call_evaluator(
        ["ev-1", "ev-tc"], tc_prompt="Judge the calls", task_type=task_type
    )
    assert [ev["uuid"] for ev in resolved] == ["ev-1"]
    with pytest.raises(runner.EvaluatorResolutionError) as e:
        _resolve_with_tool_call_evaluator(
            ["ev-tc"], tc_prompt="Judge the calls", task_type=task_type
        )
    assert str(e.value) == NOTHING_TO_RUN


REPLY_EV = {"uuid": "ev-1", "name": "Reply judge", "evaluator_type": "llm"}
TOOL_EV = {"uuid": "ev-tc", "name": "Tool judge", "evaluator_type": "tool-call"}
CALL_X = {"tool": "search", "arguments": {"q": "x"}}
CALL_Y = {"tool": "search", "arguments": {"q": "y"}}


def test_evaluators_for_item_pairs_each_kind_of_row():
    evs = [REPLY_EV, TOOL_EV]
    assert runner.evaluators_for_item(TC_ROW, evs) == [TOOL_EV]
    assert runner.evaluators_for_item(TEXT_ROW, evs) == [REPLY_EV]
    assert runner.evaluators_for_item(TC_ROW, [REPLY_EV]) == []
    assert runner.evaluators_for_item(TEXT_ROW, [TOOL_EV]) == []


def _llm_tool_row(**payload):
    row = runner._build_llm_dataset(
        [{"uuid": "tc", "payload": {"chat_history": [], **payload}}], [TOOL_EV]
    )
    assert len(row) == 1
    return row[0]


def test_build_llm_tool_call_row_with_expected_calls():
    assert runner._build_llm_dataset(
        [
            {
                "uuid": "tc",
                "payload": {
                    "chat_history": [{"role": "user", "content": "find x", "label": "en"}],
                    "agent_response": "",
                    "actual_tool_calls": [CALL_X],
                    "expected_tool_calls": [CALL_Y],
                },
            }
        ],
        [REPLY_EV, TOOL_EV],
    ) == [
        {
            "test_case": {
                "id": "tc",
                "history": [{"role": "user", "content": "find x"}],
                "evaluation": {"type": "response", "criteria": [{"name": "Tool judge"}]},
            },
            "output": {
                "response": (
                    'Tool calls made by the agent:\nsearch {"q": "x"}\n\n'
                    'Expected tool calls:\nsearch {"q": "y"}'
                ),
                "tool_calls": [],
            },
        }
    ]


def test_build_llm_tool_call_row_without_expected_calls():
    row = _llm_tool_row(agent_response="let me look", actual_tool_calls=[CALL_X])
    assert row["output"] == {
        "response": (
            "Reply from the agent:\nlet me look\n\n"
            'Tool calls made by the agent:\nsearch {"q": "x"}'
        ),
        "tool_calls": [],
    }


def test_build_llm_tool_call_row_with_tool_calls_only_and_no_reply():
    row = _llm_tool_row(tool_calls=[CALL_X])
    assert row["output"] == {
        "response": 'Tool calls made by the agent:\nsearch {"q": "x"}',
        "tool_calls": [],
    }


def test_build_llm_tool_call_row_where_the_agent_made_no_calls():
    row = _llm_tool_row(
        agent_response={"content": "I cannot"},
        actual_tool_calls=[],
        expected_tool_calls=[CALL_Y],
        tool_calls=[CALL_X],
    )
    assert row["output"]["response"] == (
        "Reply from the agent:\nI cannot\n\n"
        "The agent made no tool calls.\n\n"
        'Expected tool calls:\nsearch {"q": "y"}'
    )


def test_build_llm_tool_call_row_writes_earlier_calls_into_history():
    row = _llm_tool_row(
        chat_history=[
            {"role": "user", "content": "find x"},
            {"role": "assistant", "content": "", "tool_calls": [CALL_X]},
            {"role": "tool", "content": "found", "tool_call_id": "1"},
        ],
        actual_tool_calls=[CALL_Y],
    )
    assert row["test_case"]["history"] == [
        {"role": "user", "content": "find x"},
        {"role": "assistant", "content": '[Tool Call] search {"q": "x"}'},
        {"role": "tool", "content": "found"},
    ]


def test_build_llm_tool_call_row_needs_history():
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_dataset(
            [{"uuid": "tc", "payload": {"actual_tool_calls": [CALL_X]}}], [TOOL_EV]
        )


@pytest.mark.parametrize(
    "calls",
    [
        {"expected_tool_calls": [], "tool_calls": "search(x)"},
        {"actual_tool_calls": [], "expected_tool_calls": "search(x)"},
    ],
)
def test_build_llm_tool_call_row_rejects_calls_that_are_not_a_list(calls):
    with pytest.raises(runner.DatasetBuildError):
        _llm_tool_row(**calls)


def test_build_llm_reply_row_is_unchanged_beside_a_tool_call_evaluator():
    tool_calls = [{"tool": "t"}]
    assert runner._build_llm_dataset(
        [
            {
                "uuid": "r",
                "payload": {
                    "chat_history": [
                        {"role": "assistant", "content": "", "tool_calls": [CALL_X]}
                    ],
                    "agent_response": "hi",
                    "tool_calls": tool_calls,
                },
            }
        ],
        [REPLY_EV, TOOL_EV],
    ) == [
        {
            "test_case": {
                "id": "r",
                "history": [
                    {"role": "assistant", "content": "", "tool_calls": [CALL_X]}
                ],
                "evaluation": {"type": "response", "criteria": [{"name": "Reply judge"}]},
            },
            "output": {"response": "hi", "tool_calls": tool_calls},
        }
    ]


def test_build_llm_mixed_task_gives_each_row_only_its_evaluators():
    variables = {"ev-1": {"a": 1}, "ev-tc": {"b": 2}}
    out = runner._build_llm_dataset(
        [
            {
                "uuid": "r",
                "payload": {
                    "chat_history": [],
                    "agent_response": "hi",
                    "evaluator_variables": variables,
                },
            },
            {
                "uuid": "tc",
                "payload": {
                    "chat_history": [],
                    "actual_tool_calls": [CALL_X],
                    "evaluator_variables": variables,
                },
            },
        ],
        [REPLY_EV, TOOL_EV],
    )
    assert {r["test_case"]["id"]: r["test_case"]["evaluation"]["criteria"] for r in out} == {
        "r": [{"name": "Reply judge", "arguments": {"a": 1}}],
        "tc": [{"name": "Tool judge", "arguments": {"b": 2}}],
    }


def test_build_llm_general_tool_call_row_output_text():
    assert runner._build_llm_general_dataset(
        [
            {
                "uuid": "tc",
                "payload": {
                    "input": "find x",
                    "output": "let me look",
                    "actual_tool_calls": [CALL_X],
                    "expected_tool_calls": [CALL_Y],
                },
            }
        ],
        [TOOL_EV],
    ) == [
        {
            "id": "tc",
            "input": "find x",
            "output": (
                "Reply from the agent:\nlet me look\n\n"
                'Tool calls made by the agent:\nsearch {"q": "x"}\n\n'
                'Expected tool calls:\nsearch {"q": "y"}'
            ),
        }
    ]


def test_build_llm_general_tool_call_row_without_output_uses_tool_calls():
    out = runner._build_llm_general_dataset(
        [{"uuid": "tc", "payload": {"input": "i", "tool_calls": [CALL_X]}}], [TOOL_EV]
    )
    assert out[0]["output"] == 'Tool calls made by the agent:\nsearch {"q": "x"}'


def test_build_llm_general_tool_call_row_still_needs_input():
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_general_dataset(
            [{"uuid": "tc", "payload": {"actual_tool_calls": [CALL_X]}}], [TOOL_EV]
        )


def test_build_llm_general_mixed_task_names_only_the_rows_evaluators():
    variables = {"ev-1": {"a": 1}, "ev-tc": {"b": 2}}
    out = runner._build_llm_general_dataset(
        [
            {
                "uuid": "r",
                "payload": {"input": "i", "output": "o", "evaluator_variables": variables},
            },
            {
                "uuid": "tc",
                "payload": {
                    "input": "i",
                    "tool_calls": [CALL_X],
                    "evaluator_variables": variables,
                },
            },
        ],
        [REPLY_EV, TOOL_EV],
    )
    assert {r["id"]: r["arguments"] for r in out} == {
        "r": {"Reply judge": {"a": 1}},
        "tc": {"Tool judge": {"b": 2}},
    }


def test_build_llm_general_skips_rows_with_no_evaluator():
    assert runner._build_llm_general_dataset([TC_ROW], [REPLY_EV]) == []


def test_parse_llm_general_drops_scores_that_do_not_pair_with_their_row(tmp_path):
    # calibrate general runs every evaluator on every row.
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"ev-1": "Reply", "ev-tc": "Tool"}})
    )
    (tmp_path / "results.csv").write_text(
        "id,input,output,Reply,Reply_reasoning,Tool,Tool_reasoning\n"
        "r,i,o,1,ok,0,no\n"
        "tc,i,o,0,no,1,ok\n"
    )
    evs = [
        {**_ev_resolved("ev-1", "Reply"), "evaluator_type": "llm"},
        {**_ev_resolved("ev-tc", "Tool"), "evaluator_type": "tool-call"},
    ]
    runs = runner.parse_results_for_task_type(
        "llm-general",
        tmp_path,
        evs,
        "job-1",
        items=[
            {"uuid": "r", "payload": {"input": "i", "output": "o"}},
            {"uuid": "tc", "payload": {"input": "i", "actual_tool_calls": [CALL_X]}},
        ],
    )
    assert sorted((r["item_id"], r["evaluator_id"], r["value"]["value"]) for r in runs) == [
        ("r", "ev-1", True),
        ("tc", "ev-tc", True),
    ]


def test_build_llm_general_dataset_no_evaluators():
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_general_dataset([], [])


def test_build_llm_general_dataset_bad_payload():
    evs = [{"uuid": "e1", "name": "judge"}]
    # Missing `input`.
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_general_dataset(
            [{"uuid": "i1", "payload": {"output": "o"}}], evs
        )
    # Missing `output`.
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_general_dataset(
            [{"uuid": "i1", "payload": {"input": "i"}}], evs
        )
    # Bad evaluator_variables.
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_general_dataset(
            [
                {
                    "uuid": "i1",
                    "payload": {
                        "input": "i",
                        "output": "o",
                        "evaluator_variables": "not-dict",
                    },
                }
            ],
            evs,
        )


def test_build_llm_general_dataset_happy():
    evs = [{"uuid": "e1", "name": "judge"}]
    out = runner._build_llm_general_dataset(
        [
            {
                "uuid": "i1",
                "payload": {"input": "summarize this", "output": "a summary"},
            }
        ],
        evs,
    )
    # Flat `calibrate general` shape: {id, input, output}. No vars → no arguments.
    assert out == [
        {"id": "i1", "input": "summarize this", "output": "a summary"}
    ]


def test_build_llm_general_dataset_judge_sees_only_default_text():
    evs = [{"uuid": "e1", "name": "judge"}]
    out = runner._build_llm_general_dataset(
        [
            {
                "uuid": "i1",
                "payload": {
                    "input": {
                        "content": "summarize this",
                        "label": "en",
                        "versions": {"en": "summarize this", "hi": "saar"},
                    },
                    "output": {"content": "a summary", "label": "en"},
                },
            }
        ],
        evs,
    )
    assert out == [{"id": "i1", "input": "summarize this", "output": "a summary"}]


def test_build_llm_general_dataset_object_without_content():
    evs = [{"uuid": "e1", "name": "judge"}]
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_general_dataset(
            [{"uuid": "i1", "payload": {"input": {"label": "en"}, "output": "o"}}],
            evs,
        )
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_general_dataset(
            [{"uuid": "i1", "payload": {"input": "i", "output": {"label": "en"}}}],
            evs,
        )


def test_build_llm_general_dataset_with_arguments():
    """Per-item `evaluator_variables` (same contract as the llm task type) are
    keyed by evaluator NAME in the per-row `arguments` map — each evaluator gets
    its own box, exactly like the llm path (no shared bag, no collision)."""
    evs = [{"uuid": "e1", "name": "judge1"}, {"uuid": "e2", "name": "judge2"}]
    out = runner._build_llm_general_dataset(
        [
            {
                "uuid": "i1",
                "payload": {
                    "input": "in",
                    "output": "out",
                    "evaluator_variables": {
                        # same {{var}} name for both — must NOT collide
                        "e1": {"criteria": "be concise"},
                        "e2": {"criteria": "be accurate"},
                        # not in this run → ignored
                        "e9": {"ignored": "x"},
                    },
                },
            }
        ],
        evs,
    )
    assert out[0]["id"] == "i1"
    assert out[0]["arguments"] == {
        "judge1": {"criteria": "be concise"},
        "judge2": {"criteria": "be accurate"},
    }


def test_build_dataset_dispatch_llm_general():
    out = runner.build_dataset_for_task_type(
        "llm-general",
        [{"uuid": "i1", "payload": {"input": "i", "output": "o"}}],
        [{"uuid": "e1", "name": "judge"}],
    )
    assert out == [{"id": "i1", "input": "i", "output": "o"}]


def test_build_tts_dataset():
    # Shape-only validation: no storage probe at submit time.
    items = [
        {
            "uuid": "i1",
            "payload": {
                "text": "hello",
                "audio_path": "s3://bucket/key.wav",
            },
        }
    ]
    out = runner._build_tts_dataset(items)
    assert out == [
        {"id": "i1", "text": "hello", "audio_path": "s3://bucket/key.wav"}
    ]


def test_resolve_s3_bucket_and_key_wraps_external_url():
    # An external URL isn't a storage location — the resolver's ValueError is
    # surfaced as a DatasetBuildError so the job fails cleanly.
    with pytest.raises(runner.DatasetBuildError):
        runner._resolve_s3_bucket_and_key("https://cdn.example/audio.mp3")


def test_build_tts_dataset_missing_fields():
    with pytest.raises(runner.DatasetBuildError):
        runner._build_tts_dataset([{"uuid": "i1", "payload": {"name": "x"}}])
    with pytest.raises(runner.DatasetBuildError):
        runner._build_tts_dataset(
            [{"uuid": "i1", "payload": {"text": "hi", "audio_path": "https://x"}}]
        )


def test_prepare_tts_eval_run_dir_downloads_audio(tmp_path):
    items = [
        {
            "uuid": "i1",
            "payload": {
                "text": "hello",
                "audio_path": "tts/media/clip.wav",
            },
        }
    ]

    def _fake_download(_s3, bucket, key, local_path):
        Path(local_path).write_bytes(b"wav")

    with patch(
        "annotation_eval_runner.get_s3_client", return_value=MagicMock()
    ), patch(
        "annotation_eval_runner.get_s3_output_config", return_value="test-bucket"
    ), patch(
        "annotation_eval_runner.download_file_from_s3", side_effect=_fake_download
    ):
        run_dir = runner._prepare_tts_eval_run_dir(tmp_path, items)

    assert (run_dir / "results.csv").exists()
    assert (run_dir / "audios" / "i1.wav").exists()
    rows = list(csv.DictReader(open(run_dir / "results.csv", encoding="utf-8")))
    assert rows[0]["id"] == "i1"
    assert rows[0]["text"] == "hello"
    assert rows[0]["audio_path"].endswith("i1.wav")


def test_prepare_tts_eval_run_dir_missing_clip_fails_with_item_and_key(tmp_path):
    # The download IS the existence check: a missing clip raises here (in the
    # background worker), naming the item + key.
    items = [
        {"uuid": "i1", "payload": {"text": "hi", "audio_path": "tts/media/missing.wav"}}
    ]

    def _fake_download(_s3, _bucket, _key, _local_path):
        raise FileNotFoundError("no such object")

    with patch(
        "annotation_eval_runner.get_s3_client", return_value=MagicMock()
    ), patch(
        "annotation_eval_runner.get_s3_output_config", return_value="test-bucket"
    ), patch(
        "annotation_eval_runner.download_file_from_s3", side_effect=_fake_download
    ):
        with pytest.raises(runner.DatasetBuildError, match="tts/media/missing.wav"):
            runner._prepare_tts_eval_run_dir(tmp_path, items)


def test_build_simulation_dataset():
    out = runner._build_simulation_dataset(
        [{"uuid": "i1", "payload": {"transcript": [{"role": "user", "content": "x"}]}}]
    )
    assert out[0]["name"] == "i1"


def test_build_simulation_dataset_judge_sees_only_default_text():
    out = runner._build_simulation_dataset(
        [
            {
                "uuid": "i1",
                "payload": {
                    "transcript": [
                        {
                            "role": "user",
                            "content": "x",
                            "label": "en",
                            "versions": {"en": "x", "hi": "y"},
                        },
                        {"role": "assistant", "content": "z", "label": "en"},
                    ]
                },
            }
        ]
    )
    assert out[0]["conversation_history"] == [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "z"},
    ]


def test_build_simulation_dataset_bad_payload():
    with pytest.raises(runner.DatasetBuildError):
        runner._build_simulation_dataset(
            [{"uuid": "i1", "payload": {"transcript": []}}]
        )
    with pytest.raises(runner.DatasetBuildError):
        runner._build_simulation_dataset(
            [{"uuid": "i1", "payload": {}}]
        )


def test_build_dataset_dispatch_unknown_task_type():
    with pytest.raises(runner.DatasetBuildError):
        runner.build_dataset_for_task_type("unknown", [], [])


def test_build_dataset_dispatch_conversation():
    # The `conversation` task type routes to the simulation-shaped dataset builder.
    out = runner.build_dataset_for_task_type(
        "conversation",
        [{"uuid": "i1", "payload": {"transcript": [{"role": "user", "content": "x"}]}}],
        [],
    )
    assert out[0]["name"] == "i1"
    assert out[0]["conversation_history"] == [{"role": "user", "content": "x"}]


# ---------------------------------------------------------------------------
# calibrate_command_for_task_type
# ---------------------------------------------------------------------------


def test_calibrate_command_for_task_type():
    p = Path("/tmp")
    out_stt = runner.calibrate_command_for_task_type("stt", p, p, p)
    assert out_stt[:3] == ["calibrate-agent", "stt", "--eval-only"]
    out_llm = runner.calibrate_command_for_task_type("llm", p, p, p)
    assert out_llm[:2] == ["calibrate-agent", "llm"]
    # llm-general uses the dedicated `calibrate-agent general` command (no --eval-only).
    out_llm_general = runner.calibrate_command_for_task_type("llm-general", p, p, p)
    assert out_llm_general[:2] == ["calibrate-agent", "general"]
    assert "--eval-only" not in out_llm_general
    out_sim = runner.calibrate_command_for_task_type("conversation", p, p, p)
    assert out_sim[:2] == ["calibrate-agent", "simulations"]
    out_tts = runner.calibrate_command_for_task_type("tts", p, p, p)
    assert out_tts[:3] == ["calibrate-agent", "tts", "--eval-only"]
    with pytest.raises(runner.DatasetBuildError):
        runner.calibrate_command_for_task_type("unknown", p, p, p)


# ---------------------------------------------------------------------------
# Output reading
# ---------------------------------------------------------------------------


def test_read_results_csv(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text("id,gt,pred\n1,hi,hi\n2,bye,bye\n")
    rows = runner._read_results_csv(tmp_path)
    assert len(rows) == 2

    # Missing file
    assert runner._read_results_csv(tmp_path / "missing") is None

    # Malformed (force open() to raise via mock)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "results.csv").write_text("col1,col2\n")
    with patch("builtins.open", side_effect=RuntimeError("boom")):
        assert runner._read_results_csv(bad) is None


def test_read_metrics_json(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"a": 1}))
    assert runner._read_metrics_json(tmp_path) == {"a": 1}
    assert runner._read_metrics_json(tmp_path / "missing") is None
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "metrics.json").write_text("{not json")
    assert runner._read_metrics_json(bad) is None


def test_read_config_evaluators_map(tmp_path):
    assert runner._read_config_evaluators_map(tmp_path) == {}
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"u1": "Safety", "u2": "Faithfulness"}})
    )
    # Inverts to name → uuid
    assert runner._read_config_evaluators_map(tmp_path) == {
        "Safety": "u1",
        "Faithfulness": "u2",
    }
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "config.json").write_text("not json")
    assert runner._read_config_evaluators_map(bad) == {}


def test_read_simulation_dataset_map(tmp_path):
    assert runner._read_simulation_dataset_map(tmp_path) == {}
    (tmp_path / "dataset_map.json").write_text(
        json.dumps({"row_1": {"index": 0}, "row_2": {"index": 1}, "row_bad": "x"})
    )
    out = runner._read_simulation_dataset_map(tmp_path)
    assert out == {"row_1": 0, "row_2": 1}
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "dataset_map.json").write_text("not json")
    assert runner._read_simulation_dataset_map(bad) == {}


# ---------------------------------------------------------------------------
# Results parsers
# ---------------------------------------------------------------------------


def _ev_resolved(uuid="ev-1", name="Safety", output_type="binary", version="ver-1"):
    return {
        "uuid": uuid,
        "name": name,
        "output_type": output_type,
        "_evaluator_version_id": version,
    }


def test_parse_results_stt(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"ev-1": "Safety"}})
    )
    (tmp_path / "results.csv").write_text(
        "id,gt,pred,Safety,Safety_reasoning\n"
        "i1,a,a,1,ok\n"
        "i2,b,b,0,not ok\n"
    )
    runs = runner._parse_results_stt(tmp_path, [_ev_resolved()], "job-1")
    assert len(runs) == 2
    assert runs[0]["evaluator_id"] == "ev-1"
    assert runs[0]["value"]["value"] is True


def test_parse_results_stt_no_config_fallback(tmp_path):
    """Without config.json/evaluators_map, falls back to display name."""
    (tmp_path / "results.csv").write_text(
        "id,gt,pred,Safety,Safety_reasoning\ni1,a,a,1,ok\n"
    )
    runs = runner._parse_results_stt(tmp_path, [_ev_resolved()], "job-1")
    assert runs and runs[0]["value"]["value"] is True


def test_parse_results_stt_empty_value(tmp_path):
    """Empty score → run marked failed."""
    (tmp_path / "results.csv").write_text(
        "id,gt,pred,Safety,Safety_reasoning\ni1,a,a,,\n"
    )
    runs = runner._parse_results_stt(tmp_path, [_ev_resolved()], "job-1")
    assert runs[0]["status"] == "failed"


def test_parse_results_stt_no_id_row(tmp_path):
    (tmp_path / "results.csv").write_text(
        "id,gt,pred,Safety,Safety_reasoning\n,a,a,1,ok\n"
    )
    runs = runner._parse_results_stt(tmp_path, [_ev_resolved()], "job-1")
    assert runs == []


def test_parse_results_stt_unknown_uuid_in_map(tmp_path):
    """evaluators_map references a UUID we don't have a snapshot for → skip."""
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"unknown": "Other"}})
    )
    (tmp_path / "results.csv").write_text(
        "id,gt,pred,Other\ni1,a,a,1\n"
    )
    runs = runner._parse_results_stt(tmp_path, [_ev_resolved()], "job-1")
    assert runs == []


def test_parse_results_llm(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"ev-1": "Safety"}})
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "test_case": {"id": "i1"},
                    "metrics": {
                        "judge_results": {
                            "Safety": {"match": True, "reasoning": "ok"}
                        }
                    },
                }
            ]
        )
    )
    runs = runner._parse_results_llm(tmp_path, [_ev_resolved()], "job-1")
    assert len(runs) == 1
    assert runs[0]["value"]["value"] is True


def test_parse_results_llm_score_field(tmp_path):
    """Rating evaluator uses `score` not `match`."""
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "test_case": {"id": "i1"},
                    "metrics": {
                        "judge_results": {
                            "Safety": {"score": 3.5}
                        }
                    },
                }
            ]
        )
    )
    runs = runner._parse_results_llm(
        tmp_path, [_ev_resolved(output_type="rating")], "job-1"
    )
    assert runs[0]["value"]["value"] == 3.5


def test_parse_results_llm_missing_file(tmp_path):
    assert runner._parse_results_llm(tmp_path, [_ev_resolved()], "job-1") == []


def test_parse_results_llm_malformed(tmp_path):
    (tmp_path / "results.json").write_text("not json")
    assert runner._parse_results_llm(tmp_path, [_ev_resolved()], "job-1") == []


def test_parse_results_llm_missing_id_or_judge_results(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                "not-dict",
                {"metrics": {}},  # missing id
                {"test_case": {"id": "i1"}},  # missing metrics.judge_results
            ]
        )
    )
    assert runner._parse_results_llm(tmp_path, [_ev_resolved()], "job-1") == []


def test_parse_results_simulation(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"ev-1": "Safety"}})
    )
    (tmp_path / "dataset_map.json").write_text(
        json.dumps({"row_1": {"index": 0, "name": "fallback-uuid"}})
    )
    sim_dir = tmp_path / "row_1"
    sim_dir.mkdir()
    (sim_dir / "evaluation_results.csv").write_text(
        "name,type,value,reasoning,evaluator_id\nSafety,binary,1,ok,ev-1\n"
    )
    items = [{"uuid": "real-uuid"}]
    runs = runner._parse_results_simulation(
        tmp_path, [_ev_resolved()], "job-1", items=items
    )
    assert runs[0]["item_id"] == "real-uuid"


def test_parse_results_simulation_no_output_dir(tmp_path):
    """Missing output_dir → empty list."""
    runs = runner._parse_results_simulation(
        tmp_path / "missing", [_ev_resolved()], "job-1"
    )
    assert runs == []


def test_parse_results_simulation_fallback_name(tmp_path):
    """When items not supplied, fall back to dataset_map's name field."""
    (tmp_path / "dataset_map.json").write_text(
        json.dumps({"row_1": {"index": 0, "name": "from-name"}})
    )
    sim_dir = tmp_path / "row_1"
    sim_dir.mkdir()
    (sim_dir / "evaluation_results.csv").write_text(
        "name,type,value,reasoning,evaluator_id\nSafety,binary,1,ok,ev-1\n"
    )
    runs = runner._parse_results_simulation(
        tmp_path, [_ev_resolved()], "job-1", items=None
    )
    assert runs[0]["item_id"] == "from-name"


def test_parse_results_simulation_unmapped_row(tmp_path):
    """A sim_dir with no dataset_map entry is skipped."""
    sim_dir = tmp_path / "row_99"
    sim_dir.mkdir()
    (sim_dir / "evaluation_results.csv").write_text("name,value\nSafety,1\n")
    runs = runner._parse_results_simulation(tmp_path, [_ev_resolved()], "job-1")
    assert runs == []


def test_parse_results_for_task_type_dispatch(tmp_path):
    # Unknown task type → []
    assert (
        runner.parse_results_for_task_type(
            "unknown", tmp_path, [_ev_resolved()], "job-1"
        )
        == []
    )


def test_parse_results_general(tmp_path):
    """`calibrate general` writes the same results.csv shape as STT, with
    input/output as the built-in columns; the column-map parser is reused."""
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"ev-1": "Safety"}})
    )
    (tmp_path / "results.csv").write_text(
        "id,input,output,Safety,Safety_reasoning\n"
        "i1,in1,out1,1,ok\n"
        "i2,in2,out2,0,not ok\n"
    )
    runs = runner._parse_results_general(tmp_path, [_ev_resolved()], "job-1")
    assert len(runs) == 2
    assert runs[0]["evaluator_id"] == "ev-1"
    assert runs[0]["value"]["value"] is True
    assert runs[1]["value"]["value"] is False


def test_parse_results_for_task_type_dispatch_llm_general(tmp_path):
    # llm-general routes to the `calibrate general` CSV parser.
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"ev-1": "Safety"}})
    )
    (tmp_path / "results.csv").write_text(
        "id,input,output,Safety,Safety_reasoning\ni1,in1,out1,1,ok\n"
    )
    runs = runner.parse_results_for_task_type(
        "llm-general",
        tmp_path,
        [_ev_resolved()],
        "job-1",
        items=[{"uuid": "i1", "payload": {"input": "in1", "output": "out1"}}],
    )
    assert len(runs) == 1
    assert runs[0]["value"]["value"] is True


def test_parse_results_for_task_type_dispatch_conversation(tmp_path):
    # The `conversation` task type routes to the simulation results parser.
    (tmp_path / "dataset_map.json").write_text(
        json.dumps({"row_1": {"index": 0, "name": "from-name"}})
    )
    sim_dir = tmp_path / "row_1"
    sim_dir.mkdir()
    (sim_dir / "evaluation_results.csv").write_text(
        "name,type,value,reasoning,evaluator_id\nSafety,binary,1,ok,ev-1\n"
    )
    runs = runner.parse_results_for_task_type(
        "conversation", tmp_path, [_ev_resolved()], "job-1", items=None
    )
    assert runs[0]["item_id"] == "from-name"


# ---------------------------------------------------------------------------
# _persist_pgid
# ---------------------------------------------------------------------------


def test_persist_pgid_success():
    with patch("annotation_eval_runner.os.getpgid", return_value=999), patch(
        "annotation_eval_runner.update_job"
    ) as uj:
        runner._persist_pgid("j-1", 1234)
        uj.assert_called_once()


def test_persist_pgid_fallback_to_pid():
    with patch(
        "annotation_eval_runner.os.getpgid", side_effect=ProcessLookupError()
    ), patch("annotation_eval_runner.update_job") as uj:
        runner._persist_pgid("j-1", 1234)
        # Falls back to pid as pgid
        assert uj.call_args.kwargs["details"]["pgid"] == 1234


# ---------------------------------------------------------------------------
# _try_upload_partial_outputs
# ---------------------------------------------------------------------------


def test_try_upload_partial_outputs_missing(tmp_path):
    assert runner._try_upload_partial_outputs(None, "t", "j") is None
    assert (
        runner._try_upload_partial_outputs(tmp_path / "missing", "t", "j") is None
    )


def test_try_upload_partial_outputs_success(tmp_path):
    (tmp_path / "a.log").write_text("x")
    with patch(
        "annotation_eval_runner.get_s3_output_config", return_value="bucket"
    ), patch("annotation_eval_runner.get_s3_client", return_value=MagicMock()), patch(
        "annotation_eval_runner.upload_file_to_s3"
    ):
        s3_prefix = runner._try_upload_partial_outputs(tmp_path, "t", "j")
    assert s3_prefix and "annotation-tasks/t/evaluator-runs/j/outputs" in s3_prefix


def test_try_upload_partial_outputs_swallows_exception(tmp_path):
    with patch(
        "annotation_eval_runner.get_s3_output_config",
        side_effect=RuntimeError("nope"),
    ):
        assert runner._try_upload_partial_outputs(tmp_path, "t", "j") is None


# ---------------------------------------------------------------------------
# start_annotation_eval_job + queue starter
# ---------------------------------------------------------------------------


def test_start_annotation_eval_job_spawns_thread():
    with patch("annotation_eval_runner.threading.Thread") as thread_mock:
        runner.start_annotation_eval_job(
            "j-1", "t-1", "u-1", [_ev_resolved()], item_ids=None
        )
        thread_mock.return_value.start.assert_called_once()


def test_queue_starter_missing_fields():
    with pytest.raises(RuntimeError):
        runner._start_annotation_eval_job_from_queue(
            {"uuid": "j-1"}  # no details
        )


def test_queue_starter_no_evaluators():
    with pytest.raises(RuntimeError):
        runner._start_annotation_eval_job_from_queue(
            {"uuid": "j-1", "user_id": "u-1", "details": {"task_id": "t-1"}}
        )


def test_queue_starter_resolves_with_the_tasks_type():
    with patch(
        "db.get_evaluators_for_annotation_task", return_value=[{"uuid": "ev-1"}]
    ), patch(
        "annotation_eval_runner.get_annotation_task",
        return_value={"type": "llm-general"},
    ), patch(
        "annotation_eval_runner._resolve_evaluator_dicts",
        return_value=[_ev_resolved()],
    ) as resolve, patch(
        "annotation_eval_runner.start_annotation_eval_job"
    ):
        runner._start_annotation_eval_job_from_queue(
            {
                "uuid": "j-1",
                "user_id": "u-1",
                "details": {"task_id": "t-1", "evaluators": [{"evaluator_id": "ev-1"}]},
            }
        )
    assert resolve.call_args.args[2] == "llm-general"


def test_queue_starter_happy_path():
    with patch(
        "db.get_evaluators_for_annotation_task",
        return_value=[{"uuid": "ev-1"}],
    ), patch(
        "annotation_eval_runner._resolve_evaluator_dicts",
        return_value=[_ev_resolved()],
    ), patch(
        "annotation_eval_runner.start_annotation_eval_job"
    ) as sj:
        runner._start_annotation_eval_job_from_queue(
            {
                "uuid": "j-1",
                "user_id": "u-1",
                "details": {
                    "task_id": "t-1",
                    "evaluators": [
                        {"evaluator_id": "ev-1", "evaluator_version_id": "ver-1"}
                    ],
                    "item_ids": None,
                },
            }
        )
        sj.assert_called_once()


def test_resume_annotation_eval_job_clears_runs():
    with patch(
        "db.clear_evaluator_runs_for_job", return_value=3
    ), patch(
        "annotation_eval_runner._start_annotation_eval_job_from_queue"
    ) as sj:
        runner.resume_annotation_eval_job(
            {
                "uuid": "j-1",
                "user_id": "u-1",
                "details": {
                    "task_id": "t-1",
                    "evaluators": [
                        {"evaluator_id": "ev-1", "evaluator_version_id": "ver-1"}
                    ],
                },
            }
        )
        sj.assert_called_once()


# ---------------------------------------------------------------------------
# Streaming: results are stored as calibrate writes them
# ---------------------------------------------------------------------------


def test_read_results_csv_drops_truncated_tail_row(tmp_path):
    """A read that lands mid-append must not yield a half-written row."""
    (tmp_path / "results.csv").write_text(
        "id,gt,pred,Safety,Safety_reasoning\n"
        "i1,a,a,1,ok\n"
        "i2,b,b,0"  # no trailing newline: still being written
    )
    rows = runner._read_results_csv(tmp_path)
    assert [r["id"] for r in rows] == ["i1"]


def test_read_results_csv_keeps_blank_cells(tmp_path):
    """A complete row with empty cells is not mistaken for a truncated one."""
    (tmp_path / "results.csv").write_text(
        "id,gt,pred,Safety,Safety_reasoning\ni1,a,a,,\n"
    )
    rows = runner._read_results_csv(tmp_path)
    assert len(rows) == 1 and rows[0]["Safety"] == ""


def test_parse_results_simulation_drops_truncated_tail_row(tmp_path):
    (tmp_path / "dataset_map.json").write_text(
        json.dumps({"row_1": {"index": 0}})
    )
    d = tmp_path / "row_1"
    d.mkdir()
    (d / "evaluation_results.csv").write_text(
        "name,type,value,reasoning\nSafety,binary,1,ok\nSafety,binary,0"
    )
    runs = runner._parse_results_simulation(
        tmp_path, [_ev_resolved()], "job-1", items=[{"uuid": "i1"}]
    )
    assert len(runs) == 1 and runs[0]["value"]["value"] is True


def _run_row(item_id, evaluator_id="ev-1", status="completed"):
    return {
        "job_id": "job-1",
        "item_id": item_id,
        "evaluator_id": evaluator_id,
        "evaluator_version_id": "ver-1",
        "value": {"value": True} if status == "completed" else None,
        "status": status,
    }


def test_store_scored_runs_skips_already_stored():
    seen = set()
    with patch.object(runner, "create_evaluator_runs") as ins:
        assert runner._store_scored_runs([_run_row("i1"), _run_row("i2")], seen) == 2
        # Same parse again (next tick) stores nothing.
        assert runner._store_scored_runs([_run_row("i1"), _run_row("i2")], seen) == 0
        assert ins.call_count == 1
    assert seen == {("i1", "ev-1"), ("i2", "ev-1")}


def test_store_scored_runs_dedupes_within_one_batch():
    seen = set()
    with patch.object(runner, "create_evaluator_runs") as ins:
        assert runner._store_scored_runs([_run_row("i1"), _run_row("i1")], seen) == 1
        assert len(ins.call_args[0][0]) == 1


def test_store_scored_runs_holds_back_unscored_rows():
    """Mid-run an unscored cell means 'not judged yet', not 'judge failed'."""
    seen = set()
    with patch.object(runner, "create_evaluator_runs") as ins:
        assert (
            runner._store_scored_runs(
                [_run_row("i1"), _run_row("i2", status="failed")], seen
            )
            == 1
        )
        assert [r["item_id"] for r in ins.call_args[0][0]] == ["i1"]
    assert ("i2", "ev-1") not in seen


def test_store_scored_runs_retries_after_a_failed_write():
    """A write that blew up must not leave its rows marked as stored."""
    seen = set()
    with patch.object(runner, "create_evaluator_runs", side_effect=RuntimeError("db")):
        with pytest.raises(RuntimeError):
            runner._store_scored_runs([_run_row("i1")], seen)
    assert seen == set()
    with patch.object(runner, "create_evaluator_runs") as ins:
        assert runner._store_scored_runs([_run_row("i1")], seen) == 1
        assert ins.call_count == 1


def test_discard_partial_runs_swallows_db_errors():
    with patch.object(
        runner, "clear_evaluator_runs_for_job", side_effect=RuntimeError("db")
    ):
        runner._discard_partial_runs("job-1")  # must not raise


def test_run_calibrate_eval_only_flushes_on_disk_growth(tmp_path):
    """The polling loop calls back whenever calibrate wrote new bytes."""
    calls = []
    with patch.object(runner, "update_job"), patch.object(
        runner, "get_job", return_value={"updated_at": None}
    ):
        rc, _, _ = runner._run_calibrate_eval_only(
            ["python3", "-c", "import time; open('grew.txt','w').write('x'); time.sleep(0.3)"],
            cwd=tmp_path,
            log_dir=tmp_path,
            on_progress=lambda: calls.append(1),
            heartbeat_seconds=0.1,
            job_uuid="job-1",
        )
    assert rc == 0
    assert calls


def test_run_calibrate_eval_only_survives_a_failing_flush(tmp_path):
    with patch.object(runner, "update_job"), patch.object(
        runner, "get_job", return_value={"updated_at": None}
    ):
        rc, _, _ = runner._run_calibrate_eval_only(
            ["python3", "-c", "import time; open('grew.txt','w').write('x'); time.sleep(0.3)"],
            cwd=tmp_path,
            log_dir=tmp_path,
            on_progress=MagicMock(side_effect=RuntimeError("boom")),
            heartbeat_seconds=0.1,
            job_uuid="job-1",
        )
    assert rc == 0


def test_growing_results_csv_is_stored_across_ticks(tmp_path):
    """End to end over a results.csv that grows the way calibrate 0.0.70
    writes it: one flushed row per judged item, so a tick can land mid-line."""
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"ev-1": "Safety"}})
    )
    results = tmp_path / "results.csv"
    header = "id,gt,pred,Safety,Safety_reasoning\n"
    seen: set = set()
    stored: list = []

    def tick():
        with patch.object(
            runner, "create_evaluator_runs", side_effect=lambda rs: stored.extend(rs)
        ):
            runner._store_scored_runs(
                runner._parse_results_stt(tmp_path, [_ev_resolved()], "job-1"), seen
            )

    # One judged row, plus one caught mid-write with its score already on
    # disk and its reasoning not. Without the tail drop that row reads as
    # complete-but-unexplained, gets stored, and is never revisited.
    results.write_text(header + "i1,a,a,1,ok\ni2,b,b,0")
    tick()
    assert [r["item_id"] for r in stored] == ["i1"]

    # The cut row completes and another lands. i1 is not stored twice.
    results.write_text(header + "i1,a,a,1,ok\ni2,b,b,0,no\ni3,c,c,1,ok\n")
    tick()
    assert [r["item_id"] for r in stored] == ["i1", "i2", "i3"]
    assert stored[1]["value"] == {"value": False, "reasoning": "no"}


@pytest.mark.parametrize("env", [None, {"OPENAI_API_KEY": "sk-org"}])
def test_run_job_spawns_with_the_workspace_provider_env(env):
    resolved = {
        "uuid": "ev-1", "name": "Safety", "judge_model": "gpt",
        "system_prompt": "judge", "output_type": "binary", "output_config": {},
        "variables": [], "variable_values": {}, "kind": "single",
        "data_type": "text", "_evaluator_version_id": "ver-1",
    }
    items = [{"uuid": "item-1", "payload": {"input": "q", "output": "a"}}]
    seen = {}
    updates = []

    def fake_env(org):
        seen["org"] = org
        return env

    def popen(cmd, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    with patch.object(runner, "provider_env", side_effect=fake_env), patch.object(
        runner.subprocess, "Popen", side_effect=popen
    ), patch.object(
        runner, "get_annotation_task",
        return_value={"type": "llm-general", "org_uuid": "org-1"},
    ), patch.object(runner, "get_eval_job_items", return_value=items), patch.object(
        runner, "update_job", side_effect=lambda *a, **k: updates.append(k)
    ), patch.object(runner, "try_start_queued_job"), patch.object(
        runner, "clear_evaluator_runs_for_job", return_value=0
    ), patch.object(runner, "get_s3_client", return_value=MagicMock()), patch.object(
        runner, "get_s3_output_config", return_value="bucket"
    ), patch.object(runner, "upload_file_to_s3"):
        runner._run_job("j-1", "task-1", "u-1", [resolved], item_ids=None)

    assert seen["org"] == "org-1"
    assert seen["env"] == env
    assert any(u.get("status") == "failed" for u in updates)
