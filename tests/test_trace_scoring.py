"""Unit tests for trace-scoring eligibility / plan resolution."""

from __future__ import annotations

import uuid

import db
import trace_scoring as ts


def _ev(evaluator_type, *, live=None, name="ev", ev_uuid=None):
    return {
        "uuid": ev_uuid or str(uuid.uuid4()),
        "name": name,
        "evaluator_type": evaluator_type,
        "live_version_id": live,
    }


def _version(version_uuid, *, variables=None):
    return {"uuid": version_uuid, "variables": variables}


def test_conversation_agent_maps_to_response_llm():
    live = str(uuid.uuid4())
    ev = _ev("llm", live=live, name="clean")
    result = ts.resolve_trace_scoring("conversation", [(ev, _version(live))])
    assert result.evaluation_type == "response"
    assert result.evaluator_type == "llm"
    assert [p.evaluator_uuid for p in result.eligible] == [ev["uuid"]]
    assert result.eligible[0].evaluator_version_id == live
    assert result.ineligible == []
    assert result.as_plan() == ts.ScoringPlan(
        type="response",
        evaluators=[
            ts.ScoringPlanPin(
                evaluator_uuid=ev["uuid"], evaluator_version_id=live
            )
        ],
    )


def test_general_agent_maps_to_general_llm_general():
    live = str(uuid.uuid4())
    ev = _ev("llm-general", live=live, name="gen")
    result = ts.resolve_trace_scoring("general", [(ev, _version(live))])
    assert result.evaluation_type == "general"
    assert result.evaluator_type == "llm-general"
    assert result.eligible[0].evaluator_uuid == ev["uuid"]
    assert result.as_plan().type == "general"


def test_mixed_evaluator_types_are_filtered_before_validation():
    """Wrong-type evaluators are ineligible even if they also declare variables
    or lack a live version — type is checked first so a mixed set never raises."""
    live = str(uuid.uuid4())
    vars_id = str(uuid.uuid4())
    clean = _ev("llm", live=live, name="clean")
    wrong_with_vars = _ev("llm-general", live=vars_id, name="general-vars")
    wrong_no_live = _ev("stt", live=None, name="stt")
    conversation = _ev("conversation", live=live, name="sim")
    result = ts.resolve_trace_scoring(
        "conversation",
        [
            (clean, _version(live)),
            (wrong_with_vars, _version(vars_id, variables=[{"name": "criteria"}])),
            (wrong_no_live, None),
            (conversation, _version(live)),
        ],
    )
    assert [p.name for p in result.eligible] == ["clean"]
    by_name = {i.name: i.reason for i in result.ineligible}
    assert by_name == {
        "general-vars": ts.IneligibleReason.WRONG_TYPE,
        "stt": ts.IneligibleReason.WRONG_TYPE,
        "sim": ts.IneligibleReason.WRONG_TYPE,
    }


def test_no_live_version_disqualifies():
    ev_none = _ev("llm", live=None, name="none")
    ev_missing = _ev("llm", live=str(uuid.uuid4()), name="missing")
    result = ts.resolve_trace_scoring(
        "conversation", [(ev_none, None), (ev_missing, None)]
    )
    assert result.eligible == []
    assert {i.name: i.reason for i in result.ineligible} == {
        "none": ts.IneligibleReason.NO_LIVE_VERSION,
        "missing": ts.IneligibleReason.NO_LIVE_VERSION,
    }
    assert result.as_plan() == ts.ScoringPlanSkip(skip="no_usable_evaluators")


def test_declares_variables_disqualifies():
    live = str(uuid.uuid4())
    ev = _ev("llm", live=live, name="criteria")
    result = ts.resolve_trace_scoring(
        "conversation",
        [(ev, _version(live, variables=[{"name": "criteria"}]))],
    )
    assert result.eligible == []
    assert result.ineligible[0].reason == ts.IneligibleReason.DECLARES_VARIABLES


def test_empty_variables_list_is_eligible():
    live = str(uuid.uuid4())
    ev = _ev("llm", live=live)
    result = ts.resolve_trace_scoring(
        "conversation", [(ev, _version(live, variables=[]))]
    )
    assert len(result.eligible) == 1


def test_unsupported_interaction_type_skips_and_marks_wrong_type():
    live = str(uuid.uuid4())
    ev = _ev("llm", live=live, name="ok")
    result = ts.resolve_trace_scoring("voice", [(ev, _version(live))])
    assert result.evaluation_type is None
    assert result.eligible == []
    assert result.ineligible[0].reason == ts.IneligibleReason.WRONG_TYPE
    assert result.as_plan() == ts.ScoringPlanSkip(skip="unsupported_interaction_type")


def test_empty_linked_set_is_not_usable():
    result = ts.resolve_trace_scoring("conversation", [])
    assert result.eligible == []
    assert result.ineligible == []
    assert result.as_plan() == ts.ScoringPlanSkip(skip="no_usable_evaluators")


def test_resolve_live_evaluators_pairs_version_or_none():
    org = str(uuid.uuid4())
    agent_uuid = str(uuid.uuid4())
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO agents (uuid, org_uuid, name, config, interaction_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (agent_uuid, org, "score-me", "{}", "general"),
        )
        conn.commit()
    live_ev = db.create_evaluator(
        name=f"gen-{uuid.uuid4().hex[:6]}",
        evaluator_type="llm-general",
        org_uuid=org,
    )
    version = db.create_evaluator_version(live_ev, "openai/gpt-4.1", "Judge it.")
    db.set_evaluator_live_version(live_ev, version["uuid"])
    db.add_evaluator_to_agent(agent_uuid, live_ev)

    bare_ev = db.create_evaluator(
        name=f"bare-{uuid.uuid4().hex[:6]}",
        evaluator_type="llm-general",
        org_uuid=org,
    )
    db.add_evaluator_to_agent(agent_uuid, bare_ev)

    pairs = db.resolve_live_evaluators(agent_uuid)
    by_uuid = {ev["uuid"]: version_row for ev, version_row in pairs}
    assert by_uuid[live_ev]["uuid"] == version["uuid"]
    assert by_uuid[bare_ev] is None

    agent = db.get_agent(agent_uuid)
    result = ts.resolve_trace_scoring(agent["interaction_type"], pairs)
    assert result.evaluation_type == "general"
    assert [p.evaluator_uuid for p in result.eligible] == [live_ev]
    assert result.as_plan() == ts.ScoringPlan(
        type="general",
        evaluators=[
            ts.ScoringPlanPin(
                evaluator_uuid=live_ev, evaluator_version_id=version["uuid"]
            )
        ],
    )
    assert result.ineligible[0].evaluator_uuid == bare_ev
    assert result.ineligible[0].reason == ts.IneligibleReason.NO_LIVE_VERSION
