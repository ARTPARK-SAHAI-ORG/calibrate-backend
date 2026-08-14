"""Integration tests for the /traces router."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app():
    import main as main_mod

    return main_mod.app


@pytest.fixture(scope="module")
def client(app):
    with patch("main.recover_pending_jobs"):
        with TestClient(app) as c:
            yield c


def _signup(client):
    suffix = uuid.uuid4().hex[:8]
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "Tr",
            "last_name": "U",
            "email": f"tr-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _api_key_headers(client, h):
    created = client.post("/api-keys", json={"name": "ingest"}, headers=h)
    assert created.status_code == 201, created.text
    return {"X-API-Key": created.json()["key"]}


def _create_agent(client, h):
    return client.post(
        "/agents",
        json={"name": f"a-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=h,
    ).json()


def _signup_with_agent(client):
    """Sign up a fresh workspace and return its headers plus one agent's uuid."""
    h = _signup(client)
    return h, _create_agent(client, h)["uuid"]


def _payload(
    agent_id: str, message_id: str, conversation_id: str = "conv-1", **overrides
):
    payload = {
        "agent_id": agent_id,
        "message_id": message_id,
        "conversation_id": conversation_id,
        "input": [
            {"role": "system", "content": "You are a vaccination assistant."},
            {"role": "user", "content": "When is my daughter's next vaccination?"},
        ],
        "output": {
            "response": "Aapki beti ka agla vaccination 14 weeks pe hai.",
            "tool_calls": [
                {"tool": "get_schedule", "arguments": {"child_age_weeks": 14}}
            ],
        },
        "metadata": [{"key": "gen_ai.request.model", "value": "gpt-4"}],
    }
    payload.update(overrides)
    return payload


def _mid() -> str:
    return f"m-{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def test_ingest_requires_auth(client):
    agent_id = "00000000-0000-4000-8000-000000000001"
    assert client.post("/traces", json=_payload(agent_id, _mid())).status_code in (
        401,
        403,
    )
    assert (
        client.post(
            "/traces",
            json=_payload(agent_id, _mid()),
            headers={"X-API-Key": "sk_bogus"},
        ).status_code
        == 401
    )


def test_ingest_with_jwt_is_idempotent(client):
    h, agent_id = _signup_with_agent(client)
    mid = _mid()

    first = client.post("/traces", json=_payload(agent_id, mid), headers=h)
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["created"] is True
    assert len(body["uuid"]) == 36
    assert body["message_id"] == mid
    assert body["conversation_id"] == "conv-1"
    assert body["created_at"].endswith("Z") and "T" in body["created_at"]

    retry = client.post("/traces", json=_payload(agent_id, mid), headers=h)
    assert retry.status_code == 200
    assert retry.json()["created"] is False
    assert retry.json()["uuid"] == body["uuid"]


def test_ingest_with_api_key(client):
    h, agent_id = _signup_with_agent(client)
    key_headers = _api_key_headers(client, h)

    res = client.post("/traces", json=_payload(agent_id, _mid()), headers=key_headers)
    assert res.status_code == 200, res.text
    assert res.json()["created"] is True


def test_ingest_validation(client):
    h, agent_id = _signup_with_agent(client)

    # output is required.
    bad = _payload(agent_id, _mid())
    del bad["output"]
    assert client.post("/traces", json=bad, headers=h).status_code == 422

    # output needs a response or at least one tool call.
    empty_output = _payload(
        agent_id, _mid(), output={"response": "  ", "tool_calls": None}
    )
    assert client.post("/traces", json=empty_output, headers=h).status_code == 422

    # Tool-call-only turns are legal.
    tool_only = _payload(
        agent_id,
        _mid(),
        output={"tool_calls": [{"tool": "get_schedule", "arguments": {}}]},
    )
    ok = client.post("/traces", json=tool_only, headers=h)
    assert ok.status_code == 200 and ok.json()["created"] is True

    # input must be non-empty.
    assert (
        client.post(
            "/traces", json=_payload(agent_id, _mid(), input=[]), headers=h
        ).status_code
        == 422
    )

    # Unknown top-level keys are rejected; new needs belong in metadata.
    extra_top = _payload(agent_id, _mid())
    extra_top["custom_fields"] = []
    assert client.post("/traces", json=extra_top, headers=h).status_code == 422

    # Metadata entries are strict {key, value} pairs.
    bad_meta = _payload(
        agent_id, _mid(), metadata=[{"key": "k", "value": "v", "extra": 1}]
    )
    assert client.post("/traces", json=bad_meta, headers=h).status_code == 422

    # OpenAI-format extras on input turns pass through (tool_calls, tool_call_id).
    openai_history = _payload(
        agent_id,
        _mid(),
        input=[
            {"role": "user", "content": "check the schedule"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_schedule", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "{\"weeks\": 14}", "tool_call_id": "call_1"},
        ],
    )
    ok = client.post("/traces", json=openai_history, headers=h)
    assert ok.status_code == 200 and ok.json()["created"] is True


def test_ingest_cap_returns_429_but_keeps_retries_idempotent(client, monkeypatch):
    from routers import org_limits as org_limits_mod

    h, agent_id = _signup_with_agent(client)
    monkeypatch.setattr(org_limits_mod, "DEFAULT_MAX_TRACES", 1)

    first_mid = _mid()
    assert (
        client.post(
            "/traces", json=_payload(agent_id, first_mid), headers=h
        ).status_code
        == 200
    )

    capped = client.post("/traces", json=_payload(agent_id, _mid()), headers=h)
    assert capped.status_code == 429
    detail = capped.json()["detail"]
    assert detail["current"] == 1
    assert detail["max_traces"] == 1
    assert "hint" in detail

    # A retry of an already-stored message_id still succeeds at the cap.
    retry = client.post("/traces", json=_payload(agent_id, first_mid), headers=h)
    assert retry.status_code == 200
    assert retry.json()["created"] is False


def test_ingest_requires_a_known_agent(client):
    h, agent_id = _signup_with_agent(client)

    missing_agent = _payload(agent_id, _mid())
    del missing_agent["agent_id"]
    assert client.post("/traces", json=missing_agent, headers=h).status_code == 422

    unknown = client.post(
        "/traces",
        json=_payload("00000000-0000-4000-8000-000000000002", _mid()),
        headers=h,
    )
    assert unknown.status_code == 404, unknown.text
    assert unknown.json()["detail"] == "Agent not found"


def test_ingest_rejects_agent_from_another_workspace(client):
    h, _ = _signup_with_agent(client)
    _, other_agent_id = _signup_with_agent(client)

    res = client.post("/traces", json=_payload(other_agent_id, _mid()), headers=h)
    assert res.status_code == 404, res.text
    assert res.json()["detail"] == "Agent not found"


def test_ingest_checks_agent_before_idempotency(client):
    h, agent_id = _signup_with_agent(client)
    mid = _mid()
    assert (
        client.post("/traces", json=_payload(agent_id, mid), headers=h).status_code
        == 200
    )

    retry = client.post(
        "/traces",
        json=_payload("00000000-0000-4000-8000-000000000003", mid),
        headers=h,
    )
    assert retry.status_code == 404, retry.text


def test_message_id_is_unique_per_workspace_not_per_agent(client):
    h, agent_id = _signup_with_agent(client)
    other_agent_id = _create_agent(client, h)["uuid"]
    mid = _mid()

    first = client.post("/traces", json=_payload(agent_id, mid), headers=h)
    assert first.status_code == 200, first.text
    second = client.post("/traces", json=_payload(other_agent_id, mid), headers=h)
    assert second.status_code == 200, second.text
    assert second.json()["created"] is False
    assert second.json()["uuid"] == first.json()["uuid"]

    full = client.get(f"/traces/{first.json()['uuid']}", headers=h).json()
    assert full["agent_id"] == agent_id


# ---------------------------------------------------------------------------
# List / detail / bulk delete (curation surface, JWT-only)
# ---------------------------------------------------------------------------


def test_curation_endpoints_are_jwt_only(client):
    h = _signup(client)
    key_headers = _api_key_headers(client, h)

    assert client.get("/traces").status_code in (401, 403)
    assert client.get("/traces", headers=key_headers).status_code in (401, 403)
    assert (
        client.post(
            "/traces/bulk-delete", json={"select_all": True}, headers=key_headers
        ).status_code
        in (401, 403)
    )


def test_list_and_detail_roundtrip(client):
    h, agent_id = _signup_with_agent(client)

    mid_a = _mid()
    client.post(
        "/traces", json=_payload(agent_id, mid_a, conversation_id="conv-a"), headers=h
    )
    mid_b = _mid()
    openai_extras = [
        {"role": "user", "content": "check the POLIO schedule"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_schedule", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "{\"weeks\": 14}", "tool_call_id": "call_1"},
        {"role": "user", "content": "and in months?"},
    ]
    created_b = client.post(
        "/traces",
        json=_payload(agent_id, mid_b, conversation_id="conv-b", input=openai_extras),
        headers=h,
    ).json()

    listed = client.get("/traces", headers=h)
    assert listed.status_code == 200
    body = listed.json()
    assert set(body) == {"items", "total", "limit", "offset"}
    assert body["total"] == 2 and body["limit"] == 50 and body["offset"] == 0
    # Newest first.
    assert [item["message_id"] for item in body["items"]] == [mid_b, mid_a]
    summary_b = body["items"][0]
    assert summary_b["turn_count"] == 4
    assert summary_b["tool_call_count"] == 1
    assert summary_b["metadata_count"] == 1
    assert summary_b["input_preview"] == "and in months?"
    assert summary_b["response_preview"].startswith("Aapki beti")

    detail = client.get(f"/traces/{created_b['uuid']}", headers=h)
    assert detail.status_code == 200
    full = detail.json()
    assert full["conversation_id"] == "conv-b"
    # OpenAI-format extras on history turns survive storage verbatim.
    assert full["input"][1]["tool_calls"][0]["function"]["name"] == "get_schedule"
    assert full["input"][2]["tool_call_id"] == "call_1"
    assert full["output"]["tool_calls"][0] == {
        "tool": "get_schedule",
        "arguments": {"child_age_weeks": 14},
    }
    assert full["metadata"] == [{"key": "gen_ai.request.model", "value": "gpt-4"}]

    assert (
        client.get(
            "/traces/00000000-0000-4000-8000-000000000001", headers=h
        ).status_code
        == 404
    )
    # Another workspace can't read this trace.
    other = _signup(client)
    assert (
        client.get(f"/traces/{created_b['uuid']}", headers=other).status_code == 404
    )


def test_output_with_multiple_tool_calls_roundtrips(client):
    # A Responses turn can emit several tool calls (parallel, or accumulated
    # across round-trips) alongside its reply, so output.tool_calls is a list:
    # every entry, in order, must survive the count and the detail body.
    h, agent_id = _signup_with_agent(client)
    mid = _mid()
    output = {
        "response": "You're booked for Thursday at 4:30 PM.",
        "tool_calls": [
            {"tool": "check_availability", "arguments": {"date": "Thursday"}},
            {
                "tool": "book_appointment",
                "arguments": {
                    "patient_name": "Sam",
                    "date": "Thursday",
                    "time_slot": "4:30 PM",
                },
            },
        ],
    }
    created = client.post(
        "/traces", json=_payload(agent_id, mid, output=output), headers=h
    )
    assert created.status_code == 200, created.text
    assert created.json()["created"] is True

    summary = client.get("/traces", headers=h).json()["items"][0]
    assert summary["tool_call_count"] == 2
    assert summary["response_preview"].startswith("You're booked")

    full = client.get(f"/traces/{created.json()['uuid']}", headers=h).json()
    assert full["output"]["tool_calls"] == output["tool_calls"]
    assert full["output"]["response"] == output["response"]


def test_list_search_filter_and_pagination(client):
    h, agent_id = _signup_with_agent(client)
    mid_polio = _mid()
    client.post(
        "/traces",
        json=_payload(
            agent_id,
            mid_polio,
            conversation_id="conv-x",
            input=[{"role": "user", "content": "Tell me about POLIO boosters"}],
        ),
        headers=h,
    )
    client.post(
        "/traces", json=_payload(agent_id, _mid(), conversation_id="conv-y"), headers=h
    )
    client.post(
        "/traces", json=_payload(agent_id, _mid(), conversation_id="conv-y"), headers=h
    )

    hits = client.get("/traces", params={"q": "polio"}, headers=h).json()
    assert hits["total"] == 1
    assert hits["items"][0]["message_id"] == mid_polio

    conv = client.get(
        "/traces", params={"conversation_id": "conv-y"}, headers=h
    ).json()
    assert conv["total"] == 2

    page = client.get(
        "/traces", params={"limit": 1, "offset": 1}, headers=h
    ).json()
    assert page["total"] == 3
    assert len(page["items"]) == 1
    assert page["limit"] == 1 and page["offset"] == 1


def test_bulk_delete_router_contract(client):
    h, agent_id = _signup_with_agent(client)
    mid_keep = _mid()
    kept = client.post(
        "/traces",
        json=_payload(agent_id, mid_keep, conversation_id="conv-keep"),
        headers=h,
    ).json()
    mid_gone = _mid()
    client.post(
        "/traces",
        json=_payload(agent_id, mid_gone, conversation_id="conv-gone"),
        headers=h,
    )
    client.post(
        "/traces",
        json=_payload(agent_id, _mid(), conversation_id="conv-gone"),
        headers=h,
    )

    # Neither ids nor select_all is a 400.
    assert (
        client.post("/traces/bulk-delete", json={}, headers=h).status_code == 400
    )

    # select_all with a conversation filter deletes exactly that set.
    filtered = client.post(
        "/traces/bulk-delete",
        json={"select_all": True, "conversation_id": "conv-gone"},
        headers=h,
    )
    assert filtered.status_code == 200
    assert filtered.json() == {"deleted": 2}
    assert client.get("/traces", headers=h).json()["total"] == 1

    # Deleting frees the message_id: the same ID re-ingests as a new trace.
    by_ids = client.post(
        "/traces/bulk-delete", json={"trace_ids": [kept["uuid"]]}, headers=h
    )
    assert by_ids.status_code == 200 and by_ids.json() == {"deleted": 1}
    assert client.get(f"/traces/{kept['uuid']}", headers=h).status_code == 404

    reingested = client.post("/traces", json=_payload(agent_id, mid_keep), headers=h)
    assert reingested.status_code == 200
    assert reingested.json()["created"] is True
    assert reingested.json()["uuid"] != kept["uuid"]


def test_agent_id_is_returned_on_list_and_detail(client):
    h, agent_id = _signup_with_agent(client)
    created = client.post("/traces", json=_payload(agent_id, _mid()), headers=h).json()

    assert client.get("/traces", headers=h).json()["items"][0]["agent_id"] == agent_id

    detail = client.get(f"/traces/{created['uuid']}", headers=h)
    assert detail.status_code == 200, detail.text
    assert detail.json()["agent_id"] == agent_id


def test_list_filters_by_agent_id(client):
    h, agent_a = _signup_with_agent(client)
    agent_b = _create_agent(client, h)["uuid"]
    mid_a = _mid()
    client.post(
        "/traces",
        json=_payload(
            agent_a,
            mid_a,
            conversation_id="conv-a",
            input=[{"role": "user", "content": "POLIO booster for agent a"}],
        ),
        headers=h,
    )
    client.post(
        "/traces",
        json=_payload(agent_a, _mid(), conversation_id="conv-other"),
        headers=h,
    )
    client.post(
        "/traces",
        json=_payload(
            agent_b,
            _mid(),
            conversation_id="conv-a",
            input=[{"role": "user", "content": "POLIO booster for agent b"}],
        ),
        headers=h,
    )

    only_a = client.get("/traces", params={"agent_id": agent_a}, headers=h).json()
    assert only_a["total"] == 2
    assert {item["agent_id"] for item in only_a["items"]} == {agent_a}

    # agent_id narrows q and conversation_id rather than replacing them.
    combined = client.get(
        "/traces",
        params={"agent_id": agent_a, "q": "polio", "conversation_id": "conv-a"},
        headers=h,
    ).json()
    assert combined["total"] == 1
    assert combined["items"][0]["message_id"] == mid_a


def test_bulk_delete_select_all_scopes_to_one_agent(client):
    h, agent_a = _signup_with_agent(client)
    agent_b = _create_agent(client, h)["uuid"]
    client.post("/traces", json=_payload(agent_a, _mid()), headers=h)
    client.post("/traces", json=_payload(agent_a, _mid()), headers=h)
    mid_b = _mid()
    client.post("/traces", json=_payload(agent_b, mid_b), headers=h)

    res = client.post(
        "/traces/bulk-delete",
        json={"select_all": True, "agent_id": agent_a},
        headers=h,
    )
    assert res.status_code == 200, res.text
    assert res.json() == {"deleted": 2}

    remaining = client.get("/traces", headers=h).json()
    assert remaining["total"] == 1
    assert remaining["items"][0]["message_id"] == mid_b
    assert remaining["items"][0]["agent_id"] == agent_b


# ---------------------------------------------------------------------------
# Convert traces -> tests (Phase 3)
# ---------------------------------------------------------------------------


def _create_llm_evaluator(client, h):
    """Create an llm evaluator (36-char uuid); returns its uuid."""
    created = client.post(
        "/evaluators",
        json={
            "name": f"ev-{uuid.uuid4()}",
            "description": "d",
            "evaluator_type": "llm",
            "data_type": "text",
            "kind": "single",
            "output_type": "binary",
            "version": {
                "judge_model": "openai/gpt-4",
                "system_prompt": "Judge {{criteria}}",
                "variables": [{"name": "criteria"}],
            },
        },
        headers=h,
    )
    assert created.status_code == 200, created.text
    return created.json()["uuid"]


def test_convert_requires_scope(client):
    h = _signup(client)
    res = client.post("/traces/convert-to-tests", json={"type": "response"}, headers=h)
    assert res.status_code == 400


def test_convert_is_jwt_only(client):
    h = _signup(client)
    key_headers = _api_key_headers(client, h)
    res = client.post(
        "/traces/convert-to-tests",
        json={"trace_ids": ["x"], "type": "response"},
        headers=key_headers,
    )
    assert res.status_code in (401, 403)


def test_convert_response_requires_evaluator(client):
    h, agent_id = _signup_with_agent(client)
    mid = _mid()
    trace = client.post("/traces", json=_payload(agent_id, mid), headers=h).json()
    res = client.post(
        "/traces/convert-to-tests",
        json={"trace_ids": [trace["uuid"]], "type": "response"},
        headers=h,
    )
    assert res.status_code == 400


def test_convert_response_creates_tests_links_evaluator_and_agent(client):
    h, agent_id = _signup_with_agent(client)
    ev_uuid = _create_llm_evaluator(client, h)
    agent = _create_agent(client, h)

    mid = _mid()
    trace = client.post("/traces", json=_payload(agent_id, mid), headers=h).json()

    res = client.post(
        "/traces/convert-to-tests",
        json={
            "trace_ids": [trace["uuid"]],
            "type": "response",
            "evaluators": [
                {"evaluator_uuid": ev_uuid, "variable_values": {"criteria": "be nice"}}
            ],
            "agent_uuids": [agent["uuid"]],
        },
        headers=h,
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["created"] == 1
    test_uuid = body["test_uuids"][0]

    # The created test: type response, history copied verbatim from trace input,
    # output discarded (no tool_calls in evaluation), evaluator linked.
    detail = client.get(f"/tests/{test_uuid}", headers=h).json()
    assert detail["type"] == "response"
    assert detail["name"] == mid
    assert detail["config"]["history"] == _payload(agent_id, mid)["input"]
    assert detail["config"]["evaluation"] == {"type": "response"}
    assert [e["uuid"] for e in detail["evaluators"]] == [ev_uuid]

    # Linked to the agent.
    linked = client.get(f"/agent-tests/agent/{agent['uuid']}/tests", headers=h).json()
    linked_uuids = [t["uuid"] for t in linked["items"]]
    assert test_uuid in linked_uuids


def test_convert_tool_call_copies_recorded_calls(client):
    h, agent_id = _signup_with_agent(client)
    mid = _mid()
    trace = client.post("/traces", json=_payload(agent_id, mid), headers=h).json()

    res = client.post(
        "/traces/convert-to-tests",
        json={
            "trace_ids": [trace["uuid"]],
            "type": "tool_call",
            "accept_any_arguments": False,
        },
        headers=h,
    )
    assert res.status_code == 200, res.text
    test_uuid = res.json()["test_uuids"][0]

    detail = client.get(f"/tests/{test_uuid}", headers=h).json()
    assert detail["type"] == "tool_call"
    tool_calls = detail["config"]["evaluation"]["tool_calls"]
    # The recorded {tool, arguments} become the expected assertion; the raw
    # argument values pass straight through (legacy exact match).
    assert tool_calls == [
        {
            "tool": "get_schedule",
            "arguments": {"child_age_weeks": 14},
            "accept_any_arguments": False,
        }
    ]


def test_convert_tool_call_rejects_traces_without_tool_calls(client):
    h, agent_id = _signup_with_agent(client)
    mid = _mid()
    # Response-only trace: no tool_calls to assert.
    trace = client.post(
        "/traces",
        json=_payload(
            agent_id, mid, output={"response": "just text", "tool_calls": None}
        ),
        headers=h,
    ).json()
    res = client.post(
        "/traces/convert-to-tests",
        json={"trace_ids": [trace["uuid"]], "type": "tool_call"},
        headers=h,
    )
    assert res.status_code == 400
    assert mid in res.json()["detail"]["message_ids"]


def test_convert_select_all_with_conversation_filter(client):
    h, agent_id = _signup_with_agent(client)
    ev_uuid = _create_llm_evaluator(client, h)
    conv = f"cc-{uuid.uuid4().hex[:8]}"
    client.post(
        "/traces", json=_payload(agent_id, _mid(), conversation_id=conv), headers=h
    )
    client.post(
        "/traces", json=_payload(agent_id, _mid(), conversation_id=conv), headers=h
    )
    client.post(
        "/traces", json=_payload(agent_id, _mid(), conversation_id="other"), headers=h
    )

    res = client.post(
        "/traces/convert-to-tests",
        json={
            "select_all": True,
            "conversation_id": conv,
            "type": "response",
            "evaluators": [{"evaluator_uuid": ev_uuid}],
        },
        headers=h,
    )
    assert res.status_code == 200, res.text
    assert res.json()["created"] == 2


def test_convert_dedupes_names_on_repeat(client):
    h, agent_id = _signup_with_agent(client)
    ev_uuid = _create_llm_evaluator(client, h)
    mid = _mid()
    trace = client.post("/traces", json=_payload(agent_id, mid), headers=h).json()
    body = {
        "trace_ids": [trace["uuid"]],
        "type": "response",
        "evaluators": [{"evaluator_uuid": ev_uuid}],
    }
    first = client.post("/traces/convert-to-tests", json=body, headers=h)
    second = client.post("/traces/convert-to-tests", json=body, headers=h)
    assert first.status_code == 200 and second.status_code == 200

    first_detail = client.get(f"/tests/{first.json()['test_uuids'][0]}", headers=h).json()
    second_detail = client.get(
        f"/tests/{second.json()['test_uuids'][0]}", headers=h
    ).json()
    assert first_detail["name"] == mid
    # Re-converting the same trace does not 400 on the name clash; it suffixes.
    assert second_detail["name"] == f"{mid} (2)"


def test_convert_no_matching_traces_404(client):
    h = _signup(client)
    ev_uuid = _create_llm_evaluator(client, h)
    res = client.post(
        "/traces/convert-to-tests",
        json={
            "trace_ids": ["00000000-0000-4000-8000-000000000009"],
            "type": "response",
            "evaluators": [{"evaluator_uuid": ev_uuid}],
        },
        headers=h,
    )
    assert res.status_code == 404
