"""The public chat link: /public/agent-chat/{token}."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

import db
import provider_keys as pk
from routers import org_limits
from routers import public


@pytest.fixture(scope="module")
def client():
    import main as main_mod

    with patch("main.recover_pending_jobs"):
        with TestClient(main_mod.app) as c:
            yield c


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("FAKE_AI_PROVIDERS", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-server")
    monkeypatch.setenv("PROVIDER_KEYS_ENCRYPTION_KEY", Fernet.generate_key().decode())


def _signup(client):
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "C",
            "last_name": "L",
            "email": f"chat-{uuid.uuid4().hex[:8]}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _chat_agent(client, interaction_type=None, config=None, h=None):
    """A fresh agent with its chat link on: (headers, agent, token)."""
    h = h or _signup(client)
    body = {"name": f"a-{uuid.uuid4().hex[:6]}", "type": "agent"}
    if interaction_type:
        body["interaction_type"] = interaction_type
    agent = client.post("/agents", json=body, headers=h).json()
    if config is not None:
        db.update_agent(agent["uuid"], config=config)
    token = client.patch(
        f"/agents/{agent['uuid']}/chat-link/visibility",
        json={"is_public": True},
        headers=h,
    ).json()["share_token"]
    return h, db.get_agent(agent["uuid"]), token


def _llm(reply="Hello", tool_calls=()):
    return AsyncMock(return_value={"reply": reply, "tool_calls": list(tool_calls), "ended": False})


def _send(client, token, message="Hi", conversation_id=None, headers=None, history=()):
    return client.post(
        f"/public/agent-chat/{token}/messages",
        json={"message": message, "conversation_id": conversation_id, "history": list(history)},
        headers=headers or {},
    )


def _turns(*pairs):
    """The history the chat page sends: each user message, then the reply it got."""
    history = []
    for message, reply in pairs:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply["reply"], "tool_calls": reply["tool_calls"]})
    return history


def _traces(org_uuid):
    return db.list_traces(org_uuid, limit=100, offset=0)[0]


def _hits(org_uuid):
    with db.get_db_connection() as conn:
        return [
            tuple(r)
            for r in conn.execute(
                "SELECT limit_name, kind, requested, cap FROM limit_hits WHERE org_uuid = ?",
                (org_uuid,),
            ).fetchall()
        ]


CONNECTION = {"agent_url": "https://example.com/a", "connection_verified": True, "agent_headers": {"X-Key": "k"}}


def test_get_names_the_agent(client):
    _, agent, token = _chat_agent(client, interaction_type="general")
    assert client.get(f"/public/agent-chat/{token}").json() == {
        "agent_name": agent["name"],
        "interaction_type": "general",
    }


def test_link_that_is_off_unknown_or_of_a_deleted_agent_is_not_found(client):
    h, agent, token = _chat_agent(client)
    unknown = str(uuid.uuid4())
    assert client.get(f"/public/agent-chat/{unknown}").json()["detail"] == {"error": "not_found"}

    client.patch(f"/agents/{agent['uuid']}/chat-link/visibility", json={"is_public": False}, headers=h)
    assert client.get(f"/public/agent-chat/{token}").status_code == 404
    assert _send(client, token).json()["detail"] == {"error": "not_found"}
    stranger = _signup(client)
    assert client.get(f"/public/agent-chat/{token}", headers=stranger).status_code == 404
    bad_login = {"Authorization": "Bearer not-a-token"}
    assert client.get(f"/public/agent-chat/{token}", headers=bad_login).status_code == 404

    client.patch(f"/agents/{agent['uuid']}/chat-link/visibility", json={"is_public": True}, headers=h)
    assert client.get(f"/public/agent-chat/{token}").status_code == 200
    db.delete_agent(agent["uuid"])
    assert client.get(f"/public/agent-chat/{token}").status_code == 404


def test_a_member_logged_in_can_preview_a_private_link(client):
    h, agent, token = _chat_agent(client)
    client.patch(f"/agents/{agent['uuid']}/chat-link/visibility", json={"is_public": False}, headers=h)

    assert client.get(f"/public/agent-chat/{token}", headers=h).status_code == 200
    with patch.object(public, "_llm_reply", _llm()):
        body = _send(client, token, headers=h).json()
    assert body["reply"] == "Hello"
    assert _feedback(client, token, body["trace_uuid"], True, headers=h).status_code == 200
    assert _feedback(client, token, body["trace_uuid"], True).status_code == 404


def test_reply_is_stored_as_a_chat_trace_with_a_new_conversation(client):
    _, agent, token = _chat_agent(client)
    with patch.object(public, "_llm_reply", _llm("Hello there")) as llm:
        body = _send(client, token, "Hi").json()

    uuid.UUID(body["conversation_id"])
    assert body["reply"] == "Hello there"
    assert body["tool_calls"] == [] and body["ended"] is False
    kwargs = llm.call_args.kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "Hi"}]
    assert kwargs["model"] == agent["config"]["llm"]["model"] and kwargs["provider"] == "openrouter"
    assert kwargs["system_prompt"] == agent["config"]["system_prompt"]
    assert kwargs["tools"] == []
    (trace,) = _traces(agent["org_uuid"])
    assert trace["uuid"] == body["trace_uuid"]
    assert trace["source"] == "chat"
    assert trace["conversation_id"] == body["conversation_id"]
    assert trace["input"] == [{"role": "user", "content": "Hi"}]
    assert trace["output"] == {"response": "Hello there", "tool_calls": []}


def test_history_sent_with_the_message_reaches_the_agent_with_its_tool_calls(client):
    _, agent, token = _chat_agent(client)
    calls = [{"id": "x", "tool": "book", "arguments": {"day": 1}, "output": {"ok": True}}]
    with patch.object(public, "_llm_reply", _llm("Booked", calls)):
        first = _send(client, token, "Book it").json()
    assert first["tool_calls"] == [{"tool": "book", "arguments": {"day": 1}, "output": {"ok": True}}]

    with patch.object(public, "_llm_reply", _llm("Done")) as llm:
        second = _send(
            client, token, "Thanks", first["conversation_id"], history=_turns(("Book it", first))
        ).json()

    call_id = "call_1_0"
    expected = [
        {"role": "user", "content": "Book it"},
        {
            "role": "assistant",
            "content": "Booked",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "book", "arguments": '{"day": 1}'}}
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": '{"ok": true}'},
        {"role": "user", "content": "Thanks"},
    ]
    assert llm.call_args.kwargs["messages"] == expected
    assert second["conversation_id"] == first["conversation_id"]
    assert db.get_trace(agent["org_uuid"], second["trace_uuid"])["input"] == expected


def test_a_conversation_id_alone_brings_back_nothing_stored(client):
    _, agent, token = _chat_agent(client)
    db.create_trace_with_eval_run(
        org_uuid=agent["org_uuid"],
        agent=agent,
        conversation_id="shared",
        input=[{"role": "user", "content": "secret"}],
        output={"response": "private", "tool_calls": []},
        max_scored_traces=None,
        batch_size=20,
        source="chat",
    )
    with patch.object(public, "_llm_reply", _llm()) as llm:
        _send(client, token, "Hi", "shared")
    assert llm.call_args.kwargs["messages"] == [{"role": "user", "content": "Hi"}]


def test_a_full_workspace_keeps_the_conversation(client, monkeypatch):
    _, agent, token = _chat_agent(client)
    _limits(monkeypatch, max_traces=0)
    with patch.object(public, "_llm_reply", _llm("Hello")):
        first = _send(client, token, "Hi").json()
    assert first["trace_uuid"] is None
    with patch.object(public, "_llm_reply", _llm()) as llm:
        _send(client, token, "Again", first["conversation_id"], history=_turns(("Hi", first)))
    assert llm.call_args.kwargs["messages"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Again"},
    ]


def test_general_agent_gets_only_the_new_message(client):
    _, agent, token = _chat_agent(client, interaction_type="general")
    with patch.object(public, "_llm_reply", _llm()):
        first = _send(client, token, "One").json()
    with patch.object(public, "_llm_reply", _llm()) as llm:
        second = _send(client, token, "Two", first["conversation_id"], history=_turns(("One", first))).json()
    assert llm.call_args.kwargs["messages"] == [{"role": "user", "content": "Two"}]
    assert db.get_trace(agent["org_uuid"], second["trace_uuid"])["input"] == "Two"


def test_conversation_that_would_pass_the_turn_limit_is_refused(client, monkeypatch):
    _, agent, token = _chat_agent(client)
    monkeypatch.setattr(public, "MAX_CHAT_TURNS", 2)
    with patch.object(public, "_llm_reply", _llm()):
        first = _send(client, token).json()
    with patch.object(public, "_llm_reply", _llm()) as llm:
        refused = _send(client, token, "Again", first["conversation_id"], history=_turns(("Hi", first)))
    assert refused.status_code == 409
    assert refused.json()["detail"] == {"error": "conversation_too_long"}
    llm.assert_not_called()


def test_a_conversation_too_large_is_refused_before_a_reply_is_counted(client, monkeypatch):
    _, agent, token = _chat_agent(client)
    monkeypatch.setattr(public, "MAX_CHAT_HISTORY_CHARS", 100)
    big = [{"role": "user", "content": "x" * 200}, {"role": "assistant", "content": "ok", "tool_calls": []}]
    with patch.object(public, "_llm_reply", _llm()) as llm:
        refused = _send(client, token, "Again", history=big)
    assert refused.status_code == 409
    assert refused.json()["detail"] == {"error": "conversation_too_long"}
    llm.assert_not_called()
    assert db.count_chat_replies_today(agent["uuid"]) == 0


def test_a_reply_cut_off_by_the_person_leaving_is_not_counted(client):
    _, agent, token = _chat_agent(client)
    with patch.object(public, "_llm_reply", AsyncMock(side_effect=asyncio.CancelledError())):
        with pytest.raises(BaseException):
            _send(client, token)
    assert db.count_chat_replies_today(agent["uuid"]) == 0


def _limits(monkeypatch, **overrides):
    real = org_limits.effective_limits
    monkeypatch.setattr(public, "effective_limits", lambda org: {**real(org), **overrides})


def test_daily_limit_refuses_and_records_the_hit(client, monkeypatch):
    _, agent, token = _chat_agent(client)
    _limits(monkeypatch, max_chat_replies_per_day=1)
    with patch.object(public, "_llm_reply", _llm()):
        assert _send(client, token).status_code == 200
    with patch.object(public, "_llm_reply", _llm()) as llm:
        refused = _send(client, token)
    assert refused.status_code == 429
    assert refused.json()["detail"] == {"error": "daily_limit", "limit": 1}
    assert ("max_chat_replies_per_day", "chat", 2, 1) in _hits(agent["org_uuid"])
    llm.assert_not_called()


def test_the_last_reply_of_the_day_is_taken_before_the_agent_is_asked(client, monkeypatch):
    _, agent, token = _chat_agent(client)
    _limits(monkeypatch, max_chat_replies_per_day=1)

    # While the agent is still answering, the reply is already counted, so a
    # second message arriving now cannot take it too.
    async def answering(**kwargs):
        seen.append(db.count_chat_replies_today(agent["uuid"]))
        seen.append(db.reserve_chat_reply(agent["uuid"], 1))
        return {"reply": "Hi", "tool_calls": [], "ended": False}

    seen = []
    with patch.object(public, "_llm_reply", answering):
        assert _send(client, token).status_code == 200
    assert seen == [1, False]
    assert db.count_chat_replies_today(agent["uuid"]) == 1


def _send_until_refused(client, token, attempts=5):
    with patch.object(public, "_llm_reply", _llm()):
        for sent in range(attempts):
            response = _send(client, token)
            if response.status_code != 200:
                return sent, response
    return attempts, None


def test_the_agents_own_daily_number_is_enforced(client, monkeypatch):
    _, agent, token = _chat_agent(client, config={"talk_to_agent": {"messages_per_day": 2}})
    sent, refused = _send_until_refused(client, token)
    assert sent == 2
    assert refused.json()["detail"] == {"error": "daily_limit", "limit": 2}


def test_the_workspace_default_applies_to_an_agent_without_its_own_number(client):
    h, agent, token = _chat_agent(client)
    client.patch(
        f"/organizations/{agent['org_uuid']}",
        json={"settings": {"talk_to_agent": {"default_messages_per_day": 3}}},
        headers=h,
    )
    sent, refused = _send_until_refused(client, token)
    assert sent == 3
    assert refused.json()["detail"] == {"error": "daily_limit", "limit": 3}


def test_no_workspace_maximum_still_applies_the_agents_number(client, monkeypatch):
    _, _, token = _chat_agent(client, config={"talk_to_agent": {"messages_per_day": 1}})
    _limits(monkeypatch, max_chat_replies_per_day=None)
    sent, refused = _send_until_refused(client, token)
    assert sent == 1 and refused.status_code == 429


def test_a_full_workspace_still_replies_but_stores_no_trace(client, monkeypatch):
    _, agent, token = _chat_agent(client)
    with patch.object(public, "_llm_reply", _llm()):
        _send(client, token)
    _limits(monkeypatch, max_traces=1)
    with patch.object(public, "_llm_reply", _llm("Still here")):
        body = _send(client, token)
    assert body.status_code == 200, body.text
    assert body.json()["reply"] == "Still here"
    assert body.json()["trace_uuid"] is None
    assert len(_traces(agent["org_uuid"])) == 1
    assert ("max_traces", "trace", 2, 1) in _hits(agent["org_uuid"])
    # The reply counts towards the daily limit even with no trace stored.
    assert db.count_chat_replies_today(agent["uuid"]) == 2


def _save_key(org, provider, value):
    db.upsert_org_provider_key(org, provider, pk.encrypt(value), pk.masked_value(provider, value), None)


def test_built_agent_uses_the_workspace_key(client):
    _, agent, token = _chat_agent(client)
    _save_key(agent["org_uuid"], "openrouter", "sk-org")
    with patch.object(public, "_llm_reply", _llm()) as llm:
        _send(client, token)
    assert llm.call_args.kwargs["api_key"] == "sk-org"


def test_built_agent_uses_the_server_key_when_the_workspace_has_none(client):
    _, _, token = _chat_agent(client)
    with patch.object(public, "_llm_reply", _llm()) as llm:
        _send(client, token)
    assert llm.call_args.kwargs["api_key"] == "sk-server"


@pytest.mark.parametrize("setting, rounds", [(None, 5), ("2", 2), ("0", 1), ("lots", 5)])
def test_built_agent_tool_rounds_come_from_the_setting(client, monkeypatch, setting, rounds):
    if setting is None:
        monkeypatch.delenv("CHAT_MAX_TOOL_ROUNDS", raising=False)
    else:
        monkeypatch.setenv("CHAT_MAX_TOOL_ROUNDS", setting)
    _, _, token = _chat_agent(client)
    with patch.object(public, "_llm_reply", _llm()) as llm:
        _send(client, token)
    assert llm.call_args.kwargs["max_tool_rounds"] == rounds


def test_workspace_keys_without_this_provider_make_the_agent_unavailable(client):
    _, agent, token = _chat_agent(client)
    _save_key(agent["org_uuid"], "openai", "sk-openai")
    with patch.object(public, "_llm_reply", _llm()) as llm:
        refused = _send(client, token)
    assert refused.status_code == 503
    assert refused.json()["detail"] == {"error": "agent_unavailable"}
    llm.assert_not_called()


def test_agent_failure_stores_nothing_and_is_reported(client):
    _, agent, token = _chat_agent(client)
    with patch.object(public, "_llm_reply", AsyncMock(side_effect=RuntimeError("boom"))), patch.object(
        public, "capture_exception_to_sentry"
    ) as sentry:
        refused = _send(client, token)
    assert refused.status_code == 502
    assert refused.json()["detail"] == {"error": "agent_failed"}
    sentry.assert_called_once()
    assert _traces(agent["org_uuid"]) == []
    # The reply it never gave is not counted against the day.
    assert db.count_chat_replies_today(agent["uuid"]) == 0


def test_end_call_ends_the_chat_and_is_not_stored(client):
    _, agent, token = _chat_agent(client)
    calls = [{"id": "e", "tool": "end_call", "arguments": {}, "output": {"status": "received"}}]
    with patch.object(public, "_llm_reply", _llm("", calls)):
        body = _send(client, token).json()
    assert body["ended"] is True and body["reply"] is None
    assert [c["tool"] for c in body["tool_calls"]] == ["end_call"]
    assert db.get_trace(agent["org_uuid"], body["trace_uuid"])["output"]["tool_calls"] == []


def _connection_mock(call):
    instance = MagicMock()
    instance.call = call
    return MagicMock(return_value=instance)


def test_connection_agent_is_called_with_the_eval_header(client):
    _, agent, token = _chat_agent(client, config=CONNECTION)
    reply = {"response": "From you", "tool_calls": [{"tool": "lookup", "arguments": {"q": 1}}]}
    connection = _connection_mock(AsyncMock(return_value=reply))
    with patch.object(public, "public_url_error", return_value=None), patch.object(
        public, "TextAgentConnection", connection
    ):
        body = _send(client, token, "Hi").json()

    kwargs = connection.call_args.kwargs
    assert kwargs["url"] == "https://example.com/a"
    assert kwargs["headers"] == {"X-Key": "k", "X-Calibrate-Eval": "1"}
    assert kwargs["type"] == "conversation"
    connection.return_value.call.assert_awaited_once_with([{"role": "user", "content": "Hi"}])
    assert body["reply"] == "From you"
    assert body["tool_calls"] == [{"tool": "lookup", "arguments": {"q": 1}, "output": None}]
    assert db.get_trace(agent["org_uuid"], body["trace_uuid"])["output"] == {
        "response": "From you",
        "tool_calls": [{"tool": "lookup", "arguments": {"q": 1}}],
    }


def test_connection_agent_is_sent_plain_turns_without_its_tool_calls(client):
    _, agent, token = _chat_agent(client, config=CONNECTION)
    first = {"response": "Looked it up", "tool_calls": [{"tool": "lookup", "arguments": {"q": 1}, "output": {"n": 2}}]}
    connection = _connection_mock(AsyncMock(side_effect=[first, {"response": "Bye", "tool_calls": []}]))
    with patch.object(public, "public_url_error", return_value=None), patch.object(
        public, "TextAgentConnection", connection
    ):
        first_reply = _send(client, token, "Find it").json()
        _send(client, token, "Thanks", first_reply["conversation_id"], history=_turns(("Find it", first_reply)))

    assert connection.return_value.call.await_args_list[1].args[0] == [
        {"role": "user", "content": "Find it"},
        {"role": "assistant", "content": "Looked it up"},
        {"role": "user", "content": "Thanks"},
    ]


def test_connection_agent_that_times_out_fails_and_stores_nothing(client, monkeypatch):
    _, agent, token = _chat_agent(client, config=CONNECTION)
    monkeypatch.setattr(public, "CHAT_REPLY_TIMEOUT_SECONDS", 0.05)

    async def slow(messages):
        await asyncio.sleep(1)

    with patch.object(public, "public_url_error", return_value=None), patch.object(
        public, "TextAgentConnection", _connection_mock(slow)
    ), patch.object(public, "capture_exception_to_sentry"):
        refused = _send(client, token)
    assert refused.status_code == 502
    assert refused.json()["detail"] == {"error": "agent_failed"}
    assert _traces(agent["org_uuid"]) == []


def test_connection_agent_with_an_unsafe_url_is_not_called(client):
    _, _, token = _chat_agent(client, config=CONNECTION)
    connection = _connection_mock(AsyncMock())
    with patch.object(public, "public_url_error", return_value="must not point to localhost"), patch.object(
        public, "TextAgentConnection", connection
    ), patch.object(public, "capture_exception_to_sentry"):
        assert _send(client, token).status_code == 502
    connection.assert_not_called()


@pytest.mark.parametrize("config", [None, CONNECTION])
def test_fake_mode_replies_without_calling_anything(client, monkeypatch, config):
    from testing.fake_calibrate_agent import FAKE_RESPONSE

    monkeypatch.setenv("FAKE_AI_PROVIDERS", "1")
    _, _, token = _chat_agent(client, config=config)
    with patch.object(public, "_llm_reply") as llm, patch.object(public, "TextAgentConnection") as connection:
        body = _send(client, token).json()
    assert body["reply"] == FAKE_RESPONSE and body["tool_calls"] == []
    llm.assert_not_called()
    connection.assert_not_called()


def test_workspace_comes_from_the_link_not_the_headers(client):
    _, agent, token = _chat_agent(client)
    other = _signup(client)
    other_org = client.get("/organizations", headers=other).json()[0]["uuid"]
    with patch.object(public, "_llm_reply", _llm()):
        body = _send(client, token, headers={**other, "X-Org-UUID": other_org}).json()
    assert db.get_trace(agent["org_uuid"], body["trace_uuid"])["source"] == "chat"
    assert _traces(other_org) == []


def _feedback(client, token, trace_uuid, is_correct, comment=None, headers=None):
    return client.put(
        f"/public/agent-chat/{token}/traces/{trace_uuid}/feedback",
        json={"is_correct": is_correct, "comment": comment},
        headers=headers or {},
    )


def test_feedback_is_set_and_cleared(client):
    _, agent, token = _chat_agent(client)
    with patch.object(public, "_llm_reply", _llm()):
        trace_uuid = _send(client, token).json()["trace_uuid"]

    assert _feedback(client, token, trace_uuid, False, "Wrong date").json() == {
        "is_correct": False,
        "comment": "Wrong date",
    }
    stored = db.get_trace(agent["org_uuid"], trace_uuid)["feedback"]
    assert stored["is_correct"] is False and stored["comment"] == "Wrong date"

    assert _feedback(client, token, trace_uuid, True).json() == {"is_correct": True, "comment": None}
    assert db.get_trace(agent["org_uuid"], trace_uuid)["feedback"]["is_correct"] is True

    assert _feedback(client, token, trace_uuid, None, "ignored").json() == {"is_correct": None, "comment": None}
    assert db.get_trace(agent["org_uuid"], trace_uuid)["feedback"] is None
    assert _feedback(client, token, trace_uuid, True, "x" * 2001).status_code == 422


def test_feedback_is_refused_for_another_agents_trace_and_an_api_trace(client):
    h, agent, token = _chat_agent(client)
    _, _, other_token = _chat_agent(client, h=h)
    with patch.object(public, "_llm_reply", _llm()):
        other_trace = _send(client, other_token).json()["trace_uuid"]
    api_trace = db.create_trace_with_eval_run(
        org_uuid=agent["org_uuid"],
        agent=agent,
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello", "tool_calls": []},
        max_scored_traces=None,
        batch_size=20,
    )["uuid"]

    for trace_uuid in (other_trace, api_trace, str(uuid.uuid4())):
        refused = _feedback(client, token, trace_uuid, True)
        assert refused.status_code == 404
        assert refused.json()["detail"] == {"error": "not_found"}
    assert db.get_trace(agent["org_uuid"], other_trace)["feedback"] is None
    assert db.get_trace(agent["org_uuid"], api_trace)["feedback"] is None


def test_llm_reply_hands_everything_to_calibrates_agent_reply():
    import types

    chat_module = types.SimpleNamespace(agent_reply=AsyncMock(return_value={"reply": "ok"}))
    with patch.dict("sys.modules", {"calibrate_agent.chat": chat_module}):
        result = asyncio.run(public._llm_reply(messages=[], model="m", api_key="k"))
    assert result == {"reply": "ok"}
    chat_module.agent_reply.assert_awaited_once_with(messages=[], model="m", api_key="k")
