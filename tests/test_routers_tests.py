"""Integration tests for /tests endpoints."""

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
            "first_name": "Test",
            "last_name": "User",
            "email": f"test-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _raw_key(client, h, name="ci"):
    return client.post("/api-keys", json={"name": name}, headers=h).json()["key"]


def _create_test(client, headers, name=None):
    r = client.post(
        "/tests",
        json={"name": name or f"t-{uuid.uuid4().hex[:6]}", "type": "response", "config": {}},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    return r.json()["uuid"]


def test_create_test_with_api_key(client):
    """POST /tests must accept an API key — currently JWT-only so this should fail with 401."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    r = client.post(
        "/tests",
        json={"name": f"t-{uuid.uuid4().hex[:6]}", "type": "response", "config": {}},
        headers={"X-API-Key": key},
    )
    assert r.status_code == 200


def test_list_tests_with_api_key(client):
    """GET /tests accepts an X-API-Key and lists the caller's org tests."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    t_uuid = _create_test(client, {"X-API-Key": key})
    r = client.get("/tests", headers={"X-API-Key": key})
    assert r.status_code == 200, r.text
    assert t_uuid in {t["uuid"] for t in r.json()["items"]}


def test_list_tests_returns_trimmed_shape(client):
    """GET /tests returns the trimmed list shape: uuid/name/type + only
    config.description survives, while the heavy `evaluators` list and the
    `config.history`/`config.evaluation` blocks are dropped from list items."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    # An `llm` evaluator to link, so a full test would carry a non-empty
    # `evaluators[]` — proving the list shape drops it.
    evaluators = client.get("/evaluators", headers=jwt).json()["items"]
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")
    name = f"t-trim-{uuid.uuid4().hex[:6]}"
    created = client.post(
        "/tests",
        json={
            "name": name,
            "type": "response",
            "config": {
                "description": "search me",
                "history": [{"role": "user", "content": "hi"}],
                "evaluation": {"type": "response"},
                "settings": {"language": "en"},
            },
            "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
        },
        headers={"X-API-Key": key},
    )
    assert created.status_code == 200, created.text
    t_uuid = created.json()["uuid"]

    items = client.get("/tests", headers={"X-API-Key": key}).json()["items"]
    item = next(t for t in items if t["uuid"] == t_uuid)
    # Trimmed shape: no evaluator hydration, no heavy config blocks.
    assert "evaluators" not in item
    assert item["config"] == {"description": "search me"}
    assert "history" not in item["config"]
    assert "evaluation" not in item["config"]
    assert item["name"] == name
    assert item["type"] == "response"


def test_list_tests_never_ships_heavy_config_blocks(client):
    """The slim list summary json_extracts only `config.description`; the heavy
    `config.history`/`evaluation`/`settings` blocks (conversation transcripts,
    judge config) must never reach the wire. Sentinel strings stuffed into those
    blocks must be absent from the whole `GET /tests` response body."""
    import json as _json

    jwt = _signup(client)
    key = _raw_key(client, jwt)
    hist_sentinel = f"HIST-{uuid.uuid4().hex}"
    eval_sentinel = f"EVAL-{uuid.uuid4().hex}"
    settings_sentinel = f"SET-{uuid.uuid4().hex}"
    name = f"t-heavy-{uuid.uuid4().hex[:6]}"
    created = client.post(
        "/tests",
        json={
            "name": name,
            "type": "response",
            "config": {
                "description": "keep me",
                "history": [{"role": "user", "content": hist_sentinel}],
                "evaluation": {"type": "response", "note": eval_sentinel},
                "settings": {"language": settings_sentinel},
            },
        },
        headers={"X-API-Key": key},
    )
    assert created.status_code == 200, created.text
    t_uuid = created.json()["uuid"]

    r = client.get("/tests", headers={"X-API-Key": key})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"items", "total", "limit", "offset"}
    item = next(t for t in body["items"] if t["uuid"] == t_uuid)
    assert item["name"] == name
    assert item["type"] == "response"
    assert item["config"] == {"description": "keep me"}
    assert set(item["config"]) == {"description"}

    dumped = _json.dumps(body)
    assert hist_sentinel not in dumped
    assert eval_sentinel not in dumped
    assert settings_sentinel not in dumped


def test_list_tests_null_description(client):
    """A test with no `config.description` still lists 200 with description=null."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    name = f"t-nodesc-{uuid.uuid4().hex[:6]}"
    created = client.post(
        "/tests",
        json={
            "name": name,
            "type": "response",
            "config": {"history": [{"role": "user", "content": "hi"}]},
        },
        headers={"X-API-Key": key},
    )
    assert created.status_code == 200, created.text
    t_uuid = created.json()["uuid"]

    r = client.get("/tests", headers={"X-API-Key": key})
    assert r.status_code == 200, r.text
    item = next(t for t in r.json()["items"] if t["uuid"] == t_uuid)
    assert item["config"] == {"description": None}


def test_list_tests_q_search_and_pagination(client):
    """`?q=` filters list items by name; `?limit=/?offset=` slice the envelope
    while `total` stays the pre-slice count of the filtered set."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    tag = uuid.uuid4().hex[:8]
    names = [f"srch-{tag}-{i}" for i in range(3)]
    for n in names:
        _create_test(client, {"X-API-Key": key}, name=n)
    # A decoy that must not match the search tag.
    _create_test(client, {"X-API-Key": key}, name=f"other-{uuid.uuid4().hex[:6]}")

    hit = client.get("/tests", params={"q": tag}, headers={"X-API-Key": key})
    assert hit.status_code == 200, hit.text
    body = hit.json()
    assert body["total"] == 3
    assert {t["name"] for t in body["items"]} == set(names)

    page = client.get(
        "/tests", params={"q": tag, "limit": 2, "offset": 0}, headers={"X-API-Key": key}
    ).json()
    assert page["total"] == 3
    assert len(page["items"]) == 2
    page2 = client.get(
        "/tests", params={"q": tag, "limit": 2, "offset": 2}, headers={"X-API-Key": key}
    ).json()
    assert page2["total"] == 3
    assert len(page2["items"]) == 1


def test_get_test_with_api_key(client):
    """GET /tests/{uuid} accepts an X-API-Key and returns the full test shape."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    evaluators = client.get("/evaluators", headers=jwt).json()["items"]
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")
    created = client.post(
        "/tests",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "response",
            "config": {
                "description": "d",
                "history": [{"role": "user", "content": "hi"}],
                "evaluation": {"type": "response"},
            },
            "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
        },
        headers={"X-API-Key": key},
    )
    assert created.status_code == 200, created.text
    t_uuid = created.json()["uuid"]
    r = client.get(f"/tests/{t_uuid}", headers={"X-API-Key": key})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["uuid"] == t_uuid
    # Full detail shape keeps evaluators + the whole config.
    assert len(body["evaluators"]) == 1
    assert body["config"]["history"] == [{"role": "user", "content": "hi"}]
    assert body["config"]["evaluation"] == {"type": "response"}


def test_update_test_with_api_key(client):
    """PUT /tests/{uuid} accepts an X-API-Key."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    t_uuid = _create_test(client, {"X-API-Key": key})
    new_name = f"t-upd-{uuid.uuid4().hex[:6]}"
    r = client.put(
        f"/tests/{t_uuid}", json={"name": new_name}, headers={"X-API-Key": key}
    )
    assert r.status_code == 200, r.text
    assert r.json()["name"] == new_name


def test_bulk_create_tests_with_api_key(client):
    """POST /tests/bulk accepts an X-API-Key."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    evaluators = client.get("/evaluators", headers=jwt).json()["items"]
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")
    ev_ref = [{"evaluator_uuid": llm_ev["uuid"]}]
    r = client.post(
        "/tests/bulk",
        json={
            "type": "response",
            "tests": [
                {
                    "name": f"bulk-{uuid.uuid4().hex[:6]}",
                    "conversation_history": [{"role": "user", "content": "hi"}],
                    "evaluators": ev_ref,
                },
                {
                    "name": f"bulk-{uuid.uuid4().hex[:6]}",
                    "conversation_history": [{"role": "user", "content": "yo"}],
                    "evaluators": ev_ref,
                },
            ],
        },
        headers={"X-API-Key": key},
    )
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 2


def test_bulk_create_tests_preserves_inputs(client):
    """POST /tests/bulk keeps per-test inputs in config."""
    jwt = _signup(client)
    evaluators = client.get("/evaluators", headers=jwt).json()["items"]
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")
    r = client.post(
        "/tests/bulk",
        json={
            "type": "response",
            "tests": [
                {
                    "name": f"bulk-{uuid.uuid4().hex[:6]}",
                    "conversation_history": [{"role": "user", "content": "hi"}],
                    "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
                    "inputs": {"condition_area": "cardiology"},
                }
            ],
        },
        headers=jwt,
    )
    assert r.status_code == 200, r.text
    test_uuid = r.json()["uuids"][0]
    config = client.get(f"/tests/{test_uuid}", headers=jwt).json()["config"]
    assert config["inputs"] == {"condition_area": "cardiology"}


def test_bulk_create_rejects_system_role(client):
    """`system` is not a valid conversation_history role — the agent's system
    prompt lives in its config, not the history. Only user/assistant/tool."""
    jwt = _signup(client)
    r = client.post(
        "/tests/bulk",
        json={
            "type": "response",
            "tests": [
                {
                    "name": f"bulk-{uuid.uuid4().hex[:6]}",
                    "conversation_history": [
                        {"role": "system", "content": "you are helpful"},
                        {"role": "user", "content": "hi"},
                    ],
                    "evaluators": [],
                }
            ],
        },
        headers=jwt,
    )
    assert r.status_code == 422, r.text


def test_update_conversation_test_rejects_clearing_evaluators(client):
    """A conversation test must keep >=1 evaluator: PUT with an empty
    `evaluators` list is 400, so the description's 'clears them' promise
    correctly excludes conversation tests."""
    jwt = _signup(client)
    # Create a conversation evaluator (its first version is set live on create),
    # so the link doesn't depend on seeded-evaluator ordering/state.
    ev = client.post(
        "/evaluators",
        json={
            "name": f"conv-ev-{uuid.uuid4().hex[:6]}",
            "evaluator_type": "conversation",
            "version": {
                "judge_model": "openai/gpt-4o-mini",
                "system_prompt": "Judge the conversation.",
            },
        },
        headers=jwt,
    )
    assert ev.status_code == 200, ev.text
    conv_ev_uuid = ev.json()["uuid"]
    created = client.post(
        "/tests",
        json={
            "name": f"conv-{uuid.uuid4().hex[:6]}",
            "type": "conversation",
            "evaluators": [{"evaluator_uuid": conv_ev_uuid}],
        },
        headers=jwt,
    )
    assert created.status_code == 200, created.text
    t_uuid = created.json()["uuid"]

    cleared = client.put(f"/tests/{t_uuid}", json={"evaluators": []}, headers=jwt)
    assert cleared.status_code == 400, cleared.text
    assert "at least one evaluator" in cleared.text


def test_create_test_invalid_api_key(client):
    """POST /tests with a bogus key must 401."""
    r = client.post(
        "/tests",
        json={"name": f"t-{uuid.uuid4().hex[:6]}", "type": "response", "config": {}},
        headers={"X-API-Key": "bad_key"},
    )
    assert r.status_code == 401


def test_get_test_wrong_org_api_key(client):
    """A key from another org must not read a test — 403, with no owning org named.

    A key is bound to one org, so there is no workspace for its holder to switch
    to; the response must not carry organization_uuid.
    """
    jwt_a = _signup(client)
    t_uuid = _create_test(client, jwt_a)

    jwt_b = _signup(client)
    key_b = _raw_key(client, jwt_b)
    r = client.get(f"/tests/{t_uuid}", headers={"X-API-Key": key_b})
    assert r.status_code == 403
    assert "organization_uuid" not in r.json()


def test_create_test_bearer_sk_key(client):
    """POST /tests accepts the key via Authorization: Bearer sk_…."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    r = client.post(
        "/tests",
        json={"name": f"t-{uuid.uuid4().hex[:6]}", "type": "response", "config": {}},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 200, r.text


def _create_llm_general_evaluator(client, headers):
    ev = client.post(
        "/evaluators",
        json={
            "name": f"gen-ev-{uuid.uuid4().hex[:6]}",
            "evaluator_type": "llm-general",
            "version": {
                "judge_model": "openai/gpt-4o-mini",
                "system_prompt": "Judge the input/output pair.",
            },
        },
        headers=headers,
    )
    assert ev.status_code == 200, ev.text
    return ev.json()["uuid"]


def test_create_general_test_succeeds(client):
    """POST /tests with type=general and a linked llm-general evaluator succeeds."""
    jwt = _signup(client)
    gen_ev_uuid = _create_llm_general_evaluator(client, jwt)
    r = client.post(
        "/tests",
        json={
            "name": f"gen-{uuid.uuid4().hex[:6]}",
            "type": "general",
            "config": {
                "input": "Summarize this article.",
                "evaluation": {"type": "general"},
            },
            "evaluators": [{"evaluator_uuid": gen_ev_uuid}],
        },
        headers=jwt,
    )
    assert r.status_code == 200, r.text


def test_update_general_test_cannot_remove_all_evaluators(client):
    """PUT /tests/{uuid} on a general test must keep the same guard as
    conversation tests: an update that clears every evaluator is rejected,
    since general tests have no fallback judge."""
    jwt = _signup(client)
    gen_ev_uuid = _create_llm_general_evaluator(client, jwt)
    create = client.post(
        "/tests",
        json={
            "name": f"gen-{uuid.uuid4().hex[:6]}",
            "type": "general",
            "config": {
                "input": "Summarize this article.",
                "evaluation": {"type": "general"},
            },
            "evaluators": [{"evaluator_uuid": gen_ev_uuid}],
        },
        headers=jwt,
    )
    assert create.status_code == 200, create.text
    test_uuid = create.json()["uuid"]

    r = client.put(
        f"/tests/{test_uuid}",
        json={"evaluators": []},
        headers=jwt,
    )
    assert r.status_code == 400, r.text
    assert "at least one evaluator" in r.text


def test_create_general_test_rejects_llm_evaluator(client):
    """POST /tests with type=general and a linked `llm` (not `llm-general`)
    evaluator fails 400 — evaluator_type mismatch."""
    jwt = _signup(client)
    evaluators = client.get("/evaluators", headers=jwt).json()["items"]
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")
    r = client.post(
        "/tests",
        json={
            "name": f"gen-{uuid.uuid4().hex[:6]}",
            "type": "general",
            "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
        },
        headers=jwt,
    )
    assert r.status_code == 400, r.text
    assert "only accept 'llm-general' evaluators" in r.text


def test_create_general_test_requires_evaluator(client):
    """POST /tests with type=general and no evaluators fails 400."""
    jwt = _signup(client)
    r = client.post(
        "/tests",
        json={"name": f"gen-{uuid.uuid4().hex[:6]}", "type": "general"},
        headers=jwt,
    )
    assert r.status_code == 400, r.text
    assert "General tests require at least one evaluator" in r.text


def test_bulk_create_general_tests_succeeds(client):
    """POST /tests/bulk with type=general: an item with `input` set and no
    `conversation_history`, with a linked llm-general evaluator, succeeds and
    the created test's config carries `input`, not `history`."""
    jwt = _signup(client)
    gen_ev_uuid = _create_llm_general_evaluator(client, jwt)
    r = client.post(
        "/tests/bulk",
        json={
            "type": "general",
            "tests": [
                {
                    "name": f"bulk-gen-{uuid.uuid4().hex[:6]}",
                    "input": "Summarize this article.",
                    "evaluators": [{"evaluator_uuid": gen_ev_uuid}],
                }
            ],
        },
        headers=jwt,
    )
    assert r.status_code == 200, r.text
    test_uuid = r.json()["uuids"][0]
    config = client.get(f"/tests/{test_uuid}", headers=jwt).json()["config"]
    assert config["input"] == "Summarize this article."
    assert "history" not in config


def test_bulk_create_general_tests_requires_input(client):
    """POST /tests/bulk with type=general and a batch item missing `input`
    fails validation — the pydantic model_validator raises a ValueError,
    which FastAPI surfaces as 422."""
    jwt = _signup(client)
    gen_ev_uuid = _create_llm_general_evaluator(client, jwt)
    r = client.post(
        "/tests/bulk",
        json={
            "type": "general",
            "tests": [
                {
                    "name": f"bulk-gen-{uuid.uuid4().hex[:6]}",
                    "evaluators": [{"evaluator_uuid": gen_ev_uuid}],
                }
            ],
        },
        headers=jwt,
    )
    assert r.status_code == 422, r.text


def test_bulk_create_general_tests_rejects_conversation_agent(client):
    """POST /tests/bulk with type=general and agent_uuids pointing at an agent
    whose interaction_type is `conversation` (the DB default) rejects the whole
    batch with 400 and creates no tests."""
    jwt = _signup(client)
    gen_ev_uuid = _create_llm_general_evaluator(client, jwt)
    agent = client.post(
        "/agents",
        json={
            "name": f"agent-{uuid.uuid4().hex[:6]}",
            "type": "connection",
            "config": {"agent_url": "https://example.com/agent"},
        },
        headers=jwt,
    )
    assert agent.status_code == 200, agent.text
    agent_uuid = agent.json()["uuid"]

    test_name = f"bulk-gen-{uuid.uuid4().hex[:6]}"
    r = client.post(
        "/tests/bulk",
        json={
            "type": "general",
            "tests": [
                {
                    "name": test_name,
                    "input": "Summarize this article.",
                    "evaluators": [{"evaluator_uuid": gen_ev_uuid}],
                }
            ],
            "agent_uuids": [agent_uuid],
        },
        headers=jwt,
    )
    assert r.status_code == 400, r.text
    assert "cannot link general tests to it" in r.text

    tests = client.get("/tests", params={"q": test_name}, headers=jwt).json()["items"]
    assert tests == []


def test_general_test_type_immutable(client):
    """A test's `type` is immutable: PUT cannot switch a `general` test to
    another type, or another type to `general`."""
    jwt = _signup(client)
    gen_ev_uuid = _create_llm_general_evaluator(client, jwt)
    created = client.post(
        "/tests",
        json={
            "name": f"gen-{uuid.uuid4().hex[:6]}",
            "type": "general",
            "config": {"input": "hi", "evaluation": {"type": "general"}},
            "evaluators": [{"evaluator_uuid": gen_ev_uuid}],
        },
        headers=jwt,
    )
    assert created.status_code == 200, created.text
    t_uuid = created.json()["uuid"]

    r = client.put(f"/tests/{t_uuid}", json={"type": "response"}, headers=jwt)
    assert r.status_code == 400, r.text
    assert "Test type is immutable" in r.text

    response_uuid = _create_test(client, jwt)
    r2 = client.put(f"/tests/{response_uuid}", json={"type": "general"}, headers=jwt)
    assert r2.status_code == 400, r2.text
    assert "Test type is immutable" in r2.text
