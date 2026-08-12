"""Cross-org isolation smoke tests.

Verifies the PR-2 access boundary: two users in *different* orgs cannot see
each other's entities, while two members of the *same* org both see every
entity the org owns. One smoke test per major entity (agent, tool, test,
simulation, dataset, persona, scenario, annotator, annotation task) is
sufficient — the per-entity 404 behaviour is exercised exhaustively in
the existing router tests; this file pins the cross-tenancy contract."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import db


@pytest.fixture(scope="module")
def app():
    import main as main_mod

    return main_mod.app


@pytest.fixture(scope="module")
def client(app):
    with patch("main.recover_pending_jobs"):
        with TestClient(app) as c:
            yield c


def _signup(client: TestClient, email_prefix: str = "iso") -> dict:
    suffix = uuid.uuid4().hex[:8]
    email = f"{email_prefix}-{suffix}@example.com"
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "I",
            "last_name": "S",
            "email": email,
            "password": "passw0rd",
        },
    ).json()
    return {
        "user_uuid": body["user"]["uuid"],
        "email": email,
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
        "access_token": body["access_token"],
    }


def _invite_to_org(client, owner_auth, org_uuid: str, invitee_email: str):
    """Owner adds another user to their org as admin."""
    resp = client.post(
        f"/organizations/{org_uuid}/members",
        json={"email": invitee_email},
        headers=owner_auth["headers"],
    )
    assert resp.status_code == 201, resp.text


def _org_header(org_uuid: str) -> dict:
    return {"X-Org-UUID": org_uuid}


# ---------------------------------------------------------------------------
# Cross-org: 404 on each major entity
# ---------------------------------------------------------------------------


def test_cross_org_cannot_see_agents(client):
    a = _signup(client)
    b = _signup(client)
    create = client.post(
        "/agents",
        json={"name": f"agent-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=a["headers"],
    )
    a_uuid = create.json()["uuid"]

    # B can't see A's agent
    assert client.get(f"/agents/{a_uuid}", headers=b["headers"]).status_code == 404
    assert (
        client.put(
            f"/agents/{a_uuid}", json={"name": "x"}, headers=b["headers"]
        ).status_code
        == 404
    )
    assert client.delete(f"/agents/{a_uuid}", headers=b["headers"]).status_code == 404

    # B's list doesn't contain A's agent
    b_list = client.get("/agents", headers=b["headers"]).json()["items"]
    assert all(item["uuid"] != a_uuid for item in b_list)


def test_cross_org_cannot_link_tools_or_see_agent_tool_graph(client):
    """The /agent-tools router gates every endpoint on the caller's org.
    Cross-org link attempts return 404; cross-org reads of the agent's tools
    or a tool's agents return 404; the list endpoint only shows the caller's
    own links."""
    a = _signup(client)
    b = _signup(client)

    agent_a = client.post(
        "/agents",
        json={"name": f"agent-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=a["headers"],
    ).json()
    tool_a = client.post(
        "/tools",
        json={
            "name": f"tool-{uuid.uuid4().hex[:6]}",
            "description": "d",
            "config": {"type": "structured_output", "parameters": []},
        },
        headers=a["headers"],
    ).json()
    tool_b = client.post(
        "/tools",
        json={
            "name": f"tool-b-{uuid.uuid4().hex[:6]}",
            "description": "d",
            "config": {"type": "structured_output", "parameters": []},
        },
        headers=b["headers"],
    ).json()

    # a links own agent + own tool — succeeds.
    link_resp = client.post(
        "/agent-tools",
        json={"agent_uuid": agent_a["uuid"], "tool_uuids": [tool_a["uuid"]]},
        headers=a["headers"],
    )
    assert link_resp.status_code == 200

    # b tries to link a's agent with b's tool — 404 on the agent.
    resp = client.post(
        "/agent-tools",
        json={"agent_uuid": agent_a["uuid"], "tool_uuids": [tool_b["uuid"]]},
        headers=b["headers"],
    )
    assert resp.status_code == 404

    # a tries to link own agent with b's tool — 404 on the tool.
    resp = client.post(
        "/agent-tools",
        json={"agent_uuid": agent_a["uuid"], "tool_uuids": [tool_b["uuid"]]},
        headers=a["headers"],
    )
    assert resp.status_code == 404

    # b can't read a's agent tools / a's tool agents.
    assert (
        client.get(
            f"/agent-tools/agent/{agent_a['uuid']}/tools", headers=b["headers"]
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/agent-tools/tool/{tool_a['uuid']}/agents", headers=b["headers"]
        ).status_code
        == 404
    )

    # b can't delete a's link.
    resp = client.request(
        "DELETE",
        "/agent-tools",
        json={"agent_uuid": agent_a["uuid"], "tool_uuid": tool_a["uuid"]},
        headers=b["headers"],
    )
    assert resp.status_code == 404

    # b's list endpoint doesn't include a's link.
    b_list = client.get("/agent-tools", headers=b["headers"]).json()
    assert all(
        not (l["agent_id"] == agent_a["uuid"] and l["tool_id"] == tool_a["uuid"])
        for l in b_list
    )


def test_cross_org_cannot_see_tools(client):
    a = _signup(client)
    b = _signup(client)
    tool = client.post(
        "/tools",
        json={
            "name": f"tool-{uuid.uuid4().hex[:6]}",
            "description": "d",
            "config": {"type": "structured_output", "parameters": []},
        },
        headers=a["headers"],
    ).json()
    assert client.get(f"/tools/{tool['uuid']}", headers=b["headers"]).status_code == 404


def test_cross_org_cannot_see_tests(client):
    a = _signup(client)
    b = _signup(client)
    test = client.post(
        "/tests",
        json={
            "name": f"test-{uuid.uuid4().hex[:6]}",
            "type": "tool_call",
            "config": {
                "history": [{"role": "user", "content": "hi"}],
                "evaluation": {
                    "type": "tool_call",
                    "tool_calls": [{"tool": "x", "accept_any_arguments": True}],
                },
            },
        },
        headers=a["headers"],
    ).json()
    assert client.get(f"/tests/{test['uuid']}", headers=b["headers"]).status_code == 404


def test_cross_org_cannot_see_personas(client):
    a = _signup(client)
    b = _signup(client)
    persona = client.post(
        "/personas",
        json={"name": f"p-{uuid.uuid4().hex[:6]}", "description": "d"},
        headers=a["headers"],
    ).json()
    assert (
        client.get(f"/personas/{persona['uuid']}", headers=b["headers"]).status_code
        == 404
    )


def test_cross_org_cannot_see_scenarios(client):
    a = _signup(client)
    b = _signup(client)
    sc = client.post(
        "/scenarios",
        json={"name": f"s-{uuid.uuid4().hex[:6]}", "description": "d"},
        headers=a["headers"],
    ).json()
    assert (
        client.get(f"/scenarios/{sc['uuid']}", headers=b["headers"]).status_code == 404
    )


def test_cross_org_cannot_see_simulations(client):
    a = _signup(client)
    b = _signup(client)
    sim = client.post(
        "/simulations",
        json={"name": f"sim-{uuid.uuid4().hex[:6]}"},
        headers=a["headers"],
    ).json()
    assert (
        client.get(f"/simulations/{sim['uuid']}", headers=b["headers"]).status_code
        == 404
    )


def test_cross_org_cannot_see_datasets(client):
    a = _signup(client)
    b = _signup(client)
    ds = client.post(
        "/datasets",
        json={"name": f"ds-{uuid.uuid4().hex[:6]}", "dataset_type": "tts"},
        headers=a["headers"],
    ).json()
    assert (
        client.get(f"/datasets/{ds['uuid']}", headers=b["headers"]).status_code == 404
    )


def test_cross_org_cannot_see_annotators(client):
    a = _signup(client)
    b = _signup(client)
    ann = client.post(
        "/annotators",
        json={"name": f"ann-{uuid.uuid4().hex[:6]}"},
        headers=a["headers"],
    ).json()
    assert (
        client.get(f"/annotators/{ann['uuid']}", headers=b["headers"]).status_code
        == 404
    )


def test_cross_org_cannot_see_annotation_tasks(client):
    a = _signup(client)
    b = _signup(client)
    task = client.post(
        "/annotation-tasks",
        json={"name": f"t-{uuid.uuid4().hex[:6]}", "type": "llm"},
        headers=a["headers"],
    ).json()
    assert (
        client.get(
            f"/annotation-tasks/{task['uuid']}", headers=b["headers"]
        ).status_code
        == 404
    )


# ---------------------------------------------------------------------------
# Same org, different users — both see all entities
# ---------------------------------------------------------------------------


def test_same_org_members_see_each_others_agents(client):
    owner = _signup(client, email_prefix="owner")
    member = _signup(client, email_prefix="member")

    # Get owner's personal org
    owner_personal_org = db.get_personal_org_for_user(owner["user_uuid"])
    org_uuid = owner_personal_org["uuid"]

    # Owner invites member to their personal org. Members must be added by
    # email; the owner sees the member's hydrated row immediately.
    _invite_to_org(client, owner, org_uuid, member["email"])

    # Owner creates an agent in their personal org (default scope, no header).
    create = client.post(
        "/agents",
        json={"name": f"shared-agent-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=owner["headers"],
    )
    assert create.status_code == 200, create.text
    a_uuid = create.json()["uuid"]

    # Member, when sending the X-Org-UUID header for the owner's org, sees it.
    member_headers = {**member["headers"], **_org_header(org_uuid)}
    listing = client.get("/agents", headers=member_headers)
    assert listing.status_code == 200
    assert any(item["uuid"] == a_uuid for item in listing.json()["items"])

    # And can fetch it by UUID.
    got = client.get(f"/agents/{a_uuid}", headers=member_headers)
    assert got.status_code == 200

    # Member sees nothing without the header (their own personal org is empty).
    own_listing = client.get("/agents", headers=member["headers"])
    assert own_listing.status_code == 200
    assert all(item["uuid"] != a_uuid for item in own_listing.json()["items"])


# ---------------------------------------------------------------------------
# 404 workspace hint (`organization_uuid`)
# ---------------------------------------------------------------------------


def test_404_hints_the_org_when_caller_is_a_member(client):
    """A member fetching an agent under the wrong X-Org-UUID gets the owning
    org back on the 404, so the frontend can switch workspace and retry."""
    owner = _signup(client, email_prefix="hint-owner")
    member = _signup(client, email_prefix="hint-member")
    org_uuid = db.get_personal_org_for_user(owner["user_uuid"])["uuid"]
    _invite_to_org(client, owner, org_uuid, member["email"])

    a_uuid = client.post(
        "/agents",
        json={"name": f"hinted-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=owner["headers"],
    ).json()["uuid"]

    # Member's active workspace is their own personal org — wrong one.
    resp = client.get(f"/agents/{a_uuid}", headers=member["headers"])
    assert resp.status_code == 404
    assert resp.json()["organization_uuid"] == org_uuid


def test_404_hides_the_org_from_non_members(client):
    a = _signup(client, email_prefix="hint-a")
    b = _signup(client, email_prefix="hint-b")
    a_uuid = client.post(
        "/agents",
        json={"name": f"private-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=a["headers"],
    ).json()["uuid"]

    resp = client.get(f"/agents/{a_uuid}", headers=b["headers"])
    assert resp.status_code == 404
    assert "organization_uuid" not in resp.json()


def test_404_for_an_unknown_uuid_carries_no_hint(client):
    a = _signup(client, email_prefix="hint-missing")
    resp = client.get(f"/agents/{uuid.uuid4()}", headers=a["headers"])
    assert resp.status_code == 404
    assert "organization_uuid" not in resp.json()


def test_404_hint_covers_jobs_and_annotation_tasks(client):
    """The hint is not agent-specific: it works off any uuid in the path,
    including tables that reach their org through a join."""
    owner = _signup(client, email_prefix="hint-job-owner")
    member = _signup(client, email_prefix="hint-job-member")
    org_uuid = db.get_personal_org_for_user(owner["user_uuid"])["uuid"]
    _invite_to_org(client, owner, org_uuid, member["email"])

    task_uuid = client.post(
        "/annotation-tasks",
        json={"name": f"hint-task-{uuid.uuid4().hex[:6]}", "type": "llm"},
        headers=owner["headers"],
    ).json()["uuid"]
    resp = client.get(f"/annotation-tasks/{task_uuid}", headers=member["headers"])
    assert resp.status_code == 404
    assert resp.json()["organization_uuid"] == org_uuid

    # agent_test_jobs reaches its org via agents.
    agent_uuid = client.post(
        "/agents",
        json={"name": f"hint-agent-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=owner["headers"],
    ).json()["uuid"]
    job_uuid = db.create_agent_test_job(agent_uuid, "llm-unit-test")
    resp = client.get(f"/agent-tests/run/{job_uuid}", headers=member["headers"])
    assert resp.status_code == 404
    assert resp.json()["organization_uuid"] == org_uuid


def test_cross_org_cannot_see_evaluator_agreement_trend(client):
    a = _signup(client, email_prefix="trend-a")
    b = _signup(client, email_prefix="trend-b")
    ev = client.post(
        "/evaluators",
        json={
            "name": f"ev-{uuid.uuid4().hex[:6]}",
            "description": "d",
            "evaluator_type": "llm",
            "data_type": "text",
            "kind": "single",
            "output_type": "binary",
            "version": {
                "judge_model": "openai/gpt-4",
                "system_prompt": "Judge it",
                "output_config": {
                    "scale": [
                        {"value": True, "name": "Pass"},
                        {"value": False, "name": "Fail"},
                    ]
                },
            },
        },
        headers=a["headers"],
    )
    assert ev.status_code == 200, ev.text
    ev_uuid = ev.json()["uuid"]

    assert (
        client.get(
            f"/annotation-agreement/evaluator/{ev_uuid}/trend", headers=a["headers"]
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/annotation-agreement/evaluator/{ev_uuid}/trend", headers=b["headers"]
        ).status_code
        == 404
    )


def test_cross_org_cannot_list_agents_for_a_test(client):
    a = _signup(client, email_prefix="test-agents-a")
    b = _signup(client, email_prefix="test-agents-b")
    test = client.post(
        "/tests",
        json={
            "name": f"test-{uuid.uuid4().hex[:6]}",
            "type": "tool_call",
            "config": {
                "history": [{"role": "user", "content": "hi"}],
                "evaluation": {
                    "type": "tool_call",
                    "tool_calls": [{"tool": "x", "accept_any_arguments": True}],
                },
            },
        },
        headers=a["headers"],
    ).json()
    url = f"/agent-tests/test/{test['uuid']}/agents"
    assert client.get(url, headers=a["headers"]).status_code == 200
    assert client.get(url, headers=b["headers"]).status_code == 404


def test_404_carries_no_hint_for_the_workspace_already_active(client):
    """A parent uuid in the path resolves to the caller's own workspace, so
    it must not produce a hint the frontend would loop on."""
    a = _signup(client, email_prefix="hint-self")
    task_uuid = client.post(
        "/annotation-tasks",
        json={"name": f"self-task-{uuid.uuid4().hex[:6]}", "type": "llm"},
        headers=a["headers"],
    ).json()["uuid"]

    resp = client.get(
        f"/annotation-tasks/{task_uuid}/items/{uuid.uuid4()}", headers=a["headers"]
    )
    assert resp.status_code == 404
    assert "organization_uuid" not in resp.json()


def test_404_carries_no_hint_for_api_key_callers(client):
    """An API key only ever works in the org it was minted in, so telling its
    holder about a different workspace is useless and must not happen."""
    owner = _signup(client, email_prefix="hint-key-owner")
    other = _signup(client, email_prefix="hint-key-other")
    owner_org = db.get_personal_org_for_user(owner["user_uuid"])["uuid"]
    other_org = db.get_personal_org_for_user(other["user_uuid"])["uuid"]
    # The key holder is a member of both orgs, so only the API-key rule can
    # suppress the hint here.
    _invite_to_org(client, other, other_org, owner["email"])

    raw_key = client.post(
        "/api-keys",
        json={"name": f"key-{uuid.uuid4().hex[:6]}"},
        headers={**owner["headers"], **_org_header(owner_org)},
    ).json()["key"]

    agent_uuid = client.post(
        "/agents",
        json={"name": f"keyed-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers={**other["headers"], **_org_header(other_org)},
    ).json()["uuid"]

    resp = client.get(f"/agents/{agent_uuid}", headers={"X-API-Key": raw_key})
    assert resp.status_code == 404
    assert "organization_uuid" not in resp.json()

    # Same request as a JWT caller does get the hint, so the 404 above is the
    # API-key rule at work and not a missing link.
    jwt_resp = client.get(
        f"/agents/{agent_uuid}", headers={**owner["headers"], **_org_header(owner_org)}
    )
    assert jwt_resp.json()["organization_uuid"] == other_org


def test_404_stays_a_404_when_the_hint_lookup_fails(client):
    """The hint is resolved inside an exception handler, where an uncaught
    error would escape as a bare 500 instead of the 404 the route raised."""
    a = _signup(client, email_prefix="hint-boom")
    with patch(
        "main.find_other_member_org_for_resources",
        side_effect=RuntimeError("database is locked"),
    ):
        resp = client.get(f"/agents/{uuid.uuid4()}", headers=a["headers"])
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Agent not found"}
