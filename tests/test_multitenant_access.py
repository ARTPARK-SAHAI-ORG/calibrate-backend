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
    b_list = client.get("/agents", headers=b["headers"]).json()
    assert all(item["uuid"] != a_uuid for item in b_list)


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
    assert any(item["uuid"] == a_uuid for item in listing.json())

    # And can fetch it by UUID.
    got = client.get(f"/agents/{a_uuid}", headers=member_headers)
    assert got.status_code == 200

    # Member sees nothing without the header (their own personal org is empty).
    own_listing = client.get("/agents", headers=member["headers"])
    assert own_listing.status_code == 200
    assert all(item["uuid"] != a_uuid for item in own_listing.json())
