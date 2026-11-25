"""Integration tests for the /agent-tools router."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import db


@pytest.fixture(scope="module")
def client():
    import main as main_mod

    with patch("main.recover_pending_jobs"):
        with TestClient(main_mod.app) as c:
            yield c


def _workspace_with_agent_and_tool(client):
    suffix = uuid.uuid4().hex[:8]
    token = client.post(
        "/auth/signup",
        json={
            "first_name": "At",
            "last_name": "U",
            "email": f"at-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()["access_token"]
    h = {"Authorization": f"Bearer {token}"}
    agent = client.post(
        "/agents", json={"name": f"a-{suffix}", "type": "agent"}, headers=h
    ).json()
    tool = client.post(
        "/tools",
        json={
            "name": f"t-{suffix}",
            "description": "d",
            "config": {"type": "structured_output", "parameters": []},
        },
        headers=h,
    ).json()
    return h, agent["uuid"], tool["uuid"]


def _active_links(agent_uuid, evaluator_uuid):
    with db.get_db_connection() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM agent_evaluators "
            "WHERE agent_id = ? AND evaluator_id = ? AND deleted_at IS NULL",
            (agent_uuid, evaluator_uuid),
        ).fetchone()[0]


def _link(client, h, agent_uuid, tool_uuids):
    res = client.post(
        "/agent-tools",
        json={"agent_uuid": agent_uuid, "tool_uuids": tool_uuids},
        headers=h,
    )
    assert res.status_code == 200, res.text


def _tool_call_evaluator(agent_uuid):
    return db.get_tool_call_evaluator_for_org(db.get_agent(agent_uuid)["org_uuid"])


def test_linking_a_tool_links_the_tool_call_evaluator_once(client):
    h, agent_uuid, tool_uuid = _workspace_with_agent_and_tool(client)
    evaluator = _tool_call_evaluator(agent_uuid)
    assert _active_links(agent_uuid, evaluator["uuid"]) == 0

    _link(client, h, agent_uuid, [tool_uuid])
    assert _active_links(agent_uuid, evaluator["uuid"]) == 1

    _link(client, h, agent_uuid, [tool_uuid])
    assert _active_links(agent_uuid, evaluator["uuid"]) == 1


def test_linking_no_tools_links_nothing(client):
    h, agent_uuid, _ = _workspace_with_agent_and_tool(client)
    _link(client, h, agent_uuid, [])
    assert _active_links(agent_uuid, _tool_call_evaluator(agent_uuid)["uuid"]) == 0


def test_a_deleted_tool_call_evaluator_is_not_linked(client):
    h, agent_uuid, tool_uuid = _workspace_with_agent_and_tool(client)
    evaluator = _tool_call_evaluator(agent_uuid)
    db.delete_evaluator(evaluator["uuid"])

    _link(client, h, agent_uuid, [tool_uuid])
    assert _active_links(agent_uuid, evaluator["uuid"]) == 0


def test_relinking_an_already_linked_tool_does_not_restore_a_removed_evaluator(client):
    h, agent_uuid, tool_uuid = _workspace_with_agent_and_tool(client)
    evaluator = _tool_call_evaluator(agent_uuid)
    _link(client, h, agent_uuid, [tool_uuid])
    db.remove_evaluator_from_agent(agent_uuid, evaluator["uuid"])

    _link(client, h, agent_uuid, [tool_uuid])
    assert _active_links(agent_uuid, evaluator["uuid"]) == 0


def test_backfill_links_the_evaluator_to_agents_that_already_have_tools(client):
    h, with_tool, tool_uuid = _workspace_with_agent_and_tool(client)
    db.add_tool_to_agent(agent_id=with_tool, tool_id=tool_uuid)
    without_tool = client.post(
        "/agents", json={"name": f"b-{uuid.uuid4().hex[:8]}", "type": "agent"}, headers=h
    ).json()["uuid"]
    evaluator = _tool_call_evaluator(with_tool)

    for _ in range(2):
        with db.get_db_connection() as conn:
            conn.execute(
                "DELETE FROM _schema_migrations WHERE name = ?",
                (db.LINK_TOOL_CALL_EVALUATOR_TO_AGENTS_WITH_TOOLS_MIGRATION,),
            )
            conn.commit()
        db.init_db()

    assert _active_links(with_tool, evaluator["uuid"]) == 1
    assert _active_links(without_tool, evaluator["uuid"]) == 0
