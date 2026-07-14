"""Tests for the tools router — parameter-id uniqueness validation.

A tool parameter is identified by its `id`, and a duplicate is only representable
in the array-valued param lists (`config.parameters`, `config.webhook.queryParameters`,
`config.webhook.body.parameters`) — object children are keyed by name so JSON
parsing already dedupes them. The frontend rejects duplicate ids; these tests pin
the same enforcement on the backend so an API caller can't slip one past it.
"""

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
            "first_name": "T",
            "last_name": "U",
            "email": f"tool-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _create(client, headers, config):
    return client.post(
        "/tools",
        json={
            "name": f"tool-{uuid.uuid4().hex[:6]}",
            "description": "d",
            "config": config,
        },
        headers=headers,
    )


def test_create_tool_accepts_distinct_parameter_ids(client):
    h = _signup(client)
    resp = _create(
        client,
        h,
        {"type": "structured_output", "parameters": [{"id": "a"}, {"id": "b"}]},
    )
    assert resp.status_code == 200, resp.text


def test_create_tool_rejects_duplicate_parameter_ids(client):
    h = _signup(client)
    resp = _create(
        client,
        h,
        {"type": "structured_output", "parameters": [{"id": "a"}, {"id": "A"}]},
    )
    assert resp.status_code == 422, resp.text
    assert "parameter" in resp.text.lower()


def test_create_tool_rejects_duplicate_webhook_query_parameter_ids(client):
    h = _signup(client)
    resp = _create(
        client,
        h,
        {
            "type": "webhook",
            "webhook": {"queryParameters": [{"id": "q"}, {"id": "q"}]},
        },
    )
    assert resp.status_code == 422, resp.text


def test_create_tool_rejects_duplicate_webhook_body_parameter_ids(client):
    h = _signup(client)
    resp = _create(
        client,
        h,
        {
            "type": "webhook",
            "webhook": {"body": {"parameters": [{"id": "b"}, {"id": "b"}]}},
        },
    )
    assert resp.status_code == 422, resp.text


def test_create_tool_object_children_keyed_by_name_are_not_flagged(client):
    # Object children are a JSON object keyed by name — no `id` array, so no
    # duplicate is representable even when a key repeats the top-level id.
    h = _signup(client)
    resp = _create(
        client,
        h,
        {
            "type": "structured_output",
            "parameters": [
                {"id": "obj", "type": "object", "properties": {"city": {}, "region": {}}}
            ],
        },
    )
    assert resp.status_code == 200, resp.text


def test_update_tool_rejects_duplicate_parameter_ids(client):
    h = _signup(client)
    created = _create(
        client, h, {"type": "structured_output", "parameters": [{"id": "a"}]}
    ).json()
    resp = client.put(
        f"/tools/{created['uuid']}",
        json={"config": {"parameters": [{"id": "x"}, {"id": "x"}]}},
        headers=h,
    )
    assert resp.status_code == 422, resp.text
