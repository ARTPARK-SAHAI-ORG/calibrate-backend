"""Tests for the tools router — parameter-name uniqueness validation.

The frontend rejects duplicate parameter names among siblings (recursing into
nested object/array parameters); these tests pin the same enforcement on the
backend so an API caller can't slip a duplicate past it.
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


def test_create_tool_accepts_distinct_parameter_names(client):
    h = _signup(client)
    resp = _create(
        client,
        h,
        {"type": "structured_output", "parameters": [{"name": "a"}, {"name": "b"}]},
    )
    assert resp.status_code == 200, resp.text


def test_create_tool_rejects_duplicate_sibling_parameter_names(client):
    h = _signup(client)
    resp = _create(
        client,
        h,
        {"type": "structured_output", "parameters": [{"name": "a"}, {"name": "A"}]},
    )
    assert resp.status_code == 422, resp.text
    assert "parameter" in resp.text.lower()


def test_create_tool_rejects_duplicate_nested_parameter_names(client):
    h = _signup(client)
    resp = _create(
        client,
        h,
        {
            "type": "structured_output",
            "parameters": [
                {
                    "name": "obj",
                    "type": "object",
                    "parameters": [{"name": "dup"}, {"name": "dup"}],
                }
            ],
        },
    )
    assert resp.status_code == 422, resp.text


def test_create_tool_same_name_across_levels_is_allowed(client):
    h = _signup(client)
    resp = _create(
        client,
        h,
        {
            "type": "structured_output",
            "parameters": [
                {"name": "id"},
                {"name": "obj", "type": "object", "parameters": [{"name": "id"}]},
            ],
        },
    )
    assert resp.status_code == 200, resp.text


def test_update_tool_rejects_duplicate_parameter_names(client):
    h = _signup(client)
    created = _create(
        client, h, {"type": "structured_output", "parameters": [{"name": "a"}]}
    ).json()
    resp = client.put(
        f"/tools/{created['uuid']}",
        json={"config": {"parameters": [{"name": "x"}, {"name": "x"}]}},
        headers=h,
    )
    assert resp.status_code == 422, resp.text
