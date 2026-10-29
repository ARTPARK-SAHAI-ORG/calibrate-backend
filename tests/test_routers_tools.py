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


PUBLIC_ADDR = [(2, 1, 6, "", ("93.184.215.14", 0))]


def _fake_getaddrinfo(host, *args, **kwargs):
    """An IP address resolves to itself; any name resolves to a public address."""
    import ipaddress

    try:
        ipaddress.ip_address(host)
        return [(2, 1, 6, "", (host, 0))]
    except ValueError:
        return PUBLIC_ADDR


@pytest.fixture(autouse=True)
def _public_dns():
    """So the tests need no network."""
    with patch("utils.socket.getaddrinfo", side_effect=_fake_getaddrinfo):
        yield


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
            "webhook": {"url": "https://example.com/hook", "queryParameters": [{"id": "q"}, {"id": "q"}]},
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
            "webhook": {"url": "https://example.com/hook", "body": {"parameters": [{"id": "b"}, {"id": "b"}]}},
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


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost:8000/admin",
        "http://10.0.0.5/",
        "file:///etc/passwd",
        None,
    ],
)
def test_create_tool_rejects_webhook_url_inside_the_server_network(client, url):
    """The backend calls a webhook from its own server, so an address only that
    server can reach (such as the AWS address that hands out its credentials)
    must be refused."""
    h = _signup(client)
    resp = _create(client, h, {"type": "webhook", "webhook": {"url": url}})
    assert resp.status_code == 422, resp.text
    assert "webhook url" in resp.text


def test_create_tool_rejects_hostname_resolving_to_a_private_address(client):
    h = _signup(client)
    private = [(2, 1, 6, "", ("169.254.169.254", 0))]
    with patch("utils.socket.getaddrinfo", return_value=private):
        resp = _create(
            client,
            h,
            {"type": "webhook", "webhook": {"url": "https://looks-public.example"}},
        )
    assert resp.status_code == 422, resp.text


def test_update_tool_rejects_internal_webhook_url(client):
    h = _signup(client)
    created = _create(
        client, h, {"type": "webhook", "webhook": {"url": "https://example.com/hook"}}
    )
    assert created.status_code == 200, created.text
    resp = client.put(
        f"/tools/{created.json()['uuid']}",
        json={"config": {"type": "webhook", "webhook": {"url": "http://127.0.0.1/"}}},
        headers=h,
    )
    assert resp.status_code == 422, resp.text
