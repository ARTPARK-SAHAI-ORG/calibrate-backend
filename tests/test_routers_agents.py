"""Integration tests for /agents, focused on the name→UUID resolve endpoint."""

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
            "first_name": "Res",
            "last_name": "Olve",
            "email": f"res-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _create_agent(client, h, name):
    return client.post(
        "/agents", json={"name": name, "type": "agent"}, headers=h
    ).json()


def _raw_key(client, h, name="ci"):
    return client.post("/api-keys", json={"name": name}, headers=h).json()["key"]


def test_resolve_agent_names_with_jwt(client):
    h = _signup(client)
    n1 = f"alpha-{uuid.uuid4().hex[:6]}"
    n2 = f"beta-{uuid.uuid4().hex[:6]}"
    a1 = _create_agent(client, h, n1)
    a2 = _create_agent(client, h, n2)
    missing = f"ghost-{uuid.uuid4().hex[:6]}"

    r = client.post(
        "/agents/resolve", json={"names": [n1, n2, missing]}, headers=h
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["resolved"] == {n1: a1["uuid"], n2: a2["uuid"]}
    assert body["not_found"] == [missing]


def test_resolve_agent_names_with_api_key(client):
    h = _signup(client)
    name = f"keyed-{uuid.uuid4().hex[:6]}"
    agent = _create_agent(client, h, name)
    raw = _raw_key(client, h)

    # X-API-Key header
    r1 = client.post(
        "/agents/resolve", json={"names": [name]}, headers={"X-API-Key": raw}
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["resolved"] == {name: agent["uuid"]}

    # Authorization: Bearer sk_…
    r2 = client.post(
        "/agents/resolve",
        json={"names": [name]},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["resolved"] == {name: agent["uuid"]}


def test_resolve_dedupes_not_found(client):
    h = _signup(client)
    missing = f"none-{uuid.uuid4().hex[:6]}"
    r = client.post(
        "/agents/resolve", json={"names": [missing, missing]}, headers=h
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["resolved"] == {}
    assert body["not_found"] == [missing]


def test_resolve_is_org_scoped(client):
    """An agent in org A must not resolve for a caller in org B."""
    ha = _signup(client)
    name = f"private-{uuid.uuid4().hex[:6]}"
    _create_agent(client, ha, name)

    hb = _signup(client)
    r = client.post("/agents/resolve", json={"names": [name]}, headers=hb)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["resolved"] == {}
    assert body["not_found"] == [name]


def test_resolve_requires_auth(client):
    r = client.post("/agents/resolve", json={"names": ["whatever"]})
    assert r.status_code in (401, 403)

    bad = client.post(
        "/agents/resolve",
        json={"names": ["whatever"]},
        headers={"X-API-Key": "sk_not-a-real-key"},
    )
    assert bad.status_code == 401
