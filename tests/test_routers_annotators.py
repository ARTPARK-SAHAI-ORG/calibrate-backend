"""Integration tests for /annotators, focused on the API-key (Public API) surface."""

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
            "first_name": "Ann",
            "last_name": "Otator",
            "email": f"ann-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _raw_key(client, h, name="ci"):
    return client.post("/api-keys", json={"name": name}, headers=h).json()["key"]


def test_create_list_update_annotator_with_api_key(client):
    h = _signup(client)
    raw = _raw_key(client, h)
    name = f"ann-{uuid.uuid4().hex[:6]}"

    created = client.post("/annotators", json={"name": name}, headers={"X-API-Key": raw})
    assert created.status_code == 200, created.text
    annotator_uuid = created.json()["uuid"]

    listed = client.get("/annotators", headers={"Authorization": f"Bearer {raw}"})
    assert listed.status_code == 200, listed.text
    assert annotator_uuid in {a["uuid"] for a in listed.json()}

    new_name = f"{name}-renamed"
    updated = client.put(
        f"/annotators/{annotator_uuid}",
        json={"name": new_name},
        headers={"X-API-Key": raw},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == new_name


def test_annotators_are_org_scoped_for_api_keys(client):
    h_a = _signup(client)
    h_b = _signup(client)
    name = f"ann-{uuid.uuid4().hex[:6]}"
    created = client.post("/annotators", json={"name": name}, headers=h_a).json()
    raw_b = _raw_key(client, h_b)

    assert created["uuid"] not in {
        a["uuid"] for a in client.get("/annotators", headers={"X-API-Key": raw_b}).json()
    }
    # The annotator exists in the other org, so the answer is 403 — and a key is
    # bound to one org, so the owning org is not named.
    cross = client.put(
        f"/annotators/{created['uuid']}", json={"name": "x"}, headers={"X-API-Key": raw_b}
    )
    assert cross.status_code == 403
    assert "organization_uuid" not in cross.json()


def test_annotators_reject_invalid_api_key(client):
    bad = {"X-API-Key": "sk_not-a-real-key"}
    assert client.get("/annotators", headers=bad).status_code == 401
    assert client.post("/annotators", json={"name": "x"}, headers=bad).status_code == 401


def test_annotator_detail_and_delete_stay_jwt_only(client):
    h = _signup(client)
    raw = _raw_key(client, h)
    name = f"ann-{uuid.uuid4().hex[:6]}"
    created = client.post("/annotators", json={"name": name}, headers=h).json()

    key_headers = {"X-API-Key": raw}
    assert client.get(f"/annotators/{created['uuid']}", headers=key_headers).status_code == 403
    assert client.delete(f"/annotators/{created['uuid']}", headers=key_headers).status_code == 403
    assert client.delete(f"/annotators/{created['uuid']}", headers=h).status_code == 200
