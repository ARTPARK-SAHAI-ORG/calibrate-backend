"""Integration tests for /provider-keys and the per-workspace /providers list."""

from __future__ import annotations

import json
import os
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

import provider_keys as pk

GOOGLE_JSON = json.dumps({"project_id": "p", "client_email": "bot@p.iam.gserviceaccount.com"})


@pytest.fixture(scope="module")
def client():
    import main as main_mod

    with patch("main.recover_pending_jobs"):
        with TestClient(main_mod.app) as c:
            yield c


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PROVIDER_KEYS_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.delenv("FAKE_AI_PROVIDERS", raising=False)


@pytest.fixture
def check():
    with patch("provider_keys.check_provider_key", AsyncMock(return_value=None)) as m:
        yield m


def _signup(client):
    suffix = uuid.uuid4().hex[:8]
    body = client.post(
        "/auth/signup",
        json={"first_name": "P", "last_name": "K", "email": f"pk-{suffix}@example.com", "password": "passw0rd"},
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _org(client, h):
    return client.get("/organizations", headers=h).json()[0]["uuid"]


def test_put_list_and_replace(client, check):
    h = _signup(client)
    assert client.get("/provider-keys", headers=h).json() == {"uses_own_keys": False, "keys": []}

    resp = client.put("/provider-keys/openai", json={"value": "  sk-abcd1234  "}, headers=h)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provider"] == "openai"
    assert body["masked_value"] == "••••1234"
    assert body["updated_at"].endswith("Z")
    check.assert_awaited_with("openai", "sk-abcd1234")

    client.put("/provider-keys/openai", json={"value": "sk-zzzz9999"}, headers=h)
    listed = client.get("/provider-keys", headers=h).json()
    assert listed["uses_own_keys"] is True
    assert [k["masked_value"] for k in listed["keys"]] == ["••••9999"]
    assert "encrypted_value" not in listed["keys"][0]


def test_put_google_masks_with_client_email(client, check):
    h = _signup(client)
    resp = client.put("/provider-keys/google", json={"value": GOOGLE_JSON}, headers=h)
    assert resp.json()["masked_value"] == "bot@p.iam.gserviceaccount.com"


def test_put_rejects_bad_input(client, check):
    h = _signup(client)
    assert client.put("/provider-keys/nope", json={"value": "x"}, headers=h).status_code == 422
    assert client.put("/provider-keys/openai", json={"value": "   "}, headers=h).status_code == 422
    resp = client.put("/provider-keys/google", json={"value": "{}"}, headers=h)
    assert resp.status_code == 400
    check.assert_not_awaited()


def test_put_failed_check_saves_nothing(client, check):
    h = _signup(client)
    check.return_value = "401 Unauthorized"
    resp = client.put("/provider-keys/openai", json={"value": "bad"}, headers=h)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "401 Unauthorized"
    assert client.get("/provider-keys", headers=h).json()["keys"] == []


def test_put_without_encryption_key_is_503(client, check, monkeypatch):
    h = _signup(client)
    monkeypatch.delenv("PROVIDER_KEYS_ENCRYPTION_KEY")
    resp = client.put("/provider-keys/openai", json={"value": "x"}, headers=h)
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Saving provider keys is not set up on this server."


def test_api_key_cannot_use_routes(client):
    resp = client.get("/provider-keys", headers={"Authorization": "Bearer sk_fake"})
    assert resp.status_code == 401


def test_delete_unknown_is_404(client):
    h = _signup(client)
    assert client.delete("/provider-keys/openai", headers=h).status_code == 404


def test_delete_not_last_key_skips_trim(client, check):
    h = _signup(client)
    client.put("/provider-keys/openai", json={"value": "k1"}, headers=h)
    client.put("/provider-keys/deepgram", json={"value": "k2"}, headers=h)
    with patch("db.org_has_active_jobs", return_value=True), patch(
        "db.trim_pending_trace_runs_over_cap"
    ) as trim:
        assert client.delete("/provider-keys/openai", headers=h).status_code == 204
    trim.assert_not_called()
    assert [k["provider"] for k in client.get("/provider-keys", headers=h).json()["keys"]] == ["deepgram"]


def test_delete_last_key_blocked_by_active_runs(client, check):
    h = _signup(client)
    client.put("/provider-keys/openai", json={"value": "k1"}, headers=h)
    with patch("db.org_has_active_jobs", return_value=True):
        resp = client.delete("/provider-keys/openai", headers=h)
    assert resp.status_code == 409
    assert client.get("/provider-keys", headers=h).json()["uses_own_keys"] is True


def test_delete_last_key_trims_runs_and_removes_google_file(client, check):
    h = _signup(client)
    org = _org(client, h)
    client.put("/provider-keys/google", json={"value": GOOGLE_JSON}, headers=h)
    pk.provider_env(org)
    assert os.path.exists(pk.google_creds_path(org))
    with patch("db.trim_pending_trace_runs_over_cap") as trim:
        assert client.delete("/provider-keys/google", headers=h).status_code == 204
    trim.assert_called_once_with(org, 100)
    assert not os.path.exists(pk.google_creds_path(org))
    assert client.get("/provider-keys", headers=h).json()["uses_own_keys"] is False


def test_delete_google_without_file(client, check):
    h = _signup(client)
    client.put("/provider-keys/google", json={"value": GOOGLE_JSON}, headers=h)
    assert client.delete("/provider-keys/google", headers=h).status_code == 204


def test_providers_lists_org_keys_in_registry_order(client, check, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "server")
    h = _signup(client)
    client.put("/provider-keys/openrouter", json={"value": "k"}, headers=h)
    client.put("/provider-keys/deepgram", json={"value": "k"}, headers=h)
    assert client.get("/providers", headers=h).json() == {"providers": ["deepgram", "openrouter"]}

    h2 = _signup(client)
    assert "openai" in client.get("/providers", headers=h2).json()["providers"]
    bad = {"Authorization": "Bearer not-a-jwt"}
    assert "openai" in client.get("/providers", headers=bad).json()["providers"]
