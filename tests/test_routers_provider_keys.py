"""The /provider-keys endpoints a workspace uses to store its own provider keys."""

from __future__ import annotations

import json
import uuid
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

import provider_keys as pk


@pytest.fixture(scope="module")
def app():
    import main as main_mod

    return main_mod.app


@pytest.fixture(scope="module")
def client(app):
    with patch("main.recover_pending_jobs"):
        with TestClient(app) as c:
            yield c


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch):
    monkeypatch.setenv(pk.ENCRYPTION_KEY_ENV, Fernet.generate_key().decode())
    yield


def _signup(client):
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "P",
            "last_name": "K",
            "email": f"pk-{uuid.uuid4().hex[:8]}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


@pytest.fixture
def auth(client):
    return _signup(client)


SERVICE_ACCOUNT = json.dumps(
    {"type": "service_account", "client_email": "runner@proj.iam.gserviceaccount.com"}
)


def _entry(client, auth, provider):
    body = client.get("/provider-keys", headers=auth).json()
    return next(e for e in body if e["provider"] == provider)


def test_listing_requires_logging_in(client):
    assert client.get("/provider-keys").status_code == 403


def test_listing_shows_every_provider_with_nothing_stored(client, auth):
    body = client.get("/provider-keys", headers=auth).json()
    assert [e["provider"] for e in body] == list(pk.PROVIDER_ENV_VARS)
    assert all(e["configured"] is False for e in body)
    assert all(f["set"] is False and f["display"] is None for e in body for f in e["fields"])


def test_storing_a_key_marks_the_provider_configured(client, auth):
    resp = client.put(
        "/provider-keys/sarvam",
        headers=auth,
        json={"values": {"SARVAM_API_KEY": "sk-live-abcd1234"}},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "provider": "sarvam",
        "configured": True,
        "fields": [
            {
                "env_var": "SARVAM_API_KEY",
                "kind": "secret",
                "required": True,
                "set": True,
                "display": "••••1234",
            }
        ],
    }
    assert _entry(client, auth, "sarvam")["configured"] is True


def test_the_stored_key_is_never_returned(client, auth):
    client.put(
        "/provider-keys/sarvam",
        headers=auth,
        json={"values": {"SARVAM_API_KEY": "sk-live-abcd1234"}},
    )
    assert "sk-live-abcd1234" not in client.get("/provider-keys", headers=auth).text


def test_google_needs_both_values(client, auth):
    resp = client.put(
        "/provider-keys/google",
        headers=auth,
        json={"values": {"GOOGLE_CLOUD_PROJECT_ID": "my-project"}},
    )
    assert resp.status_code == 400
    assert "GOOGLE_APPLICATION_CREDENTIALS" in resp.json()["detail"]
    assert _entry(client, auth, "google")["configured"] is False


def test_google_shows_the_project_and_the_service_account(client, auth):
    resp = client.put(
        "/provider-keys/google",
        headers=auth,
        json={
            "values": {
                "GOOGLE_APPLICATION_CREDENTIALS": SERVICE_ACCOUNT,
                "GOOGLE_CLOUD_PROJECT_ID": "my-project",
            }
        },
    )
    assert resp.status_code == 200
    display = {f["env_var"]: f["display"] for f in resp.json()["fields"]}
    assert display["GOOGLE_CLOUD_PROJECT_ID"] == "my-project"
    assert display["GOOGLE_APPLICATION_CREDENTIALS"] == (
        "runner@proj.iam.gserviceaccount.com"
    )


def test_credentials_that_are_not_json_are_refused(client, auth):
    resp = client.put(
        "/provider-keys/google",
        headers=auth,
        json={
            "values": {
                "GOOGLE_APPLICATION_CREDENTIALS": "/etc/creds.json",
                "GOOGLE_CLOUD_PROJECT_ID": "my-project",
            }
        },
    )
    assert resp.status_code == 400


def test_a_value_belonging_to_another_provider_is_refused(client, auth):
    resp = client.put(
        "/provider-keys/sarvam",
        headers=auth,
        json={"values": {"OPENAI_API_KEY": "sk-1"}},
    )
    assert resp.status_code == 400


def test_an_unknown_provider_is_not_found(client, auth):
    assert client.get("/provider-keys", headers=auth).status_code == 200
    assert (
        client.put(
            "/provider-keys/not-a-provider", headers=auth, json={"values": {"X": "y"}}
        ).status_code
        == 404
    )
    assert client.delete("/provider-keys/not-a-provider", headers=auth).status_code == 404


def test_storing_again_replaces_the_previous_value(client, auth):
    for key in ("sk-first-1111", "sk-second-2222"):
        client.put(
            "/provider-keys/openai", headers=auth, json={"values": {"OPENAI_API_KEY": key}}
        )
    fields = _entry(client, auth, "openai")["fields"]
    assert [f["display"] for f in fields] == ["••••2222"]


def test_deleting_removes_the_keys_and_is_not_repeatable(client, auth):
    client.put(
        "/provider-keys/openai", headers=auth, json={"values": {"OPENAI_API_KEY": "sk-1234"}}
    )
    assert client.delete("/provider-keys/openai", headers=auth).status_code == 200
    assert _entry(client, auth, "openai")["configured"] is False
    assert client.delete("/provider-keys/openai", headers=auth).status_code == 404


def test_a_deleted_provider_can_be_set_up_again(client, auth):
    body = {"values": {"OPENAI_API_KEY": "sk-1234"}}
    client.put("/provider-keys/openai", headers=auth, json=body)
    client.delete("/provider-keys/openai", headers=auth)
    assert client.put("/provider-keys/openai", headers=auth, json=body).status_code == 200
    assert _entry(client, auth, "openai")["configured"] is True


def test_one_workspace_cannot_see_anothers_keys(client, auth):
    client.put(
        "/provider-keys/sarvam", headers=auth, json={"values": {"SARVAM_API_KEY": "mine"}}
    )
    other = _signup(client)
    assert _entry(client, other, "sarvam")["configured"] is False
    assert client.delete("/provider-keys/sarvam", headers=other).status_code == 404


def test_saving_is_refused_when_the_server_has_no_encryption_key(
    client, auth, monkeypatch
):
    monkeypatch.delenv(pk.ENCRYPTION_KEY_ENV, raising=False)
    resp = client.put(
        "/provider-keys/sarvam", headers=auth, json={"values": {"SARVAM_API_KEY": "k"}}
    )
    assert resp.status_code == 503
    assert pk.ENCRYPTION_KEY_ENV in resp.json()["detail"]


def test_the_run_uses_a_stored_key_over_the_servers(client, auth, monkeypatch, tmp_path):
    monkeypatch.setenv("SARVAM_API_KEY", "server-key")
    client.put(
        "/provider-keys/sarvam",
        headers=auth,
        json={"values": {"SARVAM_API_KEY": "workspace-key"}},
    )
    org = client.get("/organizations", headers=auth).json()[0]["uuid"]

    assert pk.provider_env(org, tmp_path)["SARVAM_API_KEY"] == "workspace-key"


def test_a_stored_key_makes_the_provider_available(client, auth, monkeypatch):
    import provider_status

    for env_vars in provider_status.PROVIDER_ENV_VARS.values():
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("FAKE_AI_PROVIDERS", raising=False)

    assert client.get("/providers", headers=auth).json()["providers"] == []
    client.put(
        "/provider-keys/sarvam", headers=auth, json={"values": {"SARVAM_API_KEY": "k"}}
    )
    assert client.get("/providers", headers=auth).json()["providers"] == ["sarvam"]


def test_the_provider_keys_routes_are_not_on_the_public_api(app):
    for route in app.routes:
        if getattr(route, "path", "").startswith("/provider-keys"):
            assert "Public API" not in (getattr(route, "tags", None) or [])
