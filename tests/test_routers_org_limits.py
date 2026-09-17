"""The workspace's max scored traces limit: default, stored value, validation."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import auth_utils
import db
from routers import org_limits


@pytest.fixture(scope="module")
def app():
    import main as main_mod

    return main_mod.app


@pytest.fixture(scope="module")
def client(app):
    with patch("main.recover_pending_jobs"):
        with TestClient(app) as c:
            yield c


def _signup_as_superadmin(client, monkeypatch):
    email = f"lim-{uuid.uuid4().hex[:8]}@example.com"
    body = client.post(
        "/auth/signup",
        json={"first_name": "L", "last_name": "M", "email": email, "password": "passw0rd"},
    ).json()
    monkeypatch.setattr(auth_utils, "SUPERADMIN_EMAIL", email)
    h = {"Authorization": f"Bearer {body['access_token']}"}
    return h, db.get_personal_org_for_user(body["user"]["uuid"])["uuid"]


def test_max_scored_traces_reads_default_then_stored_value(client, monkeypatch):
    h, org = _signup_as_superadmin(client, monkeypatch)

    r = client.get("/org-limits/me/max-scored-traces", headers=h)
    assert r.status_code == 200, r.text
    assert r.json() == {"max_scored_traces": org_limits.DEFAULT_MAX_SCORED_TRACES}

    created = client.post(
        "/org-limits",
        json={"org_uuid": org, "limits": {"max_rows_per_eval": 20, "max_scored_traces": 3}},
        headers=h,
    )
    assert created.status_code == 200, created.text
    assert client.get("/org-limits/me/max-scored-traces", headers=h).json() == {
        "max_scored_traces": 3
    }

    updated = client.put(
        f"/org-limits/{org}", json={"limits": {"max_rows_per_eval": 20, "max_scored_traces": 7}}, headers=h
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["limits"]["max_scored_traces"] == 7
    assert client.get("/org-limits/me/max-scored-traces", headers=h).json() == {
        "max_scored_traces": 7
    }
    assert org_limits.effective_max_scored_traces(org) == 7


def test_limits_row_without_the_key_still_reads_and_falls_back_to_default(
    client, monkeypatch
):
    h, org = _signup_as_superadmin(client, monkeypatch)
    created = client.post(
        "/org-limits",
        json={"org_uuid": org, "limits": {"max_rows_per_eval": 50}},
        headers=h,
    )
    assert created.status_code == 200, created.text

    got = client.get(f"/org-limits/{org}", headers=h)
    assert got.status_code == 200, got.text
    assert got.json()["limits"] == {"max_rows_per_eval": 50, "max_scored_traces": None}
    assert client.get("/org-limits/me/max-scored-traces", headers=h).json() == {
        "max_scored_traces": org_limits.DEFAULT_MAX_SCORED_TRACES
    }
    assert client.get("/org-limits/me/max-rows-per-eval", headers=h).json() == {
        "max_rows_per_eval": 50
    }


def test_max_scored_traces_of_zero_is_rejected(client, monkeypatch):
    h, org = _signup_as_superadmin(client, monkeypatch)
    rejected = client.post(
        "/org-limits",
        json={"org_uuid": org, "limits": {"max_rows_per_eval": 20, "max_scored_traces": 0}},
        headers=h,
    )
    assert rejected.status_code == 422, rejected.text
    assert client.get("/org-limits/me/max-scored-traces", headers=h).json() == {
        "max_scored_traces": org_limits.DEFAULT_MAX_SCORED_TRACES
    }

    created = client.post(
        "/org-limits",
        json={"org_uuid": org, "limits": {"max_rows_per_eval": 20, "max_scored_traces": 5}},
        headers=h,
    )
    assert created.status_code == 200, created.text
    rejected_update = client.put(
        f"/org-limits/{org}", json={"limits": {"max_rows_per_eval": 20, "max_scored_traces": 0}}, headers=h
    )
    assert rejected_update.status_code == 422, rejected_update.text
    assert client.get("/org-limits/me/max-scored-traces", headers=h).json() == {
        "max_scored_traces": 5
    }
