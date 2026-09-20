"""Workspace limits: defaults, stored values, validation."""

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


def test_updating_only_rows_keeps_the_stored_scored_traces_cap(client, monkeypatch):
    h, org = _signup_as_superadmin(client, monkeypatch)
    created = client.post(
        "/org-limits",
        json={"org_uuid": org, "limits": {"max_rows_per_eval": 20, "max_scored_traces": 5000}},
        headers=h,
    )
    assert created.status_code == 200, created.text

    updated = client.put(f"/org-limits/{org}", json={"limits": {"max_rows_per_eval": 40}}, headers=h)

    assert updated.status_code == 200, updated.text
    assert updated.json()["limits"] == {
        "max_rows_per_eval": 40,
        "max_scored_traces": 5000,
        "max_traces": None,
        "trace_scoring_batch_size": None,
        "max_concurrent_trace_scoring_batches": None,
    }
    assert org_limits.effective_max_scored_traces(org) == 5000
    assert client.get("/org-limits/me/max-scored-traces", headers=h).json() == {
        "max_scored_traces": 5000
    }


NEW_LIMITS = [
    (
        "max_scored_traces",
        "DEFAULT_MAX_SCORED_TRACES",
        org_limits.effective_max_scored_traces,
        7,
    ),
    ("max_traces", "DEFAULT_MAX_TRACES", org_limits.effective_max_traces, 1234),
    (
        "trace_scoring_batch_size",
        "DEFAULT_TRACE_SCORING_BATCH_SIZE",
        org_limits.effective_trace_scoring_batch_size,
        7,
    ),
    (
        "max_concurrent_trace_scoring_batches",
        "DEFAULT_MAX_CONCURRENT_TRACE_SCORING_BATCHES",
        org_limits.effective_max_concurrent_trace_scoring_batches,
        4,
    ),
]


@pytest.mark.parametrize("key,default_name,reader,value", NEW_LIMITS)
def test_new_limit_reads_default_then_stored_value(
    client, monkeypatch, key, default_name, reader, value
):
    h, org = _signup_as_superadmin(client, monkeypatch)
    default = getattr(org_limits, default_name)

    assert reader(org) == default

    created = client.post(
        "/org-limits",
        json={"org_uuid": org, "limits": {"max_rows_per_eval": 20, key: value}},
        headers=h,
    )
    assert created.status_code == 200, created.text
    assert reader(org) == value
    assert client.get("/org-limits/me", headers=h).json()[key] == value


@pytest.mark.parametrize("key,default_name,reader,value", NEW_LIMITS)
def test_limits_row_missing_the_new_key_falls_back_to_default(
    client, monkeypatch, key, default_name, reader, value
):
    h, org = _signup_as_superadmin(client, monkeypatch)
    created = client.post(
        "/org-limits",
        json={"org_uuid": org, "limits": {"max_rows_per_eval": 20}},
        headers=h,
    )
    assert created.status_code == 200, created.text

    got = client.get(f"/org-limits/{org}", headers=h)
    assert got.status_code == 200, got.text
    assert got.json()["limits"][key] is None
    assert reader(org) == getattr(org_limits, default_name)


@pytest.mark.parametrize("key,default_name,reader,value", NEW_LIMITS)
def test_new_limit_of_zero_is_rejected(
    client, monkeypatch, key, default_name, reader, value
):
    h, org = _signup_as_superadmin(client, monkeypatch)
    rejected = client.post(
        "/org-limits",
        json={"org_uuid": org, "limits": {"max_rows_per_eval": 20, key: 0}},
        headers=h,
    )
    assert rejected.status_code == 422, rejected.text
    assert reader(org) == getattr(org_limits, default_name)


def test_me_returns_every_effective_limit(client, monkeypatch):
    h, org = _signup_as_superadmin(client, monkeypatch)

    r = client.get("/org-limits/me", headers=h)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "max_rows_per_eval": org_limits.DEFAULT_MAX_ROWS_PER_EVAL,
        "max_scored_traces": org_limits.DEFAULT_MAX_SCORED_TRACES,
        "max_traces": org_limits.DEFAULT_MAX_TRACES,
        "trace_scoring_batch_size": org_limits.DEFAULT_TRACE_SCORING_BATCH_SIZE,
        "max_concurrent_trace_scoring_batches": org_limits.DEFAULT_MAX_CONCURRENT_TRACE_SCORING_BATCHES,
    }

    created = client.post(
        "/org-limits",
        json={
            "org_uuid": org,
            "limits": {
                "max_rows_per_eval": 11,
                "max_scored_traces": 12,
                "max_traces": 13,
                "trace_scoring_batch_size": 14,
                "max_concurrent_trace_scoring_batches": 15,
            },
        },
        headers=h,
    )
    assert created.status_code == 200, created.text
    assert client.get("/org-limits/me", headers=h).json() == {
        "max_rows_per_eval": 11,
        "max_scored_traces": 12,
        "max_traces": 13,
        "trace_scoring_batch_size": 14,
        "max_concurrent_trace_scoring_batches": 15,
    }
