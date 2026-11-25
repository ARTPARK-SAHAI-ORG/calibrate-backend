"""Workspace limits: defaults, stored values, validation."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi import HTTPException
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
        "max_chat_replies_per_day": None,
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
    (
        "max_chat_replies_per_day",
        "DEFAULT_MAX_CHAT_REPLIES_PER_DAY",
        org_limits.effective_max_chat_replies_per_day,
        9,
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
        "max_chat_replies_per_day": org_limits.DEFAULT_MAX_CHAT_REPLIES_PER_DAY,
        "uses_own_provider_keys": False,
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
                "max_chat_replies_per_day": 16,
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
        "max_chat_replies_per_day": 16,
        "uses_own_provider_keys": False,
    }


def test_own_provider_keys_lift_only_the_spend_limits(client, monkeypatch):
    import provider_keys

    h, org = _signup_as_superadmin(client, monkeypatch)
    client.post(
        "/org-limits",
        json={"org_uuid": org, "limits": {"max_rows_per_eval": 3, "max_scored_traces": 4, "max_traces": 5}},
        headers=h,
    )
    monkeypatch.setattr(provider_keys, "org_uses_own_keys", lambda o: o == org)

    me = client.get("/org-limits/me", headers=h).json()
    assert me["max_rows_per_eval"] is None
    assert me["max_scored_traces"] is None
    assert me["max_traces"] == 5
    assert me["trace_scoring_batch_size"] == org_limits.DEFAULT_TRACE_SCORING_BATCH_SIZE
    assert me["max_chat_replies_per_day"] is None
    assert me["uses_own_provider_keys"] is True
    assert client.get("/org-limits/me/max-rows-per-eval", headers=h).json() == {
        "max_rows_per_eval": None
    }
    assert client.get("/org-limits/me/max-scored-traces", headers=h).json() == {
        "max_scored_traces": None
    }
    assert org_limits.effective_max_traces(org) == 5
    org_limits.enforce_max_rows_per_eval(org, 10_000, "test_run")
    # The stored values survive, so deleting the keys brings them back.
    assert client.get(f"/org-limits/{org}", headers=h).json()["limits"]["max_rows_per_eval"] == 3

    monkeypatch.setattr(provider_keys, "org_uses_own_keys", lambda o: False)
    assert org_limits.effective_max_rows_per_eval(org) == 3
    assert org_limits.effective_max_scored_traces(org) == 4


def _agent(config=None, org_uuid="org"):
    return {"org_uuid": org_uuid, "config": config}


def test_chat_replies_per_day_takes_the_agents_number_else_the_workspace_default(
    monkeypatch,
):
    org = {"settings": {"talk_to_agent": {"default_messages_per_day": 30}}}
    monkeypatch.setattr(org_limits, "get_organization", lambda uuid: org)

    assert org_limits.chat_replies_per_day(_agent(), None) == 30
    assert org_limits.chat_replies_per_day(_agent({"talk_to_agent": {}}), 100) == 30
    own = _agent({"talk_to_agent": {"messages_per_day": 12}})
    assert org_limits.chat_replies_per_day(own, 100) == 12
    # A maximum lowered after either number was saved still holds.
    assert org_limits.chat_replies_per_day(own, 5) == 5
    assert org_limits.chat_replies_per_day(_agent(), 20) == 20


@pytest.mark.parametrize("value", [0, -1, 1.5, "3", True])
def test_messages_per_day_must_be_a_whole_number_of_at_least_one(value):
    with pytest.raises(HTTPException) as err:
        org_limits.ensure_messages_per_day_allowed(value, None)
    assert err.value.status_code == 422
    assert err.value.detail == "Messages per day must be a whole number of at least 1"


def test_messages_per_day_must_not_pass_the_workspace_maximum():
    org_limits.ensure_messages_per_day_allowed(10, 10)
    org_limits.ensure_messages_per_day_allowed(10_000, None)
    with pytest.raises(HTTPException) as err:
        org_limits.ensure_messages_per_day_allowed(11, 10)
    assert err.value.status_code == 422
    assert err.value.detail == (
        "Messages per day cannot be more than 10, the most your workspace allows"
    )
