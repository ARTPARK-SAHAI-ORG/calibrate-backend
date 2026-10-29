"""Every refusal for going over a limit leaves one row in limit_hits."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import db
from routers import org_limits


@pytest.fixture(scope="module")
def client():
    import main as main_mod

    with patch("main.recover_pending_jobs"):
        with TestClient(main_mod.app) as c:
            yield c


def _signup(client):
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "L",
            "last_name": "H",
            "email": f"hit-{uuid.uuid4().hex[:8]}@example.com",
            "password": "passw0rd",
        },
    ).json()
    h = {"Authorization": f"Bearer {body['access_token']}"}
    return h, db.get_personal_org_for_user(body["user"]["uuid"])["uuid"]


def hits(org_uuid):
    with db.get_db_connection() as conn:
        return [
            tuple(r)
            for r in conn.execute(
                "SELECT limit_name, kind, requested, cap FROM limit_hits WHERE org_uuid = ?",
                (org_uuid,),
            )
        ]


def test_enforce_records_one_hit_only_when_refused(monkeypatch):
    monkeypatch.setattr(org_limits, "DEFAULT_MAX_ROWS_PER_EVAL", 3)
    org = str(uuid.uuid4())
    org_limits.enforce_max_rows_per_eval(org, 3, "stt")
    assert hits(org) == []
    with pytest.raises(HTTPException):
        org_limits.enforce_max_rows_per_eval(org, 4, "stt")
    assert hits(org) == [("max_rows_per_eval", "stt", 4, 3)]


def test_a_failed_insert_still_returns_the_400(monkeypatch):
    monkeypatch.setattr(org_limits, "DEFAULT_MAX_ROWS_PER_EVAL", 3)
    with patch.object(org_limits, "create_limit_hit", side_effect=RuntimeError("db down")):
        with pytest.raises(HTTPException) as exc:
            org_limits.enforce_max_rows_per_eval(str(uuid.uuid4()), 4, "tts")
    assert exc.value.status_code == 400


def test_post_hit_records_it_with_the_workspace_cap(client, monkeypatch):
    monkeypatch.setattr(org_limits, "DEFAULT_MAX_ROWS_PER_EVAL", 7)
    h, org = _signup(client)
    resp = client.post(
        "/org-limits/hits",
        json={"limit_name": "max_rows_per_eval", "kind": "simulation", "requested": 9},
        headers=h,
    )
    assert resp.status_code == 204, resp.text
    assert resp.content == b""
    assert hits(org) == [("max_rows_per_eval", "simulation", 9, 7)]

    no_kind = client.post(
        "/org-limits/hits",
        json={"limit_name": "max_rows_per_eval", "kind": None, "requested": 8},
        headers=h,
    )
    assert no_kind.status_code == 204
    assert ("max_rows_per_eval", None, 8, 7) in hits(org)

    # A run that was not over the limit is not a hit.
    under = client.post(
        "/org-limits/hits",
        json={"limit_name": "max_rows_per_eval", "kind": "stt", "requested": 7},
        headers=h,
    )
    assert under.status_code == 204
    assert len(hits(org)) == 2


@pytest.mark.parametrize(
    "body",
    [
        {"limit_name": "max_widgets", "kind": "stt", "requested": 2},
        {"limit_name": "max_rows_per_eval", "kind": "nonsense", "requested": 2},
    ],
)
def test_post_hit_rejects_unknown_names(client, body):
    h, org = _signup(client)
    assert client.post("/org-limits/hits", json=body, headers=h).status_code == 422
    assert hits(org) == []


def test_post_hit_requires_auth(client):
    resp = client.post(
        "/org-limits/hits",
        json={"limit_name": "max_rows_per_eval", "kind": "stt", "requested": 2},
    )
    assert resp.status_code in (401, 403)


def test_trace_cap_hits_are_kept_to_one_an_hour_but_row_limit_hits_are_not():
    org = str(uuid.uuid4())
    for _ in range(3):
        db.create_limit_hit(org, "max_traces", "trace", 11, 10)
        db.create_limit_hit(org, "max_rows_per_eval", "stt", 4, 3)
    assert sorted(hits(org)) == sorted(
        [("max_traces", "trace", 11, 10)] + [("max_rows_per_eval", "stt", 4, 3)] * 3
    )
    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE limit_hits SET created_at = datetime('now', '-2 hours') WHERE org_uuid = ?",
            (org,),
        )
        conn.commit()
    db.create_limit_hit(org, "max_traces", "trace", 12, 10)
    assert hits(org).count(("max_traces", "trace", 12, 10)) == 1
