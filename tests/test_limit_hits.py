"""Every refusal for going over a limit leaves one row in limit_hits."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import db
from routers import org_limits


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


def test_chat_reply_limit_hits_are_kept_to_one_an_hour_apart_from_trace_cap_hits():
    org = str(uuid.uuid4())
    db.create_limit_hit(org, "max_traces", "trace", 11, 10)
    for _ in range(3):
        db.create_limit_hit(org, "max_chat_replies_per_day", "chat", 201, 200)
    assert sorted(hits(org)) == [
        ("max_chat_replies_per_day", "chat", 201, 200),
        ("max_traces", "trace", 11, 10),
    ]
    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE limit_hits SET created_at = datetime('now', '-2 hours') WHERE org_uuid = ?",
            (org,),
        )
        conn.commit()
    db.create_limit_hit(org, "max_chat_replies_per_day", "chat", 202, 200)
    assert hits(org).count(("max_chat_replies_per_day", "chat", 202, 200)) == 1
