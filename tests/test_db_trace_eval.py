"""Schema tests for trace_eval_runs and trace_eval_scores.

This slice ships tables + indexes only -- no enqueue/claim/settle logic yet,
so these tests write directly via raw SQL.
"""

from __future__ import annotations

import sqlite3
import uuid

import pytest

import db
import trace_scoring as ts


def _org() -> str:
    return str(uuid.uuid4())


def _ts(seconds: int) -> str:
    return ts.add_seconds("2026-01-01 00:00:00", seconds)


def _ingest_trace(org: str, agent_id: str = "agent-1") -> dict:
    return db.create_trace(
        org_uuid=org,
        agent_id=agent_id,
        message_id=str(uuid.uuid4()),
        conversation_id="conv-1",
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello", "tool_calls": None},
        metadata=None,
    )


def _insert_run(org: str, trace_uuid: str, *, run_uuid: str | None = None, **overrides):
    row = {
        "uuid": run_uuid or str(uuid.uuid4()),
        "trace_uuid": trace_uuid,
        "org_uuid": org,
        "agent_id": "agent-1",
        "status": "pending",
        "available_at": _ts(0),
        "attempts": 0,
        "error": None,
        "created_at": _ts(1),
        "updated_at": _ts(1),
        "completed_at": None,
    }
    row.update(overrides)
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO trace_eval_runs "
            "(uuid, trace_uuid, org_uuid, agent_id, status, "
            "available_at, attempts, error, created_at, updated_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["uuid"],
                row["trace_uuid"],
                row["org_uuid"],
                row["agent_id"],
                row["status"],
                row["available_at"],
                row["attempts"],
                row["error"],
                row["created_at"],
                row["updated_at"],
                row["completed_at"],
            ),
        )
        conn.commit()
    return row["uuid"]


def _insert_score(run_uuid: str, **overrides):
    row = {
        "run_uuid": run_uuid,
        "evaluator_uuid": "eval-1",
        "evaluator_version_id": "version-1",
        "value": 1,
        "output_type": "binary",
        "reasoning": "ok",
        "completed_at": _ts(10),
    }
    row.update(overrides)
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO trace_eval_scores "
            "(run_uuid, evaluator_uuid, evaluator_version_id, "
            "value, output_type, reasoning, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row["run_uuid"],
                row["evaluator_uuid"],
                row["evaluator_version_id"],
                row["value"],
                row["output_type"],
                row["reasoning"],
                row["completed_at"],
            ),
        )
        conn.commit()


def test_init_db_is_idempotent():
    db.init_db()
    db.init_db()
    with db.get_db_connection() as conn:
        names = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        indexes = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
    assert {"trace_eval_runs", "trace_eval_scores"} <= names
    assert {
        "ux_trace_eval_active",
        "ix_trace_eval_claim",
        "ix_trace_eval_agent_status",
        "ix_trace_eval_trace",
        "ix_trace_eval_org_status",
    } <= indexes


def test_active_run_uniqueness_rejects_a_second_open_run():
    org = _org()
    trace = _ingest_trace(org)
    _insert_run(org, trace["uuid"], status="pending")
    with pytest.raises(sqlite3.IntegrityError):
        _insert_run(org, trace["uuid"], status="processing")


def test_terminal_run_allows_a_new_open_run():
    org = _org()
    trace = _ingest_trace(org)
    first = _insert_run(org, trace["uuid"], status="completed", completed_at=_ts(5))
    second = _insert_run(org, trace["uuid"], status="pending", created_at=_ts(6))
    with db.get_db_connection() as conn:
        rows = conn.execute(
            "SELECT uuid, status FROM trace_eval_runs WHERE trace_uuid = ? "
            "ORDER BY created_at",
            (trace["uuid"],),
        ).fetchall()
    assert [r["uuid"] for r in rows] == [first, second]
    assert [r["status"] for r in rows] == ["completed", "pending"]


def test_typed_result_check_accepts_binary_or_rating():
    org = _org()
    trace = _ingest_trace(org)
    run = _insert_run(org, trace["uuid"], status="completed", completed_at=_ts(5))
    _insert_score(run, value=0, output_type="binary")
    _insert_score(
        run,
        evaluator_uuid="eval-rating",
        value=0.0,
        output_type="rating",
    )
    _insert_score(
        run,
        evaluator_uuid="eval-rating-high",
        value=4,
        output_type="rating",
    )
    with db.get_db_connection() as conn:
        rows = conn.execute(
            "SELECT evaluator_uuid, value, output_type FROM trace_eval_scores "
            "WHERE run_uuid = ? ORDER BY evaluator_uuid",
            (run,),
        ).fetchall()
    assert len(rows) == 3
    by_eval = {r["evaluator_uuid"]: r for r in rows}
    assert by_eval["eval-1"]["value"] == 0
    assert by_eval["eval-1"]["output_type"] == "binary"
    assert by_eval["eval-rating"]["value"] == 0.0
    assert by_eval["eval-rating"]["output_type"] == "rating"
    assert by_eval["eval-rating-high"]["value"] == 4
    assert by_eval["eval-rating-high"]["output_type"] == "rating"


def test_typed_result_check_rejects_a_non_binary_value():
    org = _org()
    trace = _ingest_trace(org)
    run = _insert_run(org, trace["uuid"], status="completed", completed_at=_ts(5))
    with pytest.raises(sqlite3.IntegrityError):
        _insert_score(run, value=2, output_type="binary")


def test_evaluator_version_id_is_required():
    org = _org()
    trace = _ingest_trace(org)
    run = _insert_run(org, trace["uuid"], status="completed", completed_at=_ts(5))
    with pytest.raises(sqlite3.IntegrityError):
        _insert_score(run, evaluator_version_id=None)


def test_same_version_scores_are_preserved_across_distinct_runs():
    org = _org()
    trace = _ingest_trace(org)
    first = _insert_run(org, trace["uuid"], status="completed", completed_at=_ts(5))
    _insert_score(
        first,
        value=1,
        output_type="binary",
        evaluator_version_id="version-same",
        reasoning="first run",
        completed_at=_ts(5),
    )
    second = _insert_run(
        org, trace["uuid"], status="completed", created_at=_ts(6), completed_at=_ts(7)
    )
    _insert_score(
        second,
        value=0,
        output_type="binary",
        evaluator_version_id="version-same",
        reasoning="rescore",
        completed_at=_ts(7),
    )
    with db.get_db_connection() as conn:
        rows = conn.execute(
            "SELECT ts.run_uuid, ts.value, ts.reasoning FROM trace_eval_scores ts "
            "JOIN trace_eval_runs r ON r.uuid = ts.run_uuid "
            "WHERE r.trace_uuid = ? ORDER BY ts.completed_at",
            (trace["uuid"],),
        ).fetchall()
    assert len(rows) == 2
    assert rows[0]["run_uuid"] == first
    assert rows[0]["value"] == 1
    assert rows[0]["reasoning"] == "first run"
    assert rows[1]["run_uuid"] == second
    assert rows[1]["value"] == 0
    assert rows[1]["reasoning"] == "rescore"


def test_update_agent_missing_row_is_false():
    assert (
        db.update_agent(str(uuid.uuid4()), name="x", delete_pending_trace_runs=True)
        is False
    )


def test_same_run_evaluator_is_unique():
    org = _org()
    trace = _ingest_trace(org)
    run = _insert_run(org, trace["uuid"], status="completed", completed_at=_ts(5))
    _insert_score(run, evaluator_version_id="v1")
    with pytest.raises(sqlite3.IntegrityError):
        _insert_score(run, evaluator_version_id="v2", value=0)


# The read helpers' user-facing contract (latest-run summary fields, full
# history, pass rule, hydration of deleted evaluators and pinned versions) is
# pinned in test_routers_traces.py. These two cover only what the endpoints
# cannot reach: the id tie-break, the helpers' own org filters, and the
# single-statement guarantee behind the "no N+1" rule.


def test_latest_run_summary_tie_breaks_on_id():
    org = _org()
    trace = _ingest_trace(org)
    first = _insert_run(
        org, trace["uuid"], status="completed", created_at=_ts(50), completed_at=_ts(50)
    )
    second = _insert_run(
        org, trace["uuid"], status="completed", created_at=_ts(50), completed_at=_ts(50)
    )
    _insert_score(first, value=1)
    _insert_score(second, value=0)
    with db.get_db_connection() as conn:
        ids = {
            r["uuid"]: r["id"]
            for r in conn.execute(
                "SELECT uuid, id FROM trace_eval_runs WHERE uuid IN (?, ?)",
                (first, second),
            ).fetchall()
        }
    assert ids[second] > ids[first]

    latest = db.get_latest_trace_run_summaries(org, [trace["uuid"]])[trace["uuid"]]
    assert [r["value"] for r in latest["results"]] == [0]


def test_score_read_helpers_are_org_scoped():
    org_a, org_b = _org(), _org()
    trace = _ingest_trace(org_a)
    run = _insert_run(org_a, trace["uuid"], status="completed", completed_at=_ts(3))
    _insert_score(run, value=1)
    assert db.get_latest_trace_run_summaries(org_b, [trace["uuid"]]) == {}
    assert db.list_trace_scoring_runs(org_b, trace["uuid"]) == []
    assert db.get_latest_trace_run_summaries(org_a, [trace["uuid"]])[trace["uuid"]][
        "results"
    ][0]["passed"] is True
    assert db.list_trace_scoring_runs(org_a, trace["uuid"])[0]["results"][0][
        "passed"
    ] is True


