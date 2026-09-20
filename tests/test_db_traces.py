"""Trace CRUD tests for the traces helpers in src/db.py."""

from __future__ import annotations

import json
import sqlite3
import uuid

import pytest

import db


def _org() -> str:
    return str(uuid.uuid4())


def _ingest(
    org: str,
    message_id: str,
    conversation_id: str = "conv-1",
    agent_id: str = "agent-1",
    **overrides,
):
    payload = {
        "input": [
            {"role": "system", "content": "You are a vaccination assistant."},
            {"role": "user", "content": "When is the next vaccination?"},
        ],
        "output": {
            "response": "At 14 weeks, for OPV and DPT.",
            "tool_calls": [{"tool": "get_schedule", "arguments": {"child_age_weeks": 14}}],
        },
        "metadata": [{"key": "gen_ai.request.model", "value": "gpt-4"}],
    }
    payload.update(overrides)
    return db.create_trace(
        org_uuid=org,
        agent_id=agent_id,
        message_id=message_id,
        conversation_id=conversation_id,
        **payload,
    )


def test_create_and_get_roundtrip():
    org = _org()
    row = _ingest(org, "m-1")
    assert len(row["uuid"]) == 36
    assert row["message_id"] == "m-1"
    assert row["conversation_id"] == "conv-1"
    assert row["input"][0]["role"] == "system"
    assert row["output"]["tool_calls"][0]["tool"] == "get_schedule"
    assert row["metadata"][0]["key"] == "gen_ai.request.model"
    assert row["created_at"].endswith("Z") and "T" in row["created_at"]

    by_uuid = db.get_trace(org, row["uuid"])
    assert by_uuid is not None and by_uuid["uuid"] == row["uuid"]


def test_create_always_inserts():
    org = _org()
    first = _ingest(org, "m-dup")
    second = _ingest(
        org, "m-dup", output={"response": "different retry body", "tool_calls": None}
    )
    assert second["uuid"] != first["uuid"]
    assert second["output"]["response"] == "different retry body"
    assert db.count_live_traces(org) == 2


def test_soft_delete_then_reingest():
    org = _org()
    row = _ingest(org, "m-free")
    assert db.soft_delete_traces(org, trace_ids=[row["uuid"]]) == 1
    assert db.get_trace(org, row["uuid"]) is None
    assert db.count_live_traces(org) == 0

    again = _ingest(org, "m-free")
    assert again["uuid"] != row["uuid"]


def test_list_and_pagination():
    org = _org()
    _ingest(org, "m-a", conversation_id="conv-a")
    _ingest(
        org,
        "m-b",
        conversation_id="conv-b",
        input=[{"role": "user", "content": "Tell me about POLIO boosters"}],
        output={"response": "Polio boosters are due at 16 months.", "tool_calls": None},
    )
    _ingest(org, "m-c", conversation_id="conv-b")

    rows, total = db.list_traces(org, limit=50, offset=0)
    assert total == 3
    # Newest first: same-second timestamps fall back to id descending.
    assert [r["message_id"] for r in rows] == ["m-c", "m-b", "m-a"]

    page, total = db.list_traces(org, limit=1, offset=1)
    assert total == 3
    assert [r["message_id"] for r in page] == ["m-b"]


def test_bulk_delete_contract():
    org = _org()
    a = _ingest(org, "m-1", conversation_id="conv-x")
    _ingest(org, "m-2", conversation_id="conv-y")
    _ingest(org, "m-3", conversation_id="conv-y")

    # An empty id list deletes nothing.
    assert db.soft_delete_traces(org, trace_ids=[]) == 0
    assert db.count_live_traces(org) == 3
    # Unknown ids are ignored, and only the named rows go.
    assert db.soft_delete_traces(org, trace_ids=[a["uuid"], "not-a-real-uuid"]) == 1
    assert db.count_live_traces(org) == 2
    # Already-deleted rows don't count a second time.
    assert db.soft_delete_traces(org, trace_ids=[a["uuid"]]) == 0


def test_bulk_delete_splits_large_id_lists():
    """SQLite caps bound values per statement, so the delete chunks. Without
    that, a big enough list raises "too many SQL variables"."""
    org = _org()
    ids = [_ingest(org, f"m-{i}")["uuid"] for i in range(3)]
    # More IDs than SQLite allows in one statement, mostly unknown ones.
    padded = ids + [str(uuid.uuid4()) for _ in range(33_000)]

    assert db.soft_delete_traces(org, trace_ids=padded) == 3
    assert db.list_traces(org, limit=10, offset=0)[1] == 0


def test_org_isolation():
    org_a, org_b = _org(), _org()
    row_a = _ingest(org_a, "m-shared")
    row_b = _ingest(org_b, "m-shared")

    # Same message_id in two workspaces is two independent traces.
    assert row_a["uuid"] != row_b["uuid"]

    assert db.get_trace(org_a, row_b["uuid"]) is None
    rows, total = db.list_traces(org_a, limit=50, offset=0)
    assert total == 1 and rows[0]["uuid"] == row_a["uuid"]
    # Deletes never cross workspaces even with explicit foreign ids.
    assert db.soft_delete_traces(org_a, trace_ids=[row_b["uuid"]]) == 0
    assert db.count_live_traces(org_b) == 1


def test_agent_id_roundtrips():
    org = _org()
    row = _ingest(org, "m-agent", agent_id="agent-x")
    assert row["agent_id"] == "agent-x"

    by_uuid = db.get_trace(org, row["uuid"])
    assert by_uuid is not None and by_uuid["agent_id"] == "agent-x"
    rows, _ = db.list_traces(org, limit=50, offset=0)
    assert rows[0]["agent_id"] == "agent-x"


def test_list_filters_by_agent_id():
    org = _org()
    _ingest(org, "m-x1", agent_id="agent-x")
    _ingest(org, "m-x2", agent_id="agent-x")
    _ingest(org, "m-y1", agent_id="agent-y")

    rows, total = db.list_traces(org, limit=50, offset=0, agent_id="agent-x")
    assert total == 2
    assert {r["message_id"] for r in rows} == {"m-x1", "m-x2"}

    rows, total = db.list_traces(org, limit=50, offset=0, agent_id="agent-y")
    assert total == 1 and rows[0]["message_id"] == "m-y1"


def test_reused_message_id_keeps_both_turns():
    """Matching on message_id once discarded a turn; every call must store one."""
    org = _org()
    first = _ingest(org, "m-same", output={"response": "first answer"})
    second = _ingest(org, "m-same", output={"response": "second answer"})

    assert second["uuid"] != first["uuid"]
    rows, total = db.list_traces(org, limit=50, offset=0)
    assert total == 2
    assert {r["output"]["response"] for r in rows} == {"first answer", "second answer"}


def test_same_message_id_on_two_agents_is_two_rows():
    org = _org()
    first = _ingest(org, "m-dup", agent_id="agent-x")
    second = _ingest(org, "m-dup", agent_id="agent-y")

    assert second["uuid"] != first["uuid"]
    assert second["agent_id"] == "agent-y"
    assert db.count_live_traces(org) == 2


def test_get_by_uuids_keeps_caller_order_and_dedupes():
    org = _org()
    a = _ingest(org, "m-a")
    b = _ingest(org, "m-b")
    c = _ingest(org, "m-c")

    asked = [c["uuid"], a["uuid"], b["uuid"], a["uuid"]]
    rows = db.get_traces_by_uuids(org, asked)
    assert [r["uuid"] for r in rows] == [c["uuid"], a["uuid"], b["uuid"]]


def test_get_by_uuids_omits_unknown_deleted_and_foreign():
    org, other = _org(), _org()
    live = _ingest(org, "m-live")
    gone = _ingest(org, "m-gone")
    assert db.soft_delete_traces(org, trace_ids=[gone["uuid"]]) == 1
    foreign = _ingest(other, "m-foreign")

    rows = db.get_traces_by_uuids(
        org, [live["uuid"], gone["uuid"], foreign["uuid"], str(uuid.uuid4())]
    )
    assert [r["uuid"] for r in rows] == [live["uuid"]]


def test_get_by_uuids_empty_skips_the_database(monkeypatch):
    def _boom():
        raise AssertionError("empty uuid list must not open a connection")

    monkeypatch.setattr(db, "get_db_connection", _boom)
    assert db.get_traces_by_uuids(_org(), []) == []


def test_get_by_uuids_returns_parsed_rows():
    org = _org()
    row = _ingest(org, "m-shape")

    fetched = db.get_traces_by_uuids(org, [row["uuid"]])[0]
    assert fetched == db.get_trace(org, row["uuid"])
    assert fetched["input"][0]["role"] == "system"
    assert fetched["output"]["tool_calls"][0]["tool"] == "get_schedule"
    assert fetched["metadata"][0]["key"] == "gen_ai.request.model"
    assert fetched["created_at"].endswith("Z") and "T" in fetched["created_at"]


def _label_notnull() -> dict:
    with db.get_db_connection() as conn:
        return {
            row["name"]: row["notnull"]
            for row in conn.execute("PRAGMA table_info(traces)").fetchall()
            if row["name"] in ("message_id", "conversation_id")
        }


def test_create_allows_null_labels():
    org = _org()
    row = db.create_trace(
        org_uuid=org,
        agent_id="agent-1",
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello"},
    )
    assert row["message_id"] is None
    assert row["conversation_id"] is None
    assert db.get_trace(org, row["uuid"])["message_id"] is None


def _insert_agent(org: str, *, interaction_type="conversation", config=None):
    agent_uuid = str(uuid.uuid4())
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO agents "
            "(uuid, org_uuid, name, config, interaction_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                agent_uuid,
                org,
                f"agent-{agent_uuid[:8]}",
                json.dumps(config if config is not None else {}),
                interaction_type,
            ),
        )
        conn.commit()
    return db.get_agent(agent_uuid)




def _eligible_evaluator(org: str, evaluator_type="llm"):
    ev = db.create_evaluator(
        name=f"eval-{uuid.uuid4().hex[:6]}",
        evaluator_type=evaluator_type,
        org_uuid=org,
        owner_user_id=str(uuid.uuid4()),
    )
    version = db.create_evaluator_version(ev, "openai/gpt-4.1", "Judge it.")
    db.set_evaluator_live_version(ev, version["uuid"])
    return ev, version["uuid"]


def _combined_ingest(
    org: str,
    agent: dict,
    max_scored_traces: int = 10_000,
    batch_size: int = 20,
    wait_seconds: int = 0,
    max_wait_seconds: int = 600,
    **overrides,
):
    payload = {
        "message_id": None,
        "conversation_id": "conv-1",
        "input": [{"role": "user", "content": "hi"}],
        "output": {"response": "hello", "tool_calls": None},
        "metadata": None,
    }
    payload.update(overrides)
    return db.create_trace_with_eval_run(
        org_uuid=org,
        agent=agent,
        max_scored_traces=max_scored_traces,
        batch_size=batch_size,
        wait_seconds=wait_seconds,
        max_wait_seconds=max_wait_seconds,
        **payload,
    )


def _runs_for(trace_uuid: str):
    with db.get_db_connection() as conn:
        return conn.execute(
            "SELECT * FROM trace_eval_runs WHERE trace_uuid = ?",
            (trace_uuid,),
        ).fetchall()


def test_ingest_writes_a_pending_run_with_no_plan_column():
    org = _org()
    agent = _insert_agent(org)
    ev, _version_id = _eligible_evaluator(org, "llm")
    db.add_evaluator_to_agent(agent["uuid"], ev)

    trace = _combined_ingest(org, agent)
    rows = _runs_for(trace["uuid"])
    assert len(rows) == 1
    run = rows[0]
    assert run["status"] == "pending"
    assert run["error"] is None
    assert run["completed_at"] is None
    assert run["org_uuid"] == org
    assert run["agent_id"] == agent["uuid"]
    assert "scoring_plan" not in run.keys()


def test_trace_and_run_roll_back_together(monkeypatch):
    org = _org()
    agent = _insert_agent(org)
    ev, _ = _eligible_evaluator(org, "llm")
    db.add_evaluator_to_agent(agent["uuid"], ev)

    def _boom(*_args, **_kwargs):
        raise sqlite3.IntegrityError("forced mid-transaction failure")

    monkeypatch.setattr(db, "_insert_trace_eval_run", _boom)

    with pytest.raises(sqlite3.IntegrityError):
        _combined_ingest(org, agent, message_id="m-atomic")

    with db.get_db_connection() as conn:
        trace_count = conn.execute(
            "SELECT COUNT(*) c FROM traces WHERE org_uuid = ?", (org,)
        ).fetchone()["c"]
        run_count = conn.execute(
            "SELECT COUNT(*) c FROM trace_eval_runs WHERE org_uuid = ?", (org,)
        ).fetchone()["c"]
    assert trace_count == 0
    assert run_count == 0


def test_create_trace_still_inserts_without_a_run():
    org = _org()
    agent = _insert_agent(org)
    ev, _ = _eligible_evaluator(org, "llm")
    db.add_evaluator_to_agent(agent["uuid"], ev)

    row = db.create_trace(
        org_uuid=org,
        agent_id=agent["uuid"],
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello"},
    )
    assert _runs_for(row["uuid"]) == []


def test_failed_runs_free_their_slot_in_the_cap():
    org = _org()
    agent = _insert_agent(org)
    ev, _ = _eligible_evaluator(org, "llm")
    db.add_evaluator_to_agent(agent["uuid"], ev)
    first = _combined_ingest(org, agent, max_scored_traces=1)
    run = _runs_for(first["uuid"])[0]
    assert run["status"] == "pending"
    with db.get_db_connection() as conn:
        conn.execute("UPDATE trace_eval_runs SET status = 'failed' WHERE uuid = ?", (run["uuid"],))
        conn.commit()

    second = _combined_ingest(org, agent, max_scored_traces=1)

    assert _runs_for(second["uuid"])[0]["status"] == "pending"


def test_deleting_traces_frees_their_share_of_the_scoring_cap():
    """A run counts against the cap through its trace, so a workspace that
    scored its allowance and deleted it can score again."""
    org = _org()
    agent = _insert_agent(org)
    ev, _ = _eligible_evaluator(org, "llm")
    db.add_evaluator_to_agent(agent["uuid"], ev)
    first = _combined_ingest(org, agent, max_scored_traces=1)
    assert _runs_for(first["uuid"])[0]["status"] == "pending"
    assert _combined_ingest(org, agent, max_scored_traces=1) and db.count_scored_traces(org) == 1

    db.soft_delete_traces(org, trace_ids=[first["uuid"]])

    assert db.count_scored_traces(org) == 0
    later = _combined_ingest(org, agent, max_scored_traces=1)
    assert _runs_for(later["uuid"])[0]["status"] == "pending"


def _waiting(agent_uuid: str):
    """This agent's unstarted runs, oldest first, as (uuid, available_at)."""
    with db.get_db_connection() as conn:
        return [
            (r["uuid"], r["available_at"])
            for r in conn.execute(
                "SELECT uuid, available_at FROM trace_eval_runs "
                "WHERE agent_id = ? AND status = 'pending' AND attempts = 0 "
                "ORDER BY created_at, id",
                (agent_uuid,),
            ).fetchall()
        ]


def test_a_new_trace_is_held_rather_than_judged_on_arrival():
    org = _org()
    agent = _insert_agent(org)
    _combined_ingest(org, agent, wait_seconds=120, max_wait_seconds=600)
    (_run, available_at), = _waiting(agent["uuid"])
    assert available_at > db.trace_scoring.utc_now()


def test_a_second_trace_restarts_the_wait_for_every_held_trace():
    org = _org()
    agent = _insert_agent(org)
    _combined_ingest(org, agent, wait_seconds=120, max_wait_seconds=600)

    # Both ingests land in the same second, so age the first run to make the
    # restart visible: a minute old, and five seconds from being judged.
    now = db.trace_scoring.utc_now()
    almost_due = db.trace_scoring.add_seconds(now, 5)
    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE trace_eval_runs SET created_at = ?, available_at = ? "
            "WHERE agent_id = ?",
            (db.trace_scoring.add_seconds(now, -60), almost_due, agent["uuid"]),
        )
        conn.commit()
    _combined_ingest(org, agent, wait_seconds=120, max_wait_seconds=600)

    held = _waiting(agent["uuid"])
    assert len(held) == 2
    assert {available_at for _uuid, available_at in held} == {held[0][1]}
    assert held[0][1] > almost_due


def test_reaching_the_batch_size_releases_every_held_trace():
    org = _org()
    agent = _insert_agent(org)
    _combined_ingest(org, agent, wait_seconds=120, batch_size=2)
    assert _waiting(agent["uuid"])[0][1] > db.trace_scoring.utc_now()

    _combined_ingest(org, agent, wait_seconds=120, batch_size=2)
    now = db.trace_scoring.utc_now()
    assert [available_at <= now for _uuid, available_at in _waiting(agent["uuid"])] == [
        True,
        True,
    ]


def test_the_oldest_trace_is_never_held_past_the_longest_wait():
    org = _org()
    agent = _insert_agent(org)
    _combined_ingest(org, agent, wait_seconds=120, max_wait_seconds=600)

    # Nine minutes of arrivals have already restarted the wait.
    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE trace_eval_runs SET created_at = ? WHERE agent_id = ?",
            (
                db.trace_scoring.add_seconds(db.trace_scoring.utc_now(), -9 * 60),
                agent["uuid"],
            ),
        )
        conn.commit()
    _combined_ingest(org, agent, wait_seconds=120, max_wait_seconds=600)

    now = db.trace_scoring.utc_now()
    # The restart alone would give two minutes; the longest wait leaves one.
    assert _waiting(agent["uuid"])[0][1] <= db.trace_scoring.add_seconds(now, 61)


def test_a_wait_of_zero_judges_on_arrival():
    org = _org()
    agent = _insert_agent(org)
    _combined_ingest(org, agent, wait_seconds=0)
    assert _waiting(agent["uuid"])[0][1] <= db.trace_scoring.utc_now()


def test_a_trace_never_moves_another_agents_held_traces():
    org = _org()
    first = _insert_agent(org)
    second = _insert_agent(org)
    _combined_ingest(org, first, wait_seconds=120)
    held = _waiting(first["uuid"])[0][1]

    _combined_ingest(org, second, wait_seconds=0)
    assert _waiting(first["uuid"])[0][1] == held


def test_a_trace_leaves_a_run_waiting_out_its_retry_alone():
    org = _org()
    agent = _insert_agent(org)
    _combined_ingest(org, agent, wait_seconds=0)
    retry_at = db.trace_scoring.add_seconds(db.trace_scoring.utc_now(), 3600)
    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE trace_eval_runs SET attempts = 1, available_at = ? WHERE agent_id = ?",
            (retry_at, agent["uuid"]),
        )
        conn.commit()

    _combined_ingest(org, agent, wait_seconds=0)
    with db.get_db_connection() as conn:
        rows = conn.execute(
            "SELECT available_at FROM trace_eval_runs "
            "WHERE agent_id = ? AND attempts = 1",
            (agent["uuid"],),
        ).fetchall()
    assert [r["available_at"] for r in rows] == [retry_at]


def test_a_new_trace_leaves_already_released_traces_alone():
    org = _org()
    agent = _insert_agent(org)
    _combined_ingest(org, agent, wait_seconds=0)
    released = _waiting(agent["uuid"])[0][1]

    _combined_ingest(org, agent, wait_seconds=120)
    held = dict(_waiting(agent["uuid"]))
    assert held.pop(_waiting(agent["uuid"])[0][0]) == released
    # Only the arriving trace is held; the released one keeps its time.
    assert all(available_at > db.trace_scoring.utc_now() for available_at in held.values())


def test_score_filter_refuses_an_operator_it_does_not_know():
    """Operators are interpolated into SQL rather than bound, so anything
    outside the known set must be refused instead of passed through."""
    with pytest.raises(ValueError):
        db._trace_score_clause(("ev-1", "; DROP TABLE traces", 1))
