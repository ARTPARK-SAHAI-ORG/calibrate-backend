"""Trace CRUD tests for the traces helpers in src/db.py."""

from __future__ import annotations

import sqlite3
import uuid

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


def test_init_db_rebuilds_legacy_not_null_label_columns():
    kept_uuid = str(uuid.uuid4())
    with db.get_db_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS traces")
        conn.execute(
            """
            CREATE TABLE traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                org_uuid TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                input TEXT NOT NULL,
                output TEXT NOT NULL,
                metadata TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO traces
                (uuid, org_uuid, agent_id, message_id, conversation_id, input, output)
            VALUES (?, 'org-legacy', 'agent-legacy', 'm-keep', 'c-keep', '[]', '{}')
            """,
            (kept_uuid,),
        )
        conn.execute(
            "DELETE FROM _schema_migrations WHERE name = ?",
            (db.TRACES_NULLABLE_IDS_MIGRATION,),
        )
        conn.commit()

    assert _label_notnull() == {"message_id": 1, "conversation_id": 1}

    db.init_db()

    assert _label_notnull() == {"message_id": 0, "conversation_id": 0}
    kept = db.get_trace("org-legacy", kept_uuid)
    assert kept is not None
    assert kept["message_id"] == "m-keep"
    row = db.create_trace(
        org_uuid="org-legacy",
        agent_id="agent-legacy",
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello"},
    )
    assert row["message_id"] is None
    assert row["conversation_id"] is None


# The pre-rebuild shape: both label columns were required.
_LEGACY_TRACES_TABLE_BODY = """
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT NOT NULL UNIQUE,
    org_uuid TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    input TEXT NOT NULL,
    output TEXT NOT NULL,
    metadata TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP DEFAULT NULL
"""


def _throwaway_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _seed_trace(conn: sqlite3.Connection, table: str, message_id: str) -> None:
    conn.execute(
        f"INSERT INTO {table} "
        "(uuid, org_uuid, agent_id, message_id, conversation_id, input, output) "
        "VALUES (?, 'org-legacy', 'agent-legacy', ?, 'c-keep', '[]', '{}')",
        (str(uuid.uuid4()), message_id),
    )


def _table_names(conn: sqlite3.Connection) -> set:
    return {
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _trace_message_ids(conn: sqlite3.Connection) -> set:
    return {r["message_id"] for r in conn.execute("SELECT message_id FROM traces")}


def _insert_trace_without_labels(conn: sqlite3.Connection) -> None:
    """Raises sqlite3.IntegrityError unless the rebuilt table allows NULL labels."""
    conn.execute(
        "INSERT INTO traces "
        "(uuid, org_uuid, agent_id, message_id, conversation_id, input, output) "
        "VALUES (?, 'org-legacy', 'agent-legacy', NULL, NULL, '[]', '{}')",
        (str(uuid.uuid4()),),
    )


def test_rebuild_recovers_when_both_traces_and_traces_new_exist():
    """Crash after the create, before the drop: redo the copy, lose no row."""
    conn = _throwaway_db()
    conn.execute(f"CREATE TABLE traces ({_LEGACY_TRACES_TABLE_BODY})")
    _seed_trace(conn, "traces", "m-keep-1")
    _seed_trace(conn, "traces", "m-keep-2")
    # Half-finished copy left by the crashed run: only one row made it across.
    conn.execute(f"CREATE TABLE traces_new ({db._TRACES_TABLE_BODY})")
    _seed_trace(conn, "traces_new", "m-keep-1")

    assert db._rebuild_traces_for_nullable_ids(conn.cursor()) is True

    assert "traces_new" not in _table_names(conn)
    assert _trace_message_ids(conn) == {"m-keep-1", "m-keep-2"}
    _insert_trace_without_labels(conn)


def test_rebuild_recovers_when_only_traces_new_exists():
    """Crash after the drop, before the rename: the copy holds the only rows.

    Dropping `traces_new` here instead of renaming it erases every trace in the
    database, so this asserts on row content, not just on a table existing.
    """
    conn = _throwaway_db()
    conn.execute(f"CREATE TABLE traces_new ({db._TRACES_TABLE_BODY})")
    _seed_trace(conn, "traces_new", "m-keep-1")
    _seed_trace(conn, "traces_new", "m-keep-2")

    db._rebuild_traces_for_nullable_ids(conn.cursor())

    assert _trace_message_ids(conn) == {"m-keep-1", "m-keep-2"}
    assert "traces_new" not in _table_names(conn)
    _insert_trace_without_labels(conn)
