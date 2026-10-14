"""Provider key encryption, run environment, the key check, and their DB rows."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

import db
import provider_keys as pk
import trace_scoring as ts

GOOGLE_JSON = json.dumps(
    {"project_id": "proj-1", "client_email": "bot@proj-1.iam.gserviceaccount.com"}
)


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("PROVIDER_KEYS_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.delenv("FAKE_AI_PROVIDERS", raising=False)


def _org() -> str:
    return str(uuid.uuid4())


def _save(org, provider, value):
    db.upsert_org_provider_key(
        org, provider, pk.encrypt(value), pk.masked_value(provider, value), "u1"
    )


def test_encrypt_round_trip_and_configured(monkeypatch):
    token = pk.encrypt("sk-secret")
    assert token != "sk-secret"
    assert pk.decrypt(token) == "sk-secret"
    assert pk.encryption_configured()
    monkeypatch.delenv("PROVIDER_KEYS_ENCRYPTION_KEY")
    assert not pk.encryption_configured()


def test_masked_values():
    assert pk.masked_value("openai", "sk-abcdef1234") == "••••1234"
    assert pk.masked_value("google", GOOGLE_JSON) == "bot@proj-1.iam.gserviceaccount.com"


@pytest.mark.parametrize(
    "value", ["not json", "[1]", json.dumps({"project_id": "p"}), json.dumps({"client_email": "e"})]
)
def test_parse_google_credentials_rejects(value):
    assert pk.parse_google_credentials(value) is None


def test_provider_env_none_without_keys():
    org = _org()
    assert pk.provider_env(org) is None
    assert not pk.org_uses_own_keys(org)


def test_provider_env_strips_server_keys_and_adds_org_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "server-openai")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "server-deepgram")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/server/creds.json")
    org = _org()
    _save(org, "openai", "org-openai")
    _save(org, "google", GOOGLE_JSON)
    assert pk.org_uses_own_keys(org)

    env = pk.provider_env(org)

    assert env["OPENAI_API_KEY"] == "org-openai"
    assert "DEEPGRAM_API_KEY" not in env
    assert env["GOOGLE_CLOUD_PROJECT_ID"] == "proj-1"
    path = env["GOOGLE_APPLICATION_CREDENTIALS"]
    assert path == pk.google_creds_path(org)
    with open(path) as f:
        assert f.read() == GOOGLE_JSON
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode) == 0o700
    assert env["PATH"] == os.environ["PATH"]
    os.remove(path)


def test_provider_env_unreadable_raises(monkeypatch):
    org = _org()
    _save(org, "openai", "org-openai")
    monkeypatch.setenv("PROVIDER_KEYS_ENCRYPTION_KEY", Fernet.generate_key().decode())
    with pytest.raises(RuntimeError, match="cannot be read"):
        pk.provider_env(org)
    monkeypatch.delenv("PROVIDER_KEYS_ENCRYPTION_KEY")
    with pytest.raises(RuntimeError, match="cannot be read"):
        pk.provider_env(org)


def _process(stdout: bytes):
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, b""))
    proc.wait = AsyncMock()
    return proc


def _events(provider, result):
    return json.dumps({"type": "result", "provider": provider, "result": result}).encode()


def test_check_provider_key_ok_passes_only_that_key(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "server-deepgram")
    proc = _process(_events("openai", {"status": "pass"}))
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn:
        assert asyncio.run(pk.check_provider_key("openai", "sk-1")) is None
    kwargs = spawn.call_args.kwargs
    assert spawn.call_args.args[1] == "status"
    assert kwargs["env"]["OPENAI_API_KEY"] == "sk-1"
    assert "DEEPGRAM_API_KEY" not in kwargs["env"]
    assert kwargs["cwd"] != os.getcwd()


def test_check_provider_key_google_writes_creds_into_temp_folder():
    proc = _process(_events("google", {"status": "pass"}))
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn:
        assert asyncio.run(pk.check_provider_key("google", GOOGLE_JSON)) is None
    env = spawn.call_args.kwargs["env"]
    assert env["GOOGLE_APPLICATION_CREDENTIALS"].startswith(spawn.call_args.kwargs["cwd"])
    assert not os.path.exists(env["GOOGLE_APPLICATION_CREDENTIALS"])


def test_check_provider_key_reports_error():
    proc = _process(_events("openai", {"status": "fail", "error": "401 Unauthorized"}))
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        assert asyncio.run(pk.check_provider_key("openai", "bad")) == "401 Unauthorized"


def test_check_provider_key_unparseable_output():
    proc = _process(b"garbage")
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        assert asyncio.run(pk.check_provider_key("openai", "x")) == "Could not check the openai key."


def test_check_provider_key_timeout(monkeypatch):
    proc = _process(b"")
    monkeypatch.setattr(pk, "CHECK_TIMEOUT_SECONDS", 0.01)

    async def slow():
        await asyncio.sleep(1)

    proc.communicate = slow
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        assert asyncio.run(pk.check_provider_key("openai", "x")) == (
            "No answer from openai within 0.01 seconds."
        )
    proc.kill.assert_called_once()


def test_check_provider_key_fake_mode_never_spawns(monkeypatch):
    monkeypatch.setenv("FAKE_AI_PROVIDERS", "1")
    with patch("asyncio.create_subprocess_exec") as spawn:
        assert asyncio.run(pk.check_provider_key("openai", "x")) is None
    spawn.assert_not_called()


def test_db_upsert_list_delete():
    org = _org()
    _save(org, "openai", "first")
    _save(org, "openai", "second1234")
    rows = db.list_org_provider_keys(org)
    assert len(rows) == 1
    assert rows[0]["masked_value"] == "••••1234"
    assert db.delete_org_provider_key(org, "openai")
    assert not db.delete_org_provider_key(org, "openai")
    assert not db.org_has_provider_keys(org)


def _insert(sql, params):
    with db.get_db_connection() as conn:
        conn.execute(sql, params)
        conn.commit()


def test_org_has_active_jobs_per_job_table():
    org = _org()
    assert not db.org_has_active_jobs(org)

    job = str(uuid.uuid4())
    _insert("INSERT INTO jobs (uuid, type, status, org_uuid) VALUES (?, 'stt-eval', 'queued', ?)", (job, org))
    assert db.org_has_active_jobs(org)
    _insert("UPDATE jobs SET status = 'done' WHERE uuid = ?", (job,))
    assert not db.org_has_active_jobs(org)

    agent = str(uuid.uuid4())
    _insert("INSERT INTO agents (uuid, name, org_uuid) VALUES (?, ?, ?)", (agent, agent, org))
    agent_job = str(uuid.uuid4())
    _insert(
        "INSERT INTO agent_test_jobs (uuid, agent_id, type, status) VALUES (?, ?, 'llm-unit-test', 'in_progress')",
        (agent_job, agent),
    )
    assert db.org_has_active_jobs(org)
    # The test database is shared, so an open job would be picked up by queue tests that run later.
    _insert("UPDATE agent_test_jobs SET status = 'done' WHERE uuid = ?", (agent_job,))

    org2 = _org()
    sim = str(uuid.uuid4())
    _insert("INSERT INTO simulations (uuid, name, org_uuid) VALUES (?, ?, ?)", (sim, sim, org2))
    sim_job = str(uuid.uuid4())
    _insert(
        "INSERT INTO simulation_jobs (uuid, simulation_id, type, status) VALUES (?, ?, 'text', 'queued')",
        (sim_job, sim),
    )
    assert db.org_has_active_jobs(org2)
    _insert("UPDATE simulation_jobs SET status = 'done' WHERE uuid = ?", (sim_job,))
    assert not db.org_has_active_jobs(org2)


def _trace_run(org, status, created_offset):
    trace = db.create_trace(
        org_uuid=org, agent_id="agent-1", input=[{"role": "user", "content": "hi"}], output={"response": "x"}
    )
    run = str(uuid.uuid4())
    stamp = ts.add_seconds("2026-01-01 00:00:00", created_offset)
    _insert(
        "INSERT INTO trace_eval_runs (uuid, trace_uuid, org_uuid, agent_id, status, available_at, "
        "created_at, updated_at) VALUES (?, ?, ?, 'agent-1', ?, ?, ?, ?)",
        (run, trace["uuid"], org, status, stamp, stamp, stamp),
    )
    return run


def _status(run):
    with db.get_db_connection() as conn:
        return tuple(
            conn.execute("SELECT status, error FROM trace_eval_runs WHERE uuid = ?", (run,)).fetchone()
        )


def test_trim_pending_trace_runs_keeps_oldest_within_cap():
    org = _org()
    _trace_run(org, "completed", 0)
    _trace_run(org, "processing", 1)
    pending = [_trace_run(org, "pending", 10 + i) for i in range(4)]

    assert db.trim_pending_trace_runs_over_cap(org, 4) == 2

    assert _status(pending[0]) == ("pending", None)
    assert _status(pending[1]) == ("pending", None)
    assert _status(pending[2]) == ("skipped", "over_limit")
    assert _status(pending[3]) == ("skipped", "over_limit")


def test_trim_pending_trace_runs_cap_below_used_skips_all():
    org = _org()
    _trace_run(org, "completed", 0)
    _trace_run(org, "completed", 1)
    run = _trace_run(org, "pending", 2)
    assert db.trim_pending_trace_runs_over_cap(org, 1) == 1
    assert _status(run) == ("skipped", "over_limit")


def test_create_trace_with_eval_run_no_cap_never_skips():
    org = _org()
    agent = {"uuid": "agent-1", "config": {}}
    for _ in range(2):
        row = db.create_trace_with_eval_run(
            org_uuid=org,
            agent=agent,
            input=[{"role": "user", "content": "hi"}],
            output={"response": "x"},
            max_scored_traces=None,
            batch_size=20,
        )
    with db.get_db_connection() as conn:
        statuses = [
            r[0] for r in conn.execute("SELECT status FROM trace_eval_runs WHERE org_uuid = ?", (org,))
        ]
    assert statuses == ["pending", "pending"]
    assert row["uuid"]
