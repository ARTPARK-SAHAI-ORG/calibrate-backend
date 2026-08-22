"""Integration tests for /agent-tests."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from conftest import NONEXISTENT_UUID, NONEXISTENT_UUID_2

@pytest.fixture(scope="module")
def app():
    import main as main_mod

    return main_mod.app


@pytest.fixture(scope="module")
def client(app):
    with patch("main.recover_pending_jobs"):
        with TestClient(app) as c:
            yield c


def _signup(client):
    suffix = uuid.uuid4().hex[:8]
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "AT",
            "last_name": "U",
            "email": f"at-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
        "user_uuid": body["user"]["uuid"],
    }


def _create_agent(client, h, name=None):
    return client.post(
        "/agents",
        json={"name": name or f"a-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=h,
    ).json()


def _create_test(client, h, name=None):
    evaluators = client.get("/evaluators", headers=h).json()["items"]
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")
    return client.post(
        "/tests",
        json={
            "name": name or f"t-{uuid.uuid4().hex[:6]}",
            "type": "response",
            "config": {"history": [], "evaluation": {"type": "response"}},
            "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
        },
        headers=h,
    ).json()


def _create_simulation_evaluator(client, h, name=None):
    """Create a simulation-type evaluator (no default is seeded for this type)."""
    return client.post(
        "/evaluators",
        json={
            "name": name or f"sim-ev-{uuid.uuid4().hex[:6]}",
            "evaluator_type": "conversation",
            "output_type": "binary",
            "version": {
                "judge_model": "openai/gpt-4.1",
                "system_prompt": "Judge the whole conversation.",
            },
        },
        headers=h,
    ).json()


def _create_general_evaluator(client, h, name=None):
    """Create an llm-general-type evaluator (no default is seeded for this type)."""
    return client.post(
        "/evaluators",
        json={
            "name": name or f"gen-ev-{uuid.uuid4().hex[:6]}",
            "evaluator_type": "llm-general",
            "output_type": "binary",
            "version": {
                "judge_model": "openai/gpt-4.1",
                "system_prompt": "Judge the input/output pair.",
            },
        },
        headers=h,
    ).json()


def _create_general_test(client, h, name=None, gen_ev_uuid=None):
    if gen_ev_uuid is None:
        gen_ev_uuid = _create_general_evaluator(client, h)["uuid"]
    return client.post(
        "/tests",
        json={
            "name": name or f"gen-{uuid.uuid4().hex[:6]}",
            "type": "general",
            "config": {
                "input": "Summarize this article: ...",
                "evaluation": {"type": "general"},
            },
            "evaluators": [{"evaluator_uuid": gen_ev_uuid}],
        },
        headers=h,
    ).json()


def _create_conversation_test(client, h, name=None, sim_ev_uuid=None):
    if sim_ev_uuid is None:
        sim_ev_uuid = _create_simulation_evaluator(client, h)["uuid"]
    return client.post(
        "/tests",
        json={
            "name": name or f"conv-{uuid.uuid4().hex[:6]}",
            "type": "conversation",
            "config": {
                "history": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                "evaluation": {"type": "conversation"},
            },
            "evaluators": [{"evaluator_uuid": sim_ev_uuid}],
        },
        headers=h,
    ).json()


# ---------------------------------------------------------------------------
# Link CRUD
# ---------------------------------------------------------------------------


def test_agent_tests_link_crud(client):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    test_a = _create_test(client, h)
    test_b = _create_test(client, h)

    link = client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test_a["uuid"]]},
        headers=h,
    )
    assert link.status_code == 200
    # Re-link (idempotent — skip already linked)
    again = client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test_a["uuid"], test_b["uuid"]]},
        headers=h,
    )
    assert again.status_code == 200

    # List
    assert client.get("/agent-tests").status_code == 200
    assert (
        client.get(
            f"/agent-tests/agent/{agent['uuid']}/tests", headers=h
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/agent-tests/test/{test_a['uuid']}/agents", headers=h
        ).status_code
        == 200
    )
    # The route is org-scoped now, so it rejects an anonymous caller.
    assert client.get(f"/agent-tests/test/{test_a['uuid']}/agents").status_code == 403
    assert (
        client.get("/agent-tests/test/missing/agents", headers=h).status_code == 404
    )
    assert (
        client.get("/agent-tests/agent/missing/tests", headers=h).status_code == 404
    )
    assert (
        client.get("/agent-tests/agent/missing/runs", headers=h).status_code == 404
    )

    # Runs list (no runs yet) — paginated envelope.
    runs = client.get(f"/agent-tests/agent/{agent['uuid']}/runs", headers=h)
    assert runs.status_code == 200
    assert runs.json()["items"] == []

    # Global runs list (auth required)
    global_runs = client.get("/agent-tests/runs", headers=h)
    assert global_runs.status_code == 200

    # Filtered
    global_runs2 = client.get(
        "/agent-tests/runs", params={"type": "llm-unit-test"}, headers=h
    )
    assert global_runs2.status_code == 200

    # Bulk-unlink validation
    empty = client.post(
        "/agent-tests/bulk-unlink",
        json={"agent_uuid": agent["uuid"], "test_uuids": []},
    )
    assert empty.status_code == 400
    bulk_unlink = client.post(
        "/agent-tests/bulk-unlink",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test_a["uuid"]]},
    )
    assert bulk_unlink.status_code == 200

    # Bulk-unlink missing agent
    missing = client.post(
        "/agent-tests/bulk-unlink",
        json={"agent_uuid": NONEXISTENT_UUID, "test_uuids": [test_b["uuid"]]},
    )
    assert missing.status_code == 404

    # Bulk-delete-tests
    bulk_del = client.post(
        "/agent-tests/bulk-delete-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test_b["uuid"]]},
        headers=h,
    )
    assert bulk_del.status_code == 200

    # Bulk-delete with empty
    empty_del = client.post(
        "/agent-tests/bulk-delete-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": []},
        headers=h,
    )
    assert empty_del.status_code == 400

    # Bulk-delete with missing agent
    missing_del = client.post(
        "/agent-tests/bulk-delete-tests",
        json={"agent_uuid": NONEXISTENT_UUID, "test_uuids": [NONEXISTENT_UUID_2]},
        headers=h,
    )
    assert missing_del.status_code == 404

    # Bulk-delete with foreign agent → 404
    other = _signup(client)
    foreign = client.post(
        "/agent-tests/bulk-delete-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test_b["uuid"]]},
        headers=other["headers"],
    )
    assert foreign.status_code == 404


def test_agent_tests_list_returns_trimmed_shape(client):
    """GET /agent-tests/agent/{uuid}/tests returns the trimmed list shape:
    uuid/name/type only, with `config.history`/`evaluation` and the hydrated
    `evaluators` list dropped from each item."""
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    # _create_test links an llm evaluator and sets config.history + evaluation,
    # so a full shape would carry both — proving the list drops them.
    name = f"t-trim-{uuid.uuid4().hex[:6]}"
    test = _create_test(client, h, name=name)
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )

    r = client.get(f"/agent-tests/agent/{agent['uuid']}/tests", headers=h)
    assert r.status_code == 200, r.text
    item = next(t for t in r.json()["items"] if t["uuid"] == test["uuid"])
    assert item["name"] == name
    assert item["type"] == "response"
    assert "evaluators" not in item
    # config carries description only (None here), never history/evaluation.
    assert "history" not in (item.get("config") or {})
    assert "evaluation" not in (item.get("config") or {})


def test_agent_tests_list_never_ships_heavy_config_blocks(client):
    """The per-agent tests list uses the slim summary (json_extract of
    `config.description` only). A linked test carrying heavy
    `history`/`evaluation`/`settings` blocks stuffed with sentinels must expose
    none of them through `GET /agent-tests/agent/{uuid}/tests`."""
    import json as _json

    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    hist_sentinel = f"HIST-{uuid.uuid4().hex}"
    eval_sentinel = f"EVAL-{uuid.uuid4().hex}"
    settings_sentinel = f"SET-{uuid.uuid4().hex}"
    name = f"t-heavy-{uuid.uuid4().hex[:6]}"
    test = client.post(
        "/tests",
        json={
            "name": name,
            "type": "response",
            "config": {
                "description": "keep me",
                "history": [{"role": "user", "content": hist_sentinel}],
                "evaluation": {"type": "response", "note": eval_sentinel},
                "settings": {"language": settings_sentinel},
            },
        },
        headers=h,
    ).json()
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )

    r = client.get(f"/agent-tests/agent/{agent['uuid']}/tests", headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"items", "total", "limit", "offset"}
    item = next(t for t in body["items"] if t["uuid"] == test["uuid"])
    assert item["name"] == name
    assert item["type"] == "response"
    assert item["config"] == {"description": "keep me"}
    assert "evaluators" not in item

    dumped = _json.dumps(body)
    assert hist_sentinel not in dumped
    assert eval_sentinel not in dumped
    assert settings_sentinel not in dumped


def test_agent_tests_list_null_description(client):
    """A linked test with no `config.description` still lists 200 with
    description=null through the per-agent tests endpoint."""
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    test = client.post(
        "/tests",
        json={
            "name": f"t-nodesc-{uuid.uuid4().hex[:6]}",
            "type": "response",
            "config": {"history": [{"role": "user", "content": "hi"}]},
        },
        headers=h,
    ).json()
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )

    r = client.get(f"/agent-tests/agent/{agent['uuid']}/tests", headers=h)
    assert r.status_code == 200, r.text
    item = next(t for t in r.json()["items"] if t["uuid"] == test["uuid"])
    assert item["config"] == {"description": None}


def test_agent_runs_list_surfaces_perf_aggregates(client):
    """A completed unit-test run surfaces run-level latency/cost/total_tokens
    aggregates through the agent runs-list endpoint. The list is a lightweight
    index: per-case rows are slimmed to flat `{name, passed}` (no per-case
    output/latency/cost/test_case) — those live on the run-detail endpoint.
    Covers the shared `_build_agent_test_run_item_fields` mapping used by both
    list endpoints."""
    from db import create_agent_test_job, update_agent_test_job

    h = _signup(client)["headers"]
    agent = _create_agent(client, h)

    job_id = create_agent_test_job(agent_id=agent["uuid"], job_type="llm-unit-test")
    update_agent_test_job(
        job_id,
        status="done",
        results={
            "total_tests": 1,
            "passed": 1,
            "failed": 0,
            "latency_ms": {"p50": 1851.0, "p95": 1851.0, "p99": 1851.0, "count": 1},
            "cost": {"mean": 0.0248, "min": 0.0248, "max": 0.0248, "count": 1},
            "total_tokens": {"mean": 4378.0, "min": 4369, "max": 4387, "count": 2},
            "test_results": [
                {
                    "name": "tc1",
                    "test_case_id": "tc1",
                    "passed": True,
                    "output": {"response": "hi", "tool_calls": None},
                    "latency_ms": 1851.0,
                    "cost": 0.0248,
                }
            ],
        },
    )

    resp = client.get(f"/agent-tests/agent/{agent['uuid']}/runs", headers=h)
    assert resp.status_code == 200
    run = resp.json()["items"][0]
    # Run-level aggregates flow through.
    assert run["latency_ms"] == {"p50": 1851.0, "p95": 1851.0, "p99": 1851.0, "count": 1}
    assert run["cost"]["mean"] == 0.0248
    assert run["total_tokens"] == {"mean": 4378.0, "min": 4369, "max": 4387, "count": 2}
    # Per-case rows are slim: name + passed only, nothing else.
    assert run["results"] == [{"name": "tc1", "passed": True}]

    # Same aggregates flow through the global runs-list endpoint (paginated
    # envelope, same as the per-agent one).
    global_resp = client.get("/agent-tests/runs", headers=h)
    assert global_resp.status_code == 200
    gruns = [r for r in global_resp.json()["items"] if r["uuid"] == job_id]
    assert gruns and gruns[0]["total_tokens"]["count"] == 2
    assert gruns[0]["results"] == [{"name": "tc1", "passed": True}]


def test_agent_runs_list_slims_benchmark_model_results(client):
    """A benchmark run in the runs-list drops each model's heavy nested
    `test_results`/`evaluator_summary`, keeping only flat scalar summaries. The
    empty/non-dict-row guards live in `_slim_*` and are unit-tested separately
    (`test_slim_run_list_helpers_guard_edge_cases`); the SQL-side summary path
    only ever sees calibrate's dict rows, so this integration case seeds those."""
    from db import create_agent_test_job, update_agent_test_job

    h = _signup(client)["headers"]
    agent = _create_agent(client, h)

    job_id = create_agent_test_job(agent_id=agent["uuid"], job_type="llm-benchmark")
    update_agent_test_job(
        job_id,
        status="done",
        results={
            # No `test_results` block on a benchmark run → collapses to None.
            "leaderboard_summary": [{"model": "openai/gpt-4.1", "rank": 1}],
            "model_results": [
                {
                    "model": "openai/gpt-4.1",
                    "success": True,
                    "message": "ok",
                    "total_tests": 2,
                    "passed": 2,
                    "failed": 0,
                    # Heavy nested detail that must be dropped from the list.
                    "test_results": [
                        {"name": "tc1", "passed": True, "output": {"response": "hi"}}
                    ],
                    "evaluator_summary": [{"name": "correctness", "pass_rate": 1.0}],
                    "latency_ms": {"p50": 10.0},
                },
            ],
        },
    )

    resp = client.get(f"/agent-tests/agent/{agent['uuid']}/runs", headers=h)
    assert resp.status_code == 200
    run = resp.json()["items"][0]
    # A benchmark run has no unit-test `test_results` → the slim rows are None.
    assert run["results"] is None
    # Model row is slimmed to flat scalars; heavy nested fields are gone.
    assert run["model_results"] == [
        {
            "model": "openai/gpt-4.1",
            "success": True,
            "message": "ok",
            "total_tests": 2,
            "passed": 2,
            "failed": 0,
        }
    ]
    # Leaderboard + the heavy rubric block are not part of the list item at all;
    # the flat `evaluators` name column is empty since this job has no
    # `details.evaluators_by_test_id` snapshot.
    assert "leaderboard_summary" not in run
    assert run["evaluators"] == []


def _create_llm_evaluator(client, h, name):
    return client.post(
        "/evaluators",
        json={
            "name": name,
            "evaluator_type": "llm",
            "output_type": "binary",
            "version": {
                "judge_model": "openai/gpt-4.1",
                "system_prompt": "Judge the reply.",
            },
        },
        headers=h,
    ).json()


def test_agent_runs_list_evaluators_column(client):
    """The `evaluators` column on a run-list item lists the CURRENT name of
    each evaluator that judged the run (not the frozen submission-time
    snapshot, which can be stale after a rename or carry calibrate's
    `-{uuid8}` name-collision suffix), deduplicated and in first-appearance
    order, with `Tool call` appended when any linked test was a tool-call
    test — the tool-call test itself never carries an evaluator, so this is
    the only signal that it ran. An evaluator uuid the snapshot references
    but that no longer resolves (e.g. deleted) falls back to the snapshot
    name rather than dropping the entry."""
    from db import (
        create_agent_test_job,
        create_test,
        get_personal_org_for_user,
        update_agent_test_job,
    )

    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    org_uuid = get_personal_org_for_user(auth["user_uuid"])["uuid"]

    live_name_a = f"Live Correctness {uuid.uuid4().hex[:6]}"
    live_name_b = f"Live Script Fidelity {uuid.uuid4().hex[:6]}"
    ev_a = _create_llm_evaluator(client, h, live_name_a)
    ev_b = _create_llm_evaluator(client, h, live_name_b)

    tool_test_uuid = create_test(
        name="uses-tool",
        type="tool_call",
        org_uuid=org_uuid,
        config={
            "history": [{"role": "user", "content": "hi"}],
            "evaluation": {
                "type": "tool_call",
                "tool_calls": [{"tool": "x", "accept_any_arguments": True}],
            },
        },
    )

    job_id = create_agent_test_job(
        agent_id=agent["uuid"],
        job_type="llm-unit-test",
        details={
            "test_uuids": ["tc1", "tc2", tool_test_uuid],
            "evaluators_by_test_id": {
                "tc1": [
                    # Snapshot name is stale/suffixed (as calibrate would write
                    # it on a name collision) — the live DB name must win.
                    {"uuid": ev_a["uuid"], "name": f"{live_name_a}-a1b2c3d4"},
                    {"uuid": ev_b["uuid"], "name": live_name_b},
                    # Snapshot references an evaluator that no longer resolves
                    # — falls back to the snapshot name instead of vanishing.
                    {"uuid": NONEXISTENT_UUID, "name": "Deleted Evaluator"},
                ],
                # ev_a repeats under its live name — must be deduped, not
                # re-listed, even though the snapshot entries differ in uuid
                # casing/name across tests.
                "tc2": [{"uuid": ev_a["uuid"], "name": f"{live_name_a}-a1b2c3d4"}],
            },
            "has_tool_call_test": True,
        },
    )
    update_agent_test_job(job_id, status="done", results={"total_tests": 3, "passed": 3, "failed": 0})

    resp = client.get(f"/agent-tests/agent/{agent['uuid']}/runs", headers=h)
    assert resp.status_code == 200
    run = resp.json()["items"][0]
    assert run["evaluators"] == [
        live_name_a,
        live_name_b,
        "Deleted Evaluator",
        "Tool call",
    ]


def test_agent_runs_list_has_tool_call_test_is_snapshotted_not_derived(client):
    """`has_tool_call_test` is read straight from the job's `details` snapshot
    (set once at job creation) rather than re-derived from the current state
    of the linked tests, so a job predating this field, or with no snapshot
    at all, shows no `Tool call` entry even if its `test_uuids` happen to
    reference a tool-call test."""
    from db import create_agent_test_job, create_test, get_personal_org_for_user, update_agent_test_job

    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    org_uuid = get_personal_org_for_user(auth["user_uuid"])["uuid"]

    tool_test_uuid = create_test(
        name="uses-tool-2",
        type="tool_call",
        org_uuid=org_uuid,
        config={
            "history": [{"role": "user", "content": "hi"}],
            "evaluation": {
                "type": "tool_call",
                "tool_calls": [{"tool": "x", "accept_any_arguments": True}],
            },
        },
    )

    # No `has_tool_call_test` key in details, unlike a job created via the
    # real launch path — simulates a pre-existing job from before this field.
    job_id = create_agent_test_job(
        agent_id=agent["uuid"],
        job_type="llm-unit-test",
        details={"test_uuids": [tool_test_uuid]},
    )
    update_agent_test_job(job_id, status="done", results={"total_tests": 1, "passed": 1, "failed": 0})

    resp = client.get(f"/agent-tests/agent/{agent['uuid']}/runs", headers=h)
    assert resp.status_code == 200
    run = resp.json()["items"][0]
    assert run["evaluators"] == []


def test_agent_runs_list_evaluator_cache_batches_lookups(client, monkeypatch):
    """Resolving evaluator names for a list of runs costs one batched DB
    query total (`_prefetch_evaluator_cache` → `get_evaluators_by_uuids`),
    not one per distinct evaluator: three runs sharing one evaluator must
    not trigger three individual `get_evaluator` lookups."""
    import db
    from db import create_agent_test_job, update_agent_test_job

    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)

    live_name = f"Shared {uuid.uuid4().hex[:6]}"
    ev = _create_llm_evaluator(client, h, live_name)

    job_uuids = set()
    for i in range(3):
        job_id = create_agent_test_job(
            agent_id=agent["uuid"],
            job_type="llm-unit-test",
            details={
                "evaluators_by_test_id": {
                    f"tc{i}": [{"uuid": ev["uuid"], "name": f"{live_name}-stale"}]
                }
            },
        )
        job_uuids.add(job_id)
        update_agent_test_job(
            job_id, status="done", results={"total_tests": 1, "passed": 1, "failed": 0}
        )

    batch_calls = []
    real_batch = db.get_evaluators_by_uuids

    def _counting_batch(uuids):
        batch_calls.append(list(uuids))
        return real_batch(uuids)

    def _fail_individual(_uuid):
        raise AssertionError(
            "get_evaluator was called individually; the batch cache should "
            "already have every referenced uuid"
        )

    monkeypatch.setattr(db, "get_evaluators_by_uuids", _counting_batch)
    monkeypatch.setattr(db, "get_evaluator", _fail_individual)

    resp = client.get(f"/agent-tests/agent/{agent['uuid']}/runs", headers=h)
    assert resp.status_code == 200
    assert len(batch_calls) == 1
    seen = {r["uuid"]: r for r in resp.json()["items"] if r["uuid"] in job_uuids}
    assert len(seen) == 3
    for run in seen.values():
        assert run["evaluators"] == [live_name]


def test_global_runs_list_evaluators_column(client):
    """The workspace-wide GET /agent-tests/runs surfaces the same `evaluators`
    column, resolved and cached the same way as the per-agent endpoint."""
    from db import create_agent_test_job, update_agent_test_job

    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)

    live_name = f"Global {uuid.uuid4().hex[:6]}"
    ev = _create_llm_evaluator(client, h, live_name)

    job_id = create_agent_test_job(
        agent_id=agent["uuid"],
        job_type="llm-unit-test",
        details={
            "evaluators_by_test_id": {
                "tc1": [{"uuid": ev["uuid"], "name": f"{live_name}-stale"}]
            },
            "has_tool_call_test": True,
        },
    )
    update_agent_test_job(job_id, status="done", results={"total_tests": 1, "passed": 1, "failed": 0})

    resp = client.get("/agent-tests/runs", headers=h)
    assert resp.status_code == 200
    run = next(r for r in resp.json()["items"] if r["uuid"] == job_id)
    assert run["evaluators"] == [live_name, "Tool call"]


def test_agent_runs_list_filters_and_pagination(client):
    """The agent runs-list accepts optional `type`/`status`/`has_failures`
    filters + `limit`/`offset` paging, returning the `{items, total, ...}`
    envelope where `total` is the pre-slice count of the filtered set. Names
    ("Run N"/"Benchmark N") stay stable regardless of filters."""
    from db import create_agent_test_job, update_agent_test_job

    h = _signup(client)["headers"]
    agent = _create_agent(client, h)
    au = agent["uuid"]

    # Run 1: unit test, done, clean (no failures).
    clean = create_agent_test_job(agent_id=au, job_type="llm-unit-test")
    update_agent_test_job(
        clean, status="done", results={"total_tests": 2, "passed": 2, "failed": 0}
    )
    # Run 2: unit test, done, with failures.
    failing = create_agent_test_job(agent_id=au, job_type="llm-unit-test")
    update_agent_test_job(
        failing, status="done", results={"total_tests": 2, "passed": 1, "failed": 1}
    )
    # Benchmark 1: in_progress, one model failed.
    bench = create_agent_test_job(agent_id=au, job_type="llm-benchmark")
    update_agent_test_job(
        bench,
        status="in_progress",
        results={
            "model_results": [
                {"model": "openai/gpt-4.1", "success": True, "message": "ok", "passed": 2, "failed": 0},
                {"model": "openai/gpt-4o", "success": True, "message": "ok", "passed": 1, "failed": 1},
            ]
        },
    )

    def _get(**params):
        r = client.get(f"/agent-tests/agent/{au}/runs", params=params, headers=h)
        assert r.status_code == 200
        return r

    # No params → every run, `total` = all runs.
    body = _get().json()
    all_runs = body["items"]
    assert len(all_runs) == 3
    assert body["total"] == 3
    uuid_to_name = {x["uuid"]: x["name"] for x in all_runs}
    assert uuid_to_name[clean] == "Run 1"
    assert uuid_to_name[failing] == "Run 2"
    assert uuid_to_name[bench] == "Benchmark 1"

    # type filter.
    body = _get(type="llm-benchmark").json()
    assert {x["uuid"] for x in body["items"]} == {bench}
    assert body["total"] == 1

    # status filter.
    assert {x["uuid"] for x in _get(status="in_progress").json()["items"]} == {bench}
    assert {x["uuid"] for x in _get(status="done").json()["items"]} == {clean, failing}

    # has_failures — covers both unit (aggregate `failed`) and benchmark (a
    # model's `failed`) shapes.
    assert {x["uuid"] for x in _get(has_failures=True).json()["items"]} == {
        failing,
        bench,
    }
    assert {x["uuid"] for x in _get(has_failures=False).json()["items"]} == {clean}

    # Filters compose; names remain stable (still "Run 2", not renumbered).
    only = _get(type="llm-unit-test", has_failures=True).json()["items"]
    assert [x["uuid"] for x in only] == [failing]
    assert only[0]["name"] == "Run 2"

    # Pagination: total reflects the filtered set (pre-slice), page is sliced.
    b1 = _get(status="done", limit=1, offset=0).json()
    assert len(b1["items"]) == 1 and b1["total"] == 2
    b2 = _get(status="done", limit=1, offset=1).json()
    assert len(b2["items"]) == 1
    assert b1["items"][0]["uuid"] != b2["items"][0]["uuid"]

    # `around` returns the page containing that run instead of `offset`'s page.
    # Runs are newest-first: bench (0), failing (1), clean (2).
    around_clean = _get(limit=1, around=clean).json()
    assert around_clean["offset"] == 2
    assert [x["uuid"] for x in around_clean["items"]] == [clean]

    around_bench = _get(limit=2, around=failing).json()
    assert around_bench["offset"] == 0  # index 1 falls in the first page of 2
    assert failing in {x["uuid"] for x in around_bench["items"]}

    # `around` without `limit` stays unbounded, same as a plain unbounded list
    # (no `limit` means "no limit" everywhere else in this API too).
    around_unbounded = _get(around=clean).json()
    assert around_unbounded["limit"] is None
    assert len(around_unbounded["items"]) == 3
    assert clean in {x["uuid"] for x in around_unbounded["items"]}

    # A run excluded by a filter (or unknown) is not in the filtered results.
    r = client.get(
        f"/agent-tests/agent/{au}/runs",
        params={"status": "in_progress", "around": clean},
        headers=h,
    )
    assert r.status_code == 404
    r = client.get(
        f"/agent-tests/agent/{au}/runs",
        params={"around": "not-a-real-uuid"},
        headers=h,
    )
    assert r.status_code == 404


def test_global_runs_list_filters_and_pagination(client):
    """The workspace-wide GET /agent-tests/runs (JWT-only) uses the same
    `{items, total, ...}` envelope as the per-agent endpoint and accepts
    `type`/`status`/`has_failures` filters + paging across every agent."""
    from db import create_agent_test_job, update_agent_test_job

    h = _signup(client)["headers"]
    a1 = _create_agent(client, h)
    a2 = _create_agent(client, h)

    # a1: one clean unit run. a2: one failing unit run + one benchmark.
    clean = create_agent_test_job(agent_id=a1["uuid"], job_type="llm-unit-test")
    update_agent_test_job(
        clean, status="done", results={"total_tests": 1, "passed": 1, "failed": 0}
    )
    failing = create_agent_test_job(agent_id=a2["uuid"], job_type="llm-unit-test")
    update_agent_test_job(
        failing, status="done", results={"total_tests": 1, "passed": 0, "failed": 1}
    )
    bench = create_agent_test_job(agent_id=a2["uuid"], job_type="llm-benchmark")
    update_agent_test_job(
        bench, status="in_progress", results={"model_results": []}
    )
    mine = {clean, failing, bench}

    def _get(**params):
        r = client.get("/agent-tests/runs", params=params, headers=h)
        assert r.status_code == 200
        return r.json()

    # Envelope shape + spans both agents.
    body = _get()
    got = {x["uuid"] for x in body["items"]} & mine
    assert got == mine
    assert body["total"] >= 3

    # Filters narrow the set (scoped to this workspace's three runs).
    def _mine(**params):
        return {x["uuid"] for x in _get(**params)["items"]} & mine

    assert _mine(type="llm-benchmark") == {bench}
    assert _mine(status="done") == {clean, failing}

    # `around` (same contract as the per-agent endpoint): page containing the
    # run, 404 when it's filtered out.
    body = _get(limit=1, around=clean)
    assert clean in {x["uuid"] for x in body["items"]}
    assert client.get(
        "/agent-tests/runs", params={"type": "llm-benchmark", "around": clean}, headers=h
    ).status_code == 404
    assert _mine(has_failures=True) == {failing}
    assert _mine(has_failures=False, type="llm-unit-test") == {clean}

    # Pagination slices; total reflects the filtered set.
    page = _get(status="done", limit=1)
    assert len(page["items"]) == 1 and page["total"] == 2


def test_slim_run_list_helpers_guard_edge_cases():
    """The run-list slimming helpers tolerate empty/missing input and skip
    non-dict rows, and lift `test_case.name` up onto a case's flat `name`."""
    from routers.agent_tests import _slim_test_results, _slim_model_results

    # Falsy input → None
    assert _slim_test_results(None) is None
    assert _slim_test_results([]) is None
    assert _slim_model_results(None) is None

    # Non-dict rows are skipped; an all-junk list collapses to None
    assert _slim_test_results(["x", None]) is None
    assert _slim_model_results([42]) is None

    # `test_case.name` is lifted onto `name` when the row has no own name
    assert _slim_test_results([{"test_case": {"name": "tc"}, "passed": False}]) == [
        {"name": "tc", "passed": False}
    ]


def test_run_evaluator_names_guards_malformed_snapshot_entries():
    """`_run_evaluator_names` skips non-dict entries and entries with no
    resolvable name (no live match and no snapshot name) rather than
    crashing or emitting a blank/junk entry."""
    from routers.agent_tests import _run_evaluator_names

    job = {
        "evaluators_by_test_id": {
            "tc1": [
                "not-a-dict",
                {"uuid": "ev1"},  # no name anywhere → skipped
                {"uuid": "ev2", "name": "Correctness"},
            ]
        }
    }
    assert _run_evaluator_names(job) == ["Correctness"]


def _seed_run_job(client, h, agent):
    """Seed a unit-test run with three cases (one pass, one fail, one still
    pending/`passed=None`) and an evaluator carrying a rubric, for the
    run-detail compact/only_failed tests."""
    from db import create_agent_test_job, update_agent_test_job

    job_id = create_agent_test_job(
        agent_id=agent["uuid"],
        job_type="llm-unit-test",
        details={
            "test_uuids": ["tc_pass", "tc_fail", "tc_pending"],
            "evaluators_by_test_id": {
                "tc_pass": [
                    {
                        "uuid": "ev1",
                        "name": "correctness",
                        "output_type": "binary",
                        "output_config": {
                            "scale": [
                                {"value": False, "name": "Wrong"},
                                {"value": True, "name": "Right"},
                            ]
                        },
                    }
                ]
            }
        },
    )
    update_agent_test_job(
        job_id,
        status="done",
        results={
            "total_tests": 3,
            "passed": 1,
            "failed": 1,
            "test_results": [
                {
                    "name": "tc_pass",
                    "test_case_id": "tc_pass",
                    "passed": True,
                    "output": {"response": "hi", "tool_calls": None},
                    "test_case": {"name": "tc_pass", "history": []},
                    "reasoning": "looks good",
                    "judge_results": [
                        {"evaluator_uuid": NONEXISTENT_UUID, "match": True}
                    ],
                },
                {
                    "name": "tc_fail",
                    "test_case_id": "tc_fail",
                    "passed": False,
                    "output": {"response": "nope", "tool_calls": None},
                    "test_case": {"name": "tc_fail", "history": []},
                    "reasoning": "wrong answer",
                    "judge_results": [
                        {"evaluator_uuid": NONEXISTENT_UUID, "match": False}
                    ],
                },
                {
                    # Pending case — not finished yet (`passed is None`),
                    # matching the pending placeholder shape.
                    "name": "tc_pending",
                    "test_case_id": None,
                    "passed": None,
                    "output": None,
                    "test_case": None,
                    "reasoning": None,
                    "judge_results": None,
                },
            ],
        },
    )
    return job_id


def test_run_detail_default_is_full(client):
    """No query params → the run-detail response carries every heavy field."""
    h = _signup(client)["headers"]
    agent = _create_agent(client, h)
    job_id = _seed_run_job(client, h, agent)

    resp = client.get(f"/agent-tests/run/{job_id}", headers=h)
    assert resp.status_code == 200
    body = resp.json()
    assert body["test_uuids"] == ["tc_pass", "tc_fail", "tc_pending"]
    assert len(body["results"]) == 3
    # Results are positionally aligned with test_uuids by test_case_id (the
    # test's UUID, echoed back by calibrate) — NOT by the display `name`. The
    # pending row is unfinished, so it carries no test_case_id yet.
    assert [r["test_case_id"] for r in body["results"]] == [
        "tc_pass",
        "tc_fail",
        None,
    ]
    case = body["results"][0]
    assert case["output"] is not None
    assert case["test_case"] is not None
    assert case["judge_results"] is not None
    assert case["reasoning"] is not None
    assert body["evaluators"][0]["output_config"] is not None


def test_run_detail_compact_nulls_heavy_fields(client):
    """`?compact=true` nulls the heavy per-case + evaluator fields but keeps the
    slim identity fields (name/passed/status)."""
    h = _signup(client)["headers"]
    agent = _create_agent(client, h)
    job_id = _seed_run_job(client, h, agent)

    resp = client.get(
        f"/agent-tests/run/{job_id}", params={"compact": "true"}, headers=h
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert len(body["results"]) == 3
    for case in body["results"]:
        assert case["output"] is None
        assert case["test_case"] is None
        assert case["judge_results"] is None
        assert case["reasoning"] is None
        # Slim fields survive.
        assert case["name"] in {"tc_pass", "tc_fail", "tc_pending"}
        assert case["passed"] in {True, False, None}
    assert body["evaluators"][0]["output_config"] is None
    # Non-heavy evaluator fields survive.
    assert body["evaluators"][0]["name"] == "correctness"


def test_run_detail_only_failed_narrows_results(client):
    """`?only_failed=true` keeps only failing cases (`passed is False`). The
    pass is dropped, and the still-pending case (`passed is None`) is dropped
    too — a mid-run poll must not surface unfinished cases as failures."""
    h = _signup(client)["headers"]
    agent = _create_agent(client, h)
    job_id = _seed_run_job(client, h, agent)

    resp = client.get(
        f"/agent-tests/run/{job_id}", params={"only_failed": "true"}, headers=h
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [c["name"] for c in body["results"]] == ["tc_fail"]
    # Heavy fields still present (compact not requested).
    assert body["results"][0]["output"] is not None


def _seed_benchmark_job(client, h, agent):
    """Seed a completed benchmark run with one model holding a pass + fail case."""
    from db import create_agent_test_job, update_agent_test_job

    job_id = create_agent_test_job(
        agent_id=agent["uuid"],
        job_type="llm-benchmark",
        details={
            "test_uuids": ["tc_pass", "tc_fail", "tc_pending"],
            "evaluators_by_test_id": {
                "tc_pass": [
                    {
                        "uuid": "ev1",
                        "name": "correctness",
                        "output_type": "binary",
                        "output_config": {
                            "scale": [
                                {"value": False, "name": "Wrong"},
                                {"value": True, "name": "Right"},
                            ]
                        },
                    }
                ]
            }
        },
    )
    update_agent_test_job(
        job_id,
        status="done",
        results={
            "model_results": [
                {
                    "model": "openai/gpt-4.1",
                    "success": True,
                    "message": "ok",
                    "total_tests": 3,
                    "passed": 1,
                    "failed": 1,
                    "test_results": [
                        {
                            "name": "tc_pass",
                            "test_case_id": "tc_pass",
                            "passed": True,
                            "output": {"response": "hi"},
                        },
                        {
                            "name": "tc_fail",
                            "test_case_id": "tc_fail",
                            "passed": False,
                            "output": {"response": "no"},
                        },
                        {
                            # Pending case — not finished yet (`passed is None`);
                            # a real pending placeholder carries no test_case_id.
                            "name": "tc_pending",
                            "test_case_id": None,
                            "passed": None,
                            "output": None,
                        },
                    ],
                }
            ],
        },
    )
    return job_id


def test_benchmark_detail_default_is_full(client):
    """No query params → benchmark detail keeps each model's test_results and the
    evaluator rubric."""
    h = _signup(client)["headers"]
    agent = _create_agent(client, h)
    job_id = _seed_benchmark_job(client, h, agent)

    resp = client.get(f"/agent-tests/benchmark/{job_id}", headers=h)
    assert resp.status_code == 200
    body = resp.json()
    assert body["test_uuids"] == ["tc_pass", "tc_fail", "tc_pending"]
    model = body["model_results"][0]
    assert len(model["test_results"]) == 3
    # Each model's test_results are positionally aligned with test_uuids by
    # test_case_id (the test's UUID), not the display `name`.
    assert [r["test_case_id"] for r in model["test_results"]] == [
        "tc_pass",
        "tc_fail",
        None,
    ]
    assert body["evaluators"][0]["output_config"] is not None


def test_benchmark_detail_compact_nulls_heavy_fields(client):
    """`?compact=true` nulls each model's test_results + the evaluator rubric,
    keeping model-level scalar fields."""
    h = _signup(client)["headers"]
    agent = _create_agent(client, h)
    job_id = _seed_benchmark_job(client, h, agent)

    resp = client.get(
        f"/agent-tests/benchmark/{job_id}", params={"compact": "true"}, headers=h
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    model = body["model_results"][0]
    assert model["test_results"] is None
    # Model-level scalars survive.
    assert model["model"] == "openai/gpt-4.1"
    assert model["passed"] == 1
    assert model["failed"] == 1
    assert body["evaluators"][0]["output_config"] is None


def test_benchmark_detail_only_failed_narrows_each_model(client):
    """`?only_failed=true` narrows each model's test_results to failing cases
    (`passed is False`), dropping the pass and the still-pending case
    (`passed is None`), and leaving model-level fields intact."""
    h = _signup(client)["headers"]
    agent = _create_agent(client, h)
    job_id = _seed_benchmark_job(client, h, agent)

    resp = client.get(
        f"/agent-tests/benchmark/{job_id}", params={"only_failed": "true"}, headers=h
    )
    assert resp.status_code == 200
    body = resp.json()
    model = body["model_results"][0]
    assert [c["name"] for c in model["test_results"]] == ["tc_fail"]
    # Model-level fields untouched.
    assert model["passed"] == 1
    assert model["total_tests"] == 3


def test_agent_tests_link_with_missing(client):
    auth = _signup(client)
    h = auth["headers"]
    # Missing agent
    resp = client.post(
        "/agent-tests",
        json={"agent_uuid": NONEXISTENT_UUID, "test_uuids": []},
        headers=h,
    )
    assert resp.status_code == 404

    agent = _create_agent(client, h)
    # Missing test
    bad = client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [NONEXISTENT_UUID_2]},
        headers=h,
    )
    assert bad.status_code == 404


def test_agent_tests_delete_link_not_found(client):
    auth = _signup(client)
    h = auth["headers"]
    resp = client.request(
        "DELETE",
        "/agent-tests",
        json={"agent_uuid": NONEXISTENT_UUID, "test_uuid": NONEXISTENT_UUID_2},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Run + benchmark validations (queue path, no thread)
# ---------------------------------------------------------------------------


def test_run_agent_test_validation(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]

    # Unauthenticated → 403 (HTTPBearer rejects the missing header)
    assert client.post("/agent-tests/agent/missing/run", json={}).status_code == 403

    # Missing agent
    resp = client.post("/agent-tests/agent/missing/run", json={}, headers=h)
    assert resp.status_code == 404

    # Agent with no linked tests
    agent = _create_agent(client, h)
    no_tests = client.post(
        f"/agent-tests/agent/{agent['uuid']}/run", json={}, headers=h
    )
    assert no_tests.status_code == 400

    # Provide bogus test_uuids
    bad = client.post(
        f"/agent-tests/agent/{agent['uuid']}/run",
        json={"test_uuids": [NONEXISTENT_UUID]},
        headers=h,
    )
    assert bad.status_code == 404

    # Someone in another workspace cannot run tests on this agent → refused (403)
    other = _signup(client)
    cross = client.post(
        f"/agent-tests/agent/{agent['uuid']}/run",
        json={},
        headers=other["headers"],
    )
    assert cross.status_code == 403


def test_run_agent_test_rejects_cross_org_test_uuid(client):
    """A test_uuid from another org must not be runnable against my agent —
    404 (existence parity), matching how a missing test_uuid already 404s."""
    other = _signup(client)
    other_test = _create_test(client, other["headers"])

    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)

    r = client.post(
        f"/agent-tests/agent/{agent['uuid']}/run",
        json={"test_uuids": [other_test["uuid"]]},
        headers=h,
    )
    assert r.status_code == 404


def test_run_agent_test_queued_path(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    test = _create_test(client, h)
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.agent_tests.can_start_agent_test_job", return_value=False), patch(
        "threading.Thread"
    ):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run", json={}, headers=h
        )
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]
    assert resp.json()["status"] == "queued"

    # Status requires auth
    assert client.get(f"/agent-tests/run/{task_id}").status_code == 403
    got = client.get(f"/agent-tests/run/{task_id}", headers=h)
    assert got.status_code == 200
    # results_s3_prefix is an internal storage key — never exposed in the API
    # response (the frontend doesn't read it either).
    assert "results_s3_prefix" not in got.json()

    # Someone in another workspace cannot poll this run → refused (403)
    other_poll = _signup(client)
    assert (
        client.get(
            f"/agent-tests/run/{task_id}", headers=other_poll["headers"]
        ).status_code
        == 403
    )

    # 404 unknown run
    assert client.get("/agent-tests/run/missing", headers=h).status_code == 404

    # Visibility toggle
    on = client.patch(
        f"/agent-tests/run/{task_id}/visibility",
        json={"is_public": True},
        headers=h,
    )
    assert on.status_code == 200
    off = client.patch(
        f"/agent-tests/run/{task_id}/visibility",
        json={"is_public": False},
        headers=h,
    )
    assert off.status_code == 200
    # Someone in another workspace cannot change sharing on this run → refused (403)
    other = _signup(client)
    assert (
        client.patch(
            f"/agent-tests/run/{task_id}/visibility",
            json={"is_public": True},
            headers=other["headers"],
        ).status_code
        == 403
    )
    assert (
        client.patch(
            "/agent-tests/run/missing/visibility",
            json={"is_public": True},
            headers=h,
        ).status_code
        == 404
    )

    # Delete
    deleted = client.delete(f"/agent-tests/job/{task_id}", headers=h)
    assert deleted.status_code == 200
    # already gone
    assert (
        client.delete(f"/agent-tests/job/{task_id}", headers=h).status_code == 404
    )
    assert client.delete("/agent-tests/job/missing", headers=h).status_code == 404


def test_run_conversation_test_queued_path(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    conv = _create_conversation_test(client, h)
    assert conv.get("uuid"), conv
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [conv["uuid"]]},
        headers=h,
    )

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("threading.Thread"):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run", json={}, headers=h
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "queued"
    task_id = resp.json()["task_id"]

    # Conversation tests flow through the normal calibrate-llm config: the
    # frozen calibrate_config carries the conversation test case (with its
    # evaluation.type + criteria) and the top-level evaluators list.
    import db

    job = db.get_agent_test_job(task_id)
    details = job["details"]
    cfg = details["calibrate_config"]
    assert cfg.get("evaluators"), cfg
    case = next(c for c in cfg["test_cases"] if c["id"] == conv["uuid"])
    assert case["evaluation"]["type"] == "conversation"
    assert case["evaluation"]["criteria"]
    assert conv["uuid"] in details["evaluators_by_test_id"]

    # Status endpoint serializes fine for a conversation run.
    got = client.get(f"/agent-tests/run/{task_id}", headers=h)
    assert got.status_code == 200

    # Clean up the queued job so it doesn't pollute the shared session DB
    # (try_start_queued_agent_test_job picks the oldest queued job).
    assert client.delete(f"/agent-tests/job/{task_id}", headers=h).status_code == 200


def test_run_mixed_conversation_and_response_allowed(client, monkeypatch):
    """The calibrate CLI dispatches per test case on evaluation.type, so a run
    may mix conversation with response tests in a single job."""
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    response_test = _create_test(client, h)
    conv = _create_conversation_test(client, h)

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("threading.Thread"):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run",
            json={"test_uuids": [response_test["uuid"], conv["uuid"]]},
            headers=h,
        )
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]

    import db

    cfg = db.get_agent_test_job(task_id)["details"]["calibrate_config"]
    types = {c["evaluation"]["type"] for c in cfg["test_cases"]}
    assert types == {"response", "conversation"}

    assert client.delete(f"/agent-tests/job/{task_id}", headers=h).status_code == 200


def test_benchmark_allows_conversation_tests(client, monkeypatch):
    """Conversation rows ignore the benchmarked model, but a benchmark that
    includes them is still accepted (handled by the same calibrate-llm path)."""
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    conv = _create_conversation_test(client, h)
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [conv["uuid"]]},
        headers=h,
    )

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("threading.Thread"):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/benchmark",
            json={"models": ["openai/gpt-4"]},
            headers=h,
        )
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]
    assert client.delete(f"/agent-tests/job/{task_id}", headers=h).status_code == 200


def test_benchmark_response_test_judge_results_completes(client, monkeypatch):
    """Regression: a benchmark that includes a graded (``response``) test must
    complete and expose ``judge_results`` as a LIST.

    calibrate emits ``judge_results`` as a dict keyed by evaluator name. The
    benchmark runner used to wrap each model's parsed rows in a Pydantic
    ``ModelResult`` at WRITE time, whose ``TestCaseResult.judge_results`` is
    typed ``List[JudgeResult]`` — so the raw dict raised a ``ValidationError``
    and the job crashed to ``failed``. The runner now stores raw dicts (like
    ``run_llm_test_task``) and the read endpoint converts dict→list. Only
    ``response``/``conversation`` tests trip it — ``tool_call`` has
    ``judge_results=None`` — which is why it slipped past earlier tests that
    only forced the failure path or stubbed the worker thread.
    """
    import json
    from pathlib import Path

    import db
    from routers.agent_tests import run_benchmark_task

    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    test_name = f"t-{uuid.uuid4().hex[:6]}"
    test = _create_test(client, h, name=test_name)  # response-type, seeded llm evaluator
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )

    # Name calibrate keys judge_results by = the linked evaluator's name.
    evaluators = client.get("/evaluators", headers=h).json()["items"]
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")
    ev_name = llm_ev["name"]

    agent_row = db.get_agent(agent["uuid"])
    test_row = db.get_test(test["uuid"])
    job_uuid = db.create_agent_test_job(
        agent_id=agent["uuid"], job_type="llm-benchmark", status="in_progress"
    )

    class _P:
        def __init__(self):
            self.returncode = 0
            self.pid = 4242
            self._poll = [None, 0]

        def poll(self):
            return self._poll.pop(0) if self._poll else 0

        def wait(self, *a, **k):
            return 0

    def fake_popen(*args, **kwargs):
        model_dir = Path(kwargs["cwd"]) / "output" / "gpt-4.1"
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "results.json", "w") as f:
            json.dump(
                [
                    {
                        "output": {"response": "Yes.", "tool_calls": []},
                        "metrics": {
                            "passed": True,
                            "reasoning": "ok",
                            # dict keyed by evaluator name — the shape that used
                            # to crash the benchmark write path.
                            "judge_results": {
                                ev_name: {"reasoning": "good", "match": True}
                            },
                        },
                        "test_case": {"id": test["uuid"], "name": test_name},
                        "test_case_id": test["uuid"],
                    }
                ],
                f,
            )
        with open(model_dir / "metrics.json", "w") as f:
            json.dump({"total": 1, "passed": 1, "criteria": {}}, f)
        return _P()

    with patch(
        "routers.agent_tests.subprocess.Popen", side_effect=fake_popen
    ), patch(
        "routers.agent_tests.get_s3_client", return_value=MagicMock()
    ), patch("routers.agent_tests.upload_directory_tree_to_s3"), patch(
        "routers.agent_tests.upload_file_to_s3"
    ), patch(
        "routers.agent_tests.try_start_queued_agent_test_job"
    ), patch(
        "routers.agent_tests.time.sleep"
    ):
        run_benchmark_task(job_uuid, agent_row, [test_row], ["gpt-4.1"], "bucket")

    # Write path no longer crashes: job reaches ``done``.
    job = db.get_agent_test_job(job_uuid)
    assert job["status"] == "done", job.get("results")

    # Read path returns 200 and reshapes judge_results into a list.
    resp = client.get(f"/agent-tests/benchmark/{job_uuid}", headers=h)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "done"
    judge_results = data["model_results"][0]["test_results"][0]["judge_results"]
    assert isinstance(judge_results, list), judge_results
    assert judge_results[0]["match"] is True


def test_unverified_connection_blocks_all_test_types(client, monkeypatch):
    """Every test type runs the agent (conversation tests are live too), so an
    unverified agent-connection agent blocks response AND conversation runs."""
    import db

    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    # Agent-connection agent that hasn't been verified.
    db.update_agent(
        agent["uuid"],
        config={"agent_url": "http://agent.local/run", "connection_verified": False},
    )

    response_test = _create_test(client, h)
    conv = _create_conversation_test(client, h)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")

    for test_uuid in (response_test["uuid"], conv["uuid"]):
        blocked = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run",
            json={"test_uuids": [test_uuid]},
            headers=h,
        )
        assert blocked.status_code == 400, blocked.text
        assert "not verified" in blocked.json()["detail"].lower()


def test_drifted_config_eval_type_follows_immutable_row_type(client, monkeypatch):
    """The immutable row `type` is authoritative: a conversation-typed test whose
    stored config.evaluation.type has drifted to "response" is normalized back to
    conversation at CLI handoff, so calibrate dispatches it as a conversation."""
    import db

    auth = _signup(client)
    h = auth["headers"]
    # Plain (calibrate-agent-mode) agent — no agent_url, so the connection guard
    # doesn't apply and we can inspect the built config.
    agent = _create_agent(client, h)
    sim_ev = _create_simulation_evaluator(client, h)["uuid"]
    # Schema lets config be arbitrary while row `type` is immutable, so this
    # divergent state is reachable via the API.
    drifted = client.post(
        "/tests",
        json={
            "name": f"mm-{uuid.uuid4().hex[:6]}",
            "type": "conversation",
            "config": {"history": [], "evaluation": {"type": "response"}},
            "evaluators": [{"evaluator_uuid": sim_ev}],
        },
        headers=h,
    ).json()

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("threading.Thread"):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run",
            json={"test_uuids": [drifted["uuid"]]},
            headers=h,
        )
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]
    # The built calibrate config normalizes evaluation.type to the row type.
    cfg = db.get_agent_test_job(task_id)["details"]["calibrate_config"]
    case = next(c for c in cfg["test_cases"] if c["id"] == drifted["uuid"])
    assert case["evaluation"]["type"] == "conversation"
    assert client.delete(f"/agent-tests/job/{task_id}", headers=h).status_code == 200


def test_run_agent_test_missing_s3_config_500(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    conv = _create_conversation_test(client, h)

    with patch(
        "routers.agent_tests.get_s3_output_config",
        side_effect=ValueError("no bucket configured"),
    ):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run",
            json={"test_uuids": [conv["uuid"]]},
            headers=h,
        )
    assert resp.status_code == 500


def test_run_agent_benchmark_validation(client):
    auth = _signup(client)
    h = auth["headers"]
    # Unauthenticated → 403 (HTTPBearer rejects the missing header)
    assert (
        client.post(
            "/agent-tests/agent/missing/benchmark", json={"models": ["x"]}
        ).status_code
        == 403
    )

    # Missing agent
    resp = client.post(
        "/agent-tests/agent/missing/benchmark", json={"models": ["x"]}, headers=h
    )
    assert resp.status_code == 404

    # No models
    agent = _create_agent(client, h)
    bad = client.post(
        f"/agent-tests/agent/{agent['uuid']}/benchmark",
        json={"models": []},
        headers=h,
    )
    assert bad.status_code == 400

    # No tests linked
    no_tests = client.post(
        f"/agent-tests/agent/{agent['uuid']}/benchmark",
        json={"models": ["openai/gpt-4"]},
        headers=h,
    )
    assert no_tests.status_code == 400

    # test_uuids referencing a test not linked to the agent → 404
    test = _create_test(client, h)
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )
    unlinked = _create_test(client, h)  # exists but not linked to this agent
    bad_subset = client.post(
        f"/agent-tests/agent/{agent['uuid']}/benchmark",
        json={"models": ["openai/gpt-4"], "test_uuids": [unlinked["uuid"]]},
        headers=h,
    )
    assert bad_subset.status_code == 404
    assert unlinked["uuid"] in bad_subset.json()["detail"]


def test_run_agent_benchmark_subset_scoping(client, monkeypatch):
    """A benchmark with `test_uuids` runs only the requested linked subset."""
    from db import get_agent_test_job

    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    t1 = _create_test(client, h)
    t2 = _create_test(client, h)
    for t in (t1, t2):
        client.post(
            "/agent-tests",
            json={"agent_uuid": agent["uuid"], "test_uuids": [t["uuid"]]},
            headers=h,
        )

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("threading.Thread"):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/benchmark",
            json={"models": ["openai/gpt-4"], "test_uuids": [t2["uuid"]]},
            headers=h,
        )
    assert resp.status_code == 200
    job = get_agent_test_job(resp.json()["task_id"])
    assert job["details"]["test_uuids"] == [t2["uuid"]]


def test_run_agent_benchmark_queued_path(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    test = _create_test(client, h)
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.agent_tests.can_start_agent_test_job", return_value=False), patch(
        "threading.Thread"
    ):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/benchmark",
            json={"models": ["openai/gpt-4"]},
            headers=h,
        )
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    # Status requires auth
    assert client.get(f"/agent-tests/benchmark/{task_id}").status_code == 403
    got = client.get(f"/agent-tests/benchmark/{task_id}", headers=h)
    assert got.status_code == 200
    # `evaluators[]` block is now exposed on the benchmark status response
    # the same way it is on the unit-test status — confirm the field is
    # at least present (may be empty for a queued/never-run job).
    assert "evaluators" in got.json()

    # Someone in another workspace cannot poll this benchmark → refused (403)
    other_poll = _signup(client)
    assert (
        client.get(
            f"/agent-tests/benchmark/{task_id}", headers=other_poll["headers"]
        ).status_code
        == 403
    )
    assert (
        client.get("/agent-tests/benchmark/missing", headers=h).status_code == 404
    )

    # Visibility toggle
    on = client.patch(
        f"/agent-tests/benchmark/{task_id}/visibility",
        json={"is_public": True},
        headers=h,
    )
    assert on.status_code == 200
    off = client.patch(
        f"/agent-tests/benchmark/{task_id}/visibility",
        json={"is_public": False},
        headers=h,
    )
    assert off.status_code == 200
    assert (
        client.patch(
            "/agent-tests/benchmark/missing/visibility",
            json={"is_public": True},
            headers=h,
        ).status_code
        == 404
    )


def test_agent_test_inflight(client, monkeypatch):
    """Cover the can_start=True branch where the thread is spawned."""
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    test = _create_test(client, h)
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.agent_tests.can_start_agent_test_job", return_value=True), patch(
        "routers.agent_tests.threading.Thread"
    ) as thread_mock:
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run", json={}, headers=h
        )
        assert resp.status_code == 200
        thread_mock.return_value.start.assert_called_once()


# ---------------------------------------------------------------------------
# Batch run endpoint: POST /agent-tests/run (optional agent_names payload)
# ---------------------------------------------------------------------------


def test_run_tests_batch_by_names(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    n1, n2, n3 = (f"agent-{uuid.uuid4().hex[:6]}" for _ in range(3))
    a1 = _create_agent(client, h, name=n1)
    a2 = _create_agent(client, h, name=n2)
    a3_no_tests = _create_agent(client, h, name=n3)
    t1 = _create_test(client, h)
    t2 = _create_test(client, h)
    client.post(
        "/agent-tests",
        json={"agent_uuid": a1["uuid"], "test_uuids": [t1["uuid"]]},
        headers=h,
    )
    client.post(
        "/agent-tests",
        json={"agent_uuid": a2["uuid"], "test_uuids": [t2["uuid"]]},
        headers=h,
    )

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")

    # Auth required.
    assert (
        client.post(
            "/agent-tests/run",
            json={"agent_names": [n1]},
        ).status_code
        == 403
    )

    # An unknown name fails validation up front → 404, NO tasks created.
    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("routers.agent_tests._launch_agent_test_run") as launch_mock:
        bad = client.post(
            "/agent-tests/run",
            json={"agent_names": [n1, "does-not-exist"]},
            headers=h,
        )
    assert bad.status_code == 404
    assert "does-not-exist" in bad.json()["detail"]["not_found"]
    launch_mock.assert_not_called()  # nothing launched when validation fails

    # Valid batch: a1 + a2 launch; a3 (no linked tests) is skipped.
    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("threading.Thread"):
        resp = client.post(
            "/agent-tests/run",
            json={"agent_names": [n1, n2, n3]},
            headers=h,
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    runs_by_name = {r["agent_name"]: r for r in data["runs"]}
    assert set(runs_by_name) == {n1, n2}
    for run in runs_by_name.values():
        assert run["agent_uuid"]
        assert run["task_id"]
        assert run["status"] == "queued"
    assert runs_by_name[n1]["agent_uuid"] == a1["uuid"]
    skipped = {s["agent_name"]: s["reason"] for s in data["skipped"]}
    assert skipped == {n3: "no_linked_tests"}

    # Clean up queued jobs so they don't pollute the shared session queue.
    for run in data["runs"]:
        client.delete(f"/agent-tests/job/{run['task_id']}", headers=h)


def test_run_tests_batch_skips_unverified(client, monkeypatch):
    import db

    auth = _signup(client)
    h = auth["headers"]
    name = f"agent-{uuid.uuid4().hex[:6]}"
    agent = _create_agent(client, h, name=name)
    test = _create_test(client, h)
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )
    # Make it a connection-type agent that hasn't been verified.
    db.update_agent(
        agent["uuid"],
        config={"agent_url": "http://agent.local/run", "connection_verified": False},
    )

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("threading.Thread"):
        resp = client.post(
            "/agent-tests/run",
            json={"agent_names": [name]},
            headers=h,
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["runs"] == []
    assert data["skipped"] == [
        {
            "agent_name": name,
            "agent_uuid": agent["uuid"],
            "reason": "connection_not_verified",
        }
    ]


def test_run_tests_batch_all_agents(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    n1 = f"agent-{uuid.uuid4().hex[:6]}"
    n2 = f"agent-{uuid.uuid4().hex[:6]}"
    a1 = _create_agent(client, h, name=n1)
    _create_agent(client, h, name=n2)  # no linked tests
    t1 = _create_test(client, h)
    client.post(
        "/agent-tests",
        json={"agent_uuid": a1["uuid"], "test_uuids": [t1["uuid"]]},
        headers=h,
    )

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")

    # Auth required.
    assert client.post("/agent-tests/run").status_code == 403

    # No body, empty body, and explicit empty list all mean "run all agents".
    created_task_ids: list[str] = []
    for body in (None, {}, {"agent_names": []}):
        with patch(
            "routers.agent_tests.can_start_agent_test_job", return_value=False
        ), patch("threading.Thread"):
            resp = client.post("/agent-tests/run", json=body, headers=h)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        run_names = {r["agent_name"] for r in data["runs"]}
        assert n1 in run_names
        assert any(
            s["agent_name"] == n2 and s["reason"] == "no_linked_tests"
            for s in data["skipped"]
        )
        created_task_ids.extend(r["task_id"] for r in data["runs"])

    # Clean up queued jobs so they don't pollute the shared session queue.
    for task_id in created_task_ids:
        client.delete(f"/agent-tests/job/{task_id}", headers=h)

    # Org scoping: a fresh org sees none of the above agents.
    other = _signup(client)
    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("threading.Thread"):
        other_resp = client.post("/agent-tests/run", headers=other["headers"])
    assert other_resp.status_code == 200
    other_data = other_resp.json()
    assert other_data["runs"] == []
    assert other_data["skipped"] == []


# ---------------------------------------------------------------------------
# Run-list endpoints read slim DB summaries — heavy detail must never leak
# ---------------------------------------------------------------------------


_HEAVY_OUTPUT_MARK = "HEAVY-OUTPUT-SENTINEL-XZ1"
_HEAVY_JUDGE_MARK = "HEAVY-JUDGE-SENTINEL-XZ2"
_HEAVY_REASONING_MARK = "HEAVY-REASONING-SENTINEL-XZ3"
_HEAVY_DETAILS_MARK = "HEAVY-DETAILS-SENTINEL-XZ4"
_HEAVY_TESTCASE_MARK = "HEAVY-TESTCASE-SENTINEL-XZ5"
_HEAVY_LEADERBOARD_MARK = "HEAVY-LEADERBOARD-SENTINEL-XZ6"

_ALL_HEAVY_MARKS = (
    _HEAVY_OUTPUT_MARK,
    _HEAVY_JUDGE_MARK,
    _HEAVY_REASONING_MARK,
    _HEAVY_DETAILS_MARK,
    _HEAVY_TESTCASE_MARK,
    _HEAVY_LEADERBOARD_MARK,
)


def _seed_heavy_unit_run(agent_uuid):
    """Seed a done unit-test run whose stored `results` + `details` are stuffed
    with heavy per-case sub-trees and unique sentinel strings, so a leak of any
    heavy field is detectable by scanning the serialized response."""
    from db import create_agent_test_job, update_agent_test_job

    job_id = create_agent_test_job(
        agent_id=agent_uuid,
        job_type="llm-unit-test",
        details={"calibrate_config": {"note": _HEAVY_DETAILS_MARK}},
    )
    update_agent_test_job(
        job_id,
        status="done",
        results={
            "total_tests": 2,
            "passed": 1,
            "failed": 1,
            "latency_ms": {"p50": 12.0, "p95": 12.0, "p99": 12.0, "count": 2},
            "test_results": [
                {
                    "name": "case_named",
                    "passed": True,
                    "output": {"response": _HEAVY_OUTPUT_MARK, "tool_calls": None},
                    "test_case": {"name": "case_named", "history": [_HEAVY_TESTCASE_MARK]},
                    "reasoning": _HEAVY_REASONING_MARK,
                    "judge_results": [
                        {"evaluator_uuid": NONEXISTENT_UUID, "reasoning": _HEAVY_JUDGE_MARK}
                    ],
                },
                {
                    # No top-level name → the flat `name` falls back to test_case.name.
                    "passed": False,
                    "output": {"response": _HEAVY_OUTPUT_MARK, "tool_calls": None},
                    "test_case": {"name": "case_from_test_case", "history": [_HEAVY_TESTCASE_MARK]},
                    "reasoning": _HEAVY_REASONING_MARK,
                    "judge_results": [
                        {"evaluator_uuid": NONEXISTENT_UUID, "reasoning": _HEAVY_JUDGE_MARK}
                    ],
                },
            ],
        },
    )
    return job_id


def _seed_heavy_benchmark_run(agent_uuid):
    """Seed a done benchmark run with heavy nested per-model `test_results` +
    leaderboard + a heavy `details` blob."""
    from db import create_agent_test_job, update_agent_test_job

    job_id = create_agent_test_job(
        agent_id=agent_uuid,
        job_type="llm-benchmark",
        details={"calibrate_config": {"note": _HEAVY_DETAILS_MARK}},
    )
    update_agent_test_job(
        job_id,
        status="done",
        results={
            "leaderboard_summary": [{"model": "openai/gpt-4.1", "note": _HEAVY_LEADERBOARD_MARK}],
            "model_results": [
                {
                    "model": "openai/gpt-4.1",
                    "success": True,
                    "message": "ok",
                    "total_tests": 2,
                    "passed": 1,
                    "failed": 1,
                    "test_results": [
                        {
                            "name": "case_named",
                            "passed": True,
                            "output": {"response": _HEAVY_OUTPUT_MARK},
                            "test_case": {"name": "case_named", "history": [_HEAVY_TESTCASE_MARK]},
                            "reasoning": _HEAVY_REASONING_MARK,
                            "judge_results": [{"reasoning": _HEAVY_JUDGE_MARK}],
                        }
                    ],
                    "evaluator_summary": [{"name": "correctness", "note": _HEAVY_LEADERBOARD_MARK}],
                }
            ],
        },
    )
    return job_id


def _assert_no_heavy_leak(run_item):
    """No heavy sentinel appears anywhere in the serialized run-list item, and no
    per-case/per-model object carries a heavy key."""
    import json

    blob = json.dumps(run_item)
    for mark in _ALL_HEAVY_MARKS:
        assert mark not in blob, f"heavy sentinel {mark} leaked into run-list item"
    assert "details" not in run_item
    for case in run_item.get("results") or []:
        assert set(case) <= {"name", "passed"}
        for heavy in ("output", "judge_results", "reasoning", "test_case"):
            assert heavy not in case
    for model in run_item.get("model_results") or []:
        assert "test_results" not in model
        assert "evaluator_summary" not in model
    assert "leaderboard_summary" not in run_item


def test_agent_runs_list_hides_heavy_detail_both_endpoints(client):
    """Both run-list endpoints read the slim DB summary: the per-agent and global
    lists surface aggregates + slim `{name, passed}` rows and slim model scalars,
    but never the heavy per-case/per-model sub-trees or the `details` blob. The
    `test_case.name` fallback surfaces a name for a case with no top-level name."""
    h = _signup(client)["headers"]
    agent_name = f"a-heavy-{uuid.uuid4().hex[:6]}"
    agent = _create_agent(client, h, name=agent_name)

    unit_id = _seed_heavy_unit_run(agent["uuid"])
    bench_id = _seed_heavy_benchmark_run(agent["uuid"])

    # Per-agent endpoint.
    resp = client.get(f"/agent-tests/agent/{agent['uuid']}/runs", headers=h)
    assert resp.status_code == 200
    items = {r["uuid"]: r for r in resp.json()["items"]}
    assert set(items) == {unit_id, bench_id}

    unit = items[unit_id]
    assert unit["type"] == "llm-unit-test"
    assert unit["status"] == "done"
    assert unit["name"] == "Run 1"
    assert (unit["total_tests"], unit["passed"], unit["failed"]) == (2, 1, 1)
    assert unit["latency_ms"] == {"p50": 12.0, "p95": 12.0, "p99": 12.0, "count": 2}
    # Slim rows: name + passed only; the second case's name comes from test_case.name.
    assert unit["results"] == [
        {"name": "case_named", "passed": True},
        {"name": "case_from_test_case", "passed": False},
    ]
    _assert_no_heavy_leak(unit)

    bench = items[bench_id]
    assert bench["type"] == "llm-benchmark"
    assert bench["name"] == "Benchmark 1"
    assert bench["model_results"] == [
        {
            "model": "openai/gpt-4.1",
            "success": True,
            "message": "ok",
            "total_tests": 2,
            "passed": 1,
            "failed": 1,
        }
    ]
    _assert_no_heavy_leak(bench)

    # Global endpoint returns the same slim shape.
    gresp = client.get("/agent-tests/runs", headers=h)
    assert gresp.status_code == 200
    gitems = {r["uuid"]: r for r in gresp.json()["items"]}
    assert {unit_id, bench_id} <= set(gitems)
    for jid in (unit_id, bench_id):
        item = gitems[jid]
        # Global items also carry agent_id/agent_name and stay heavy-free.
        assert item["agent_id"] == agent["uuid"]
        assert item["agent_name"] == agent_name
        _assert_no_heavy_leak(item)
    assert gitems[unit_id]["results"] == [
        {"name": "case_named", "passed": True},
        {"name": "case_from_test_case", "passed": False},
    ]


def test_agent_runs_list_filters_and_pagination_with_heavy_jobs(client):
    """Filters (`type`/`status`/`has_failures`) and paging (`limit`/`offset`) work
    over heavy-seeded jobs on both endpoints, still returning the slim envelope."""
    from db import create_agent_test_job, update_agent_test_job

    h = _signup(client)["headers"]
    agent = _create_agent(client, h)
    au = agent["uuid"]

    unit_pass = create_agent_test_job(agent_id=au, job_type="llm-unit-test")
    update_agent_test_job(
        unit_pass, status="done", results={"total_tests": 1, "passed": 1, "failed": 0}
    )
    unit_id = _seed_heavy_unit_run(au)  # done, has failures
    bench_prog = create_agent_test_job(agent_id=au, job_type="llm-benchmark")
    update_agent_test_job(bench_prog, status="in_progress", results={"model_results": []})

    def _agent(**params):
        r = client.get(f"/agent-tests/agent/{au}/runs", params=params, headers=h)
        assert r.status_code == 200
        return r.json()

    # type filter.
    assert {x["uuid"] for x in _agent(type="llm-benchmark")["items"]} == {bench_prog}
    # status filter.
    assert {x["uuid"] for x in _agent(status="in_progress")["items"]} == {bench_prog}
    assert {x["uuid"] for x in _agent(status="done")["items"]} == {unit_pass, unit_id}
    # has_failures — only the heavy unit run has a failing case; the clean unit
    # run and the empty in-progress benchmark are both failure-free.
    assert {x["uuid"] for x in _agent(has_failures=True)["items"]} == {unit_id}
    assert {x["uuid"] for x in _agent(has_failures=False)["items"]} == {
        unit_pass,
        bench_prog,
    }
    # Pagination: total is the pre-slice filtered count; page is sliced.
    p1 = _agent(status="done", limit=1, offset=0)
    p2 = _agent(status="done", limit=1, offset=1)
    assert len(p1["items"]) == 1 and p1["total"] == 2
    assert len(p2["items"]) == 1
    assert p1["items"][0]["uuid"] != p2["items"][0]["uuid"]

    # Global endpoint: same filters, scoped to this workspace's three jobs.
    mine = {unit_pass, unit_id, bench_prog}

    def _global(**params):
        r = client.get("/agent-tests/runs", params=params, headers=h)
        assert r.status_code == 200
        return {x["uuid"] for x in r.json()["items"]} & mine

    assert _global(type="llm-benchmark") == {bench_prog}
    assert _global(status="done") == {unit_pass, unit_id}
    assert _global(has_failures=True) == {unit_id}


# ---------------------------------------------------------------------------
# Org scoping for the link + read endpoints (security fix)
# ---------------------------------------------------------------------------


def test_agent_tests_link_and_reads_are_org_scoped(client):
    """POST /agent-tests and the two agent-scoped read endpoints
    (GET /agent-tests/agent/{uuid}/{tests,runs}) require auth and only see the
    caller's own org. Reading another workspace's agent is refused (403); the
    link call sends the agent id in the body instead of the address, so it stays
    404. An unauthenticated link 403s."""
    a = _signup(client)
    b = _signup(client)
    a_agent = _create_agent(client, a["headers"])
    a_test = _create_test(client, a["headers"])

    # User A links their own test — succeeds.
    linked = client.post(
        "/agent-tests",
        json={"agent_uuid": a_agent["uuid"], "test_uuids": [a_test["uuid"]]},
        headers=a["headers"],
    )
    assert linked.status_code == 200

    # User B cannot link against A's agent. The agent id travels in the body, so
    # this stays a plain 404 with no workspace hint.
    foreign_link = client.post(
        "/agent-tests",
        json={"agent_uuid": a_agent["uuid"], "test_uuids": [a_test["uuid"]]},
        headers=b["headers"],
    )
    assert foreign_link.status_code == 404

    # User B cannot read A's agent tests or runs → refused (403).
    assert (
        client.get(
            f"/agent-tests/agent/{a_agent['uuid']}/tests", headers=b["headers"]
        ).status_code
        == 403
    )
    assert (
        client.get(
            f"/agent-tests/agent/{a_agent['uuid']}/runs", headers=b["headers"]
        ).status_code
        == 403
    )

    # Unauthenticated link → 403 (HTTPBearer rejects the missing header).
    assert (
        client.post(
            "/agent-tests",
            json={"agent_uuid": a_agent["uuid"], "test_uuids": [a_test["uuid"]]},
        ).status_code
        == 403
    )


# ---------------------------------------------------------------------------
# Linking validates test type against agent interaction_type
# ---------------------------------------------------------------------------


def test_link_general_test_to_general_agent_succeeds(client):
    auth = _signup(client)
    h = auth["headers"]
    agent = client.post(
        "/agents",
        json={"name": f"a-{uuid.uuid4().hex[:6]}", "type": "agent", "interaction_type": "general"},
        headers=h,
    ).json()
    gen_test = _create_general_test(client, h)

    link = client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [gen_test["uuid"]]},
        headers=h,
    )
    assert link.status_code == 200
    assert link.json()["ids"]

    linked_tests = client.get(f"/agent-tests/agent/{agent['uuid']}/tests", headers=h).json()
    assert gen_test["uuid"] in {t["uuid"] for t in linked_tests["items"]}


def test_link_general_test_to_conversation_agent_fails(client):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)  # default interaction_type="conversation"
    gen_test = _create_general_test(client, h)

    link = client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [gen_test["uuid"]]},
        headers=h,
    )
    assert link.status_code == 400
    assert gen_test["uuid"] in link.json()["detail"]

    linked_tests = client.get(f"/agent-tests/agent/{agent['uuid']}/tests", headers=h).json()
    assert gen_test["uuid"] not in {t["uuid"] for t in linked_tests["items"]}


def test_link_conversation_test_to_general_agent_fails(client):
    auth = _signup(client)
    h = auth["headers"]
    agent = client.post(
        "/agents",
        json={"name": f"a-{uuid.uuid4().hex[:6]}", "type": "agent", "interaction_type": "general"},
        headers=h,
    ).json()

    response_test = _create_test(client, h)
    conv_test = _create_conversation_test(client, h)

    for t in (response_test, conv_test):
        link = client.post(
            "/agent-tests",
            json={"agent_uuid": agent["uuid"], "test_uuids": [t["uuid"]]},
            headers=h,
        )
        assert link.status_code == 400
        assert t["uuid"] in link.json()["detail"]

    linked_tests = client.get(f"/agent-tests/agent/{agent['uuid']}/tests", headers=h).json()
    linked_uuids = {t["uuid"] for t in linked_tests["items"]}
    assert response_test["uuid"] not in linked_uuids
    assert conv_test["uuid"] not in linked_uuids


def test_link_batch_with_one_mismatch_rejects_whole_request(client):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)  # default interaction_type="conversation"

    valid_test = _create_test(client, h)
    mismatched_test = _create_general_test(client, h)

    link = client.post(
        "/agent-tests",
        json={
            "agent_uuid": agent["uuid"],
            "test_uuids": [valid_test["uuid"], mismatched_test["uuid"]],
        },
        headers=h,
    )
    assert link.status_code == 400
    assert mismatched_test["uuid"] in link.json()["detail"]

    linked_tests = client.get(f"/agent-tests/agent/{agent['uuid']}/tests", headers=h).json()
    linked_uuids = {t["uuid"] for t in linked_tests["items"]}
    assert valid_test["uuid"] not in linked_uuids
    assert mismatched_test["uuid"] not in linked_uuids


def test_link_response_test_to_conversation_agent_succeeds(client):
    """Regression check: the default agent/test combo still links fine."""
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)  # default interaction_type="conversation"
    response_test = _create_test(client, h)

    link = client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [response_test["uuid"]]},
        headers=h,
    )
    assert link.status_code == 200
    assert link.json()["ids"]

    linked_tests = client.get(f"/agent-tests/agent/{agent['uuid']}/tests", headers=h).json()
    assert response_test["uuid"] in {t["uuid"] for t in linked_tests["items"]}


# ---------------------------------------------------------------------------
# tool_call tests span both interaction types, decided by their config shape
# ---------------------------------------------------------------------------


def _create_tool_call_test(client, h, *, input_text=None, history=None, name=None):
    config = {"evaluation": {"type": "tool_call", "tool_calls": [{"tool": "search", "arguments": {"q": "x"}}]}}
    if input_text is not None:
        config["input"] = input_text
    else:
        config["history"] = history or [{"role": "user", "content": "find x"}]
    return client.post(
        "/tests",
        json={
            "name": name or f"tc-{uuid.uuid4().hex[:6]}",
            "type": "tool_call",
            "config": config,
        },
        headers=h,
    ).json()


def _create_general_agent(client, h):
    return client.post(
        "/agents",
        json={
            "name": f"a-{uuid.uuid4().hex[:6]}",
            "type": "agent",
            "interaction_type": "general",
        },
        headers=h,
    ).json()


def test_link_input_tool_call_test_to_general_agent_succeeds(client):
    h = _signup(client)["headers"]
    agent = _create_general_agent(client, h)
    test = _create_tool_call_test(client, h, input_text="find the weather in Delhi")

    link = client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )
    assert link.status_code == 200, link.json()
    assert link.json()["ids"]


def test_link_history_tool_call_test_to_general_agent_fails(client):
    h = _signup(client)["headers"]
    agent = _create_general_agent(client, h)
    test = _create_tool_call_test(client, h)

    link = client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )
    assert link.status_code == 400
    assert test["uuid"] in link.json()["detail"]


def test_link_input_tool_call_test_to_conversation_agent_fails(client):
    h = _signup(client)["headers"]
    agent = _create_agent(client, h)  # default interaction_type="conversation"
    test = _create_tool_call_test(client, h, input_text="find the weather in Delhi")

    link = client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )
    assert link.status_code == 400


def test_link_history_tool_call_test_to_conversation_agent_succeeds(client):
    h = _signup(client)["headers"]
    agent = _create_agent(client, h)
    test = _create_tool_call_test(client, h)

    link = client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )
    assert link.status_code == 200, link.json()


def test_build_calibrate_config_widens_tool_call_input_to_one_user_turn(client):
    """An `input`-shaped tool_call test reaches calibrate as a one-message history."""
    from routers.agent_tests import _build_calibrate_config

    h = _signup(client)["headers"]
    agent = _create_general_agent(client, h)
    test = _create_tool_call_test(client, h, input_text="find the weather in Delhi")
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )

    from db import get_agent as _get_agent, get_test as _get_test

    config, _ = _build_calibrate_config(
        _get_agent(agent["uuid"]), [_get_test(test["uuid"])], "bucket"
    )
    case = config["test_cases"][0]
    assert case["history"] == [
        {"role": "user", "content": "find the weather in Delhi"}
    ]
    # The plain input survives on the test case so the run result can show it
    # as an Input/Output pair instead of a chat.
    assert case["input"] == "find the weather in Delhi"
    assert case["evaluation"]["type"] == "tool_call"
    assert case["evaluation"]["tool_calls"] == [
        {"tool": "search", "arguments": {"q": "x"}}
    ]


def test_tool_call_test_with_both_input_and_history_reads_as_conversational(client):
    """The link gate and the calibrate handoff must agree on which one is sent."""
    from routers.agent_tests import _build_calibrate_config
    from routers.tests import required_agent_interaction_type
    from db import get_agent as _get_agent, get_test as _get_test

    h = _signup(client)["headers"]
    history = [{"role": "user", "content": "what is the weather"}]
    test = _create_tool_call_test(client, h, history=history)
    # A config can carry both only when hand-written: POST /tests stores it verbatim.
    client.put(
        f"/tests/{test['uuid']}",
        json={
            "config": {
                "input": "find weather",
                "history": history,
                "evaluation": {
                    "type": "tool_call",
                    "tool_calls": [{"tool": "search", "arguments": {"q": "x"}}],
                },
            }
        },
        headers=h,
    )
    stored = _get_test(test["uuid"])
    assert required_agent_interaction_type("tool_call", stored["config"]) == "conversation"

    general_agent = _create_general_agent(client, h)
    link = client.post(
        "/agent-tests",
        json={"agent_uuid": general_agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )
    assert link.status_code == 400

    conv_agent = _create_agent(client, h)
    config, _ = _build_calibrate_config(_get_agent(conv_agent["uuid"]), [stored], "b")
    assert config["test_cases"][0]["history"] == history


def test_build_calibrate_config_tolerates_expected_tool_call_without_arguments(client):
    from routers.agent_tests import _build_calibrate_config
    from db import get_agent as _get_agent, get_test as _get_test

    h = _signup(client)["headers"]
    agent = _create_agent(client, h)
    test = client.post(
        "/tests",
        json={
            "name": f"tc-{uuid.uuid4().hex[:6]}",
            "type": "tool_call",
            "config": {
                "history": [{"role": "user", "content": "find x"}],
                "evaluation": {"type": "tool_call", "tool_calls": [{"tool": "search"}]},
            },
        },
        headers=h,
    ).json()

    config, _ = _build_calibrate_config(
        _get_agent(agent["uuid"]), [_get_test(test["uuid"])], "bucket"
    )
    assert config["test_cases"][0]["evaluation"]["tool_calls"] == [
        {"tool": "search", "arguments": None}
    ]


@pytest.mark.parametrize(
    "interaction_type,expected", [(None, "conversation"), ("general", "general")]
)
def test_build_calibrate_config_sends_the_agents_type(
    client, interaction_type, expected
):
    """calibrate reads `agent_type` to decide whether to POST `messages` or `input`."""
    import db
    from routers.agent_tests import _build_calibrate_config

    h = _signup(client)["headers"]
    agent = (
        _create_general_agent(client, h)
        if interaction_type
        else _create_agent(client, h)
    )
    db.update_agent(
        agent["uuid"],
        config={"agent_url": "http://agent.local/run", "connection_verified": True},
    )
    test = (
        _create_tool_call_test(client, h, input_text="find the weather in Delhi")
        if interaction_type
        else _create_test(client, h)
    )

    config, _ = _build_calibrate_config(
        db.get_agent(agent["uuid"]), [db.get_test(test["uuid"])], "bucket"
    )
    assert config["agent_type"] == expected


def test_build_calibrate_config_rejects_an_unknown_agent_type(client):
    """calibrate exits the whole run on an unknown `agent_type`, so catch it here."""
    import db
    from routers.agent_tests import _build_calibrate_config

    h = _signup(client)["headers"]
    agent = _create_agent(client, h)
    db.update_agent(
        agent["uuid"],
        config={"agent_url": "http://agent.local/run", "connection_verified": True},
    )
    test = _create_test(client, h)

    stored = db.get_agent(agent["uuid"])
    stored["interaction_type"] = "chat"

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _build_calibrate_config(stored, [db.get_test(test["uuid"])], "bucket")
    assert exc.value.status_code == 400
    assert "conversation, general" in exc.value.detail


def test_run_by_test_uuids_rejects_a_test_the_agent_cannot_be_sent(client):
    """Linking checks this; running by ID never did. calibrate raises on the
    first test whose shape the agent cannot take and gathers without
    return_exceptions, so one of these would take the whole run down."""
    import db

    h = _signup(client)["headers"]
    agent = _create_general_agent(client, h)
    db.update_agent(
        agent["uuid"],
        config={"agent_url": "http://agent.local/run", "connection_verified": True},
    )
    conversational = _create_test(client, h)
    fits = _create_tool_call_test(client, h, input_text="find the weather in Delhi")

    blocked = client.post(
        f"/agent-tests/agent/{agent['uuid']}/run",
        json={"test_uuids": [fits["uuid"], conversational["uuid"]]},
        headers=h,
    )
    assert blocked.status_code == 400, blocked.text
    detail = blocked.json()["detail"]
    assert detail["test_uuids"] == [conversational["uuid"]]
    assert "general" in detail["error"]

    # No job was created, so the run never started.
    runs = client.get(f"/agent-tests/agent/{agent['uuid']}/runs", headers=h)
    assert runs.json()["total"] == 0
