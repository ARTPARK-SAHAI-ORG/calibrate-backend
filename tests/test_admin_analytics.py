"""GET /admin/access and GET /admin/analytics against a fresh database."""

import json
import uuid

import pytest
from fastapi.testclient import TestClient

import db
from auth_utils import create_access_token

# Periods used below: this one is 2026-10-10..12, the previous one 2026-10-07..09.
Q = {"start": "2026-10-10", "end": "2026-10-12"}

ADMIN, ALICE, BOB, STUB = (str(uuid.uuid4()) for _ in range(4))
A, B, C, STUB_ORG, GONE = (str(uuid.uuid4()) for _ in range(5))


def _ins(conn, table, **cols):
    conn.execute(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        list(cols.values()),
    )


def _seed(conn):
    for uid, first, email, pw, at in [
        (ADMIN, "Ad", "admin@example.com", "x", "2026-10-10 10:00:00"),
        (ALICE, "Alice", "alice@example.com", "x", "2026-10-11T09:00:00"),
        (BOB, "Bob", "bob@example.com", "x", "2026-10-08 10:00:00"),
        (STUB, "", "stub@example.com", None, "2026-10-11 10:00:00"),
    ]:
        _ins(conn, "users", uuid=uid, first_name=first, last_name="", email=email,
             password_hash=pw, created_at=at)
    for org, name, by, personal, deleted in [
        (A, "Acme", ALICE, 0, None),
        (B, "Alice personal", ALICE, 1, None),
        (C, "Bob personal", BOB, 1, None),
        (STUB_ORG, "Stub personal", STUB, 1, None),
        (GONE, "Gone", BOB, 0, "2026-10-09 00:00:00"),
    ]:
        _ins(conn, "organizations", uuid=org, name=name, created_by_user_id=by,
             is_personal=personal, deleted_at=deleted)
    for org, user, role in [(A, ALICE, "owner"), (A, BOB, "admin"), (A, STUB, "admin"),
                            (B, ALICE, "owner"), (C, BOB, "owner")]:
        _ins(conn, "organization_members", org_uuid=org, user_id=user, role=role)

    agents = {org: str(uuid.uuid4()) for org in (A, B, GONE, STUB_ORG)}
    for org, ag in agents.items():
        _ins(conn, "agents", uuid=ag, name="ag", org_uuid=org)
    sim = str(uuid.uuid4())
    _ins(conn, "simulations", uuid=sim, name="sim", org_uuid=A)

    err = lambda e: json.dumps({"error": e})
    for org, typ, status, cost, results, at in [
        (A, "llm-unit-test", "completed", 0.5, None, "2026-10-10 12:00:00"),
        (A, "llm-unit-test", "failed", None, err("boom"), "2026-10-11T08:00:00.123456+00:00"),
        (B, "llm-benchmark", "completed", 1.25, None, "2026-10-12 23:00:00"),
        (A, "llm-benchmark", "in_progress", None, None, "2026-10-08 10:00:00"),
        (GONE, "llm-unit-test", "failed", 9.0, err("gone"), "2026-10-10 12:00:00"),
        (STUB_ORG, "llm-unit-test", "queued", None, None, "2026-10-10 12:00:00"),
    ]:
        _ins(conn, "agent_test_jobs", uuid=str(uuid.uuid4()), agent_id=agents[org], type=typ,
             status=status, cost=cost, results=results, created_at=at)
    FAILED_TEST = conn.execute(
        "SELECT uuid FROM agent_test_jobs WHERE status = 'failed' AND results LIKE '%boom%'"
    ).fetchone()[0]

    for status, results, at in [
        ("failed", err(True), "2026-10-11 10:00:00"),
        ("queued", None, "2026-10-12 09:30:00"),
    ]:
        _ins(conn, "simulation_jobs", uuid=str(uuid.uuid4()), simulation_id=sim, type="simulation",
             status=status, results=results, created_at=at)

    for org, typ, status, results, at, deleted in [
        (B, "stt-eval", "completed", None, "2026-10-10 11:00:00", None),
        (B, "tts-eval", "failed", err("tts down"), "2026-10-12 03:00:00", None),
        (A, "annotation-eval", "completed", None, "2026-10-09 11:00:00", None),
        (A, "stt-eval", "in_progress", None, "2026-10-10 11:00:00", "2026-10-10 12:00:00"),
        (B, "tts-eval", "queued", None, "2026-10-07 11:00:00", None),
    ]:
        _ins(conn, "jobs", uuid=str(uuid.uuid4()), type=typ, status=status, results=results,
             org_uuid=org, created_at=at, deleted_at=deleted)

    traces = []
    for org, at, deleted in [
        (A, "2026-10-10 10:00:00", None),
        (A, "2026-10-11 10:00:00", None),
        (A, "2026-10-11 10:00:00", "2026-10-11 11:00:00"),
        (B, "2026-10-09 10:00:00", None),
    ]:
        traces.append(str(uuid.uuid4()))
        _ins(conn, "traces", uuid=traces[-1], org_uuid=org, agent_id=agents[org], input="[]",
             output="{}", created_at=at, deleted_at=deleted)
    for org, status, error, at, done in [
        (A, "completed", None, "2026-10-10 10:00:00", "2026-10-11 01:00:00"),
        (A, "failed", "judge unreachable", "2026-10-12 05:00:00", None),
        (A, "skipped", "over_limit", "2026-10-12 06:00:00", None),
        (B, "skipped", "trace_deleted", "2026-10-12 06:00:00", None),
    ]:
        _ins(conn, "trace_eval_runs", uuid=str(uuid.uuid4()), trace_uuid=traces[0], org_uuid=org,
             agent_id=agents[org], status=status, error=error, available_at=at, created_at=at,
             updated_at=at, completed_at=done)
    # Scored, but the trace has since been deleted: not counted.
    _ins(conn, "trace_eval_runs", uuid=str(uuid.uuid4()), trace_uuid=traces[2], org_uuid=A,
         agent_id=agents[A], status="completed", available_at="2026-10-11 10:00:00",
         created_at="2026-10-11 10:00:00", updated_at="2026-10-11 10:00:00",
         completed_at="2026-10-11 11:00:00")
    FAILED_SCORING = conn.execute(
        "SELECT uuid FROM trace_eval_runs WHERE status = 'failed'"
    ).fetchone()[0]

    task = str(uuid.uuid4())
    _ins(conn, "annotation_tasks", uuid=task, user_id=ALICE, name="t", org_uuid=A)
    done_job, open_job = str(uuid.uuid4()), str(uuid.uuid4())
    _ins(conn, "annotation_jobs", uuid=done_job, task_id=task, annotator_id="x",
         public_token=str(uuid.uuid4()), status="completed", completed_at="2026-10-11 12:00:00")
    _ins(conn, "annotation_jobs", uuid=open_job, task_id=task, annotator_id="y",
         public_token=str(uuid.uuid4()), status="pending")
    for job, item, ev, at, deleted in [
        (done_job, "i1", "e1", "2026-10-11 12:00:00", None),
        (done_job, "i1", "e2", "2026-10-11 12:00:00", None),
        (done_job, "i2", "e1", "2026-10-11 12:00:00", None),
        (done_job, "i4", "e2", "2026-10-11 12:00:00", "2026-10-11 13:00:00"),
        (open_job, "i3", "e1", "2026-10-08 12:00:00", None),
    ]:
        _ins(conn, "annotations", uuid=str(uuid.uuid4()), job_id=job, item_id=item,
             evaluator_id=ev, created_at=at, deleted_at=deleted)

    # A deleted task: its finished job and its labels are not counted.
    gone_task, gone_job = str(uuid.uuid4()), str(uuid.uuid4())
    _ins(conn, "annotation_tasks", uuid=gone_task, user_id=ALICE, name="gone", org_uuid=A,
         deleted_at="2026-10-11 13:00:00")
    _ins(conn, "annotation_jobs", uuid=gone_job, task_id=gone_task, annotator_id="z",
         public_token=str(uuid.uuid4()), status="completed", completed_at="2026-10-11 12:00:00")
    _ins(conn, "annotations", uuid=str(uuid.uuid4()), job_id=gone_job, item_id="i9",
         evaluator_id="e1", created_at="2026-10-11 12:00:00")

    for org, name, kind, req, cap, at in [
        (A, "max_rows_per_eval", "test_run", 600, 500, "2026-10-10 14:00:00"),
        (B, "max_traces", "trace", 10, 5, "2026-10-06 14:00:00"),
        (GONE, "max_rows_per_eval", "stt", 600, 500, "2026-10-10 14:00:00"),
    ]:
        _ins(conn, "limit_hits", uuid=str(uuid.uuid4()), org_uuid=org, limit_name=name, kind=kind,
             requested=req, cap=cap, created_at=at)
    return FAILED_TEST, FAILED_SCORING


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(db, "DB_PATH", tmp_path_factory.mktemp("admin") / "pense.db")
        db.init_db()
        with db.get_db_connection() as conn:
            ids = _seed(conn)
            conn.commit()
        import main

        yield TestClient(main.app), ids


def _h(user, email):
    return {"Authorization": f"Bearer {create_access_token(user, email)}"}


ADMIN_H = lambda: _h(ADMIN, "admin@example.com")
ALICE_H = lambda: _h(ALICE, "alice@example.com")


def _get(client, **params):
    r = client.get("/admin/analytics", params=params, headers=ADMIN_H())
    assert r.status_code == 200, r.text
    return r.json()


def test_access_says_who_is_superadmin(env):
    client, _ = env
    assert client.get("/admin/access", headers=ADMIN_H()).json() == {"is_superadmin": True}
    assert client.get("/admin/access", headers=ALICE_H()).json() == {"is_superadmin": False}


def test_analytics_is_superadmin_only(env):
    client, _ = env
    assert client.get("/admin/analytics", headers=ALICE_H()).status_code == 403


def test_end_before_start_is_422(env):
    client, _ = env
    r = client.get("/admin/analytics", params={"start": "2026-10-12", "end": "2026-10-10"},
                   headers=ADMIN_H())
    assert r.status_code == 422


ZERO_DAY = dict(new_users=0, active_workspaces=0, test_runs=0, model_comparisons=0,
                simulations=0, stt=0, tts=0, labelling_evaluator_runs=0, failed_evaluations=0,
                traces_received=0, traces_scored=0, labels_collected=0, labelling_jobs_finished=0)
ZERO_RANGE = dict(evaluations=0, failed_evaluations=0, test_runs=0, model_comparisons=0,
                  simulations=0, stt=0, tts=0, labelling_evaluator_runs=0, cost=None,
                  traces_received=0, traces_scored=0, labelling_jobs_finished=0,
                  labels_collected=0, items_labelled=0, limit_hits=0)


def test_numbers_for_a_period(env):
    client, (failed_test, failed_scoring) = env
    body = _get(client, **Q)

    assert body["totals"] == {"users": 3, "workspaces": 3, "evaluations_running": 3}
    assert body["activity"] == {
        "new_users": 2, "active_workspaces": 2, "evaluations": 7, "failed_evaluations": 3,
        "test_runs": 2, "model_comparisons": 1, "simulations": 2, "stt": 1, "tts": 1,
        "labelling_evaluator_runs": 0, "cost": 1.75, "traces_received": 2, "traces_scored": 1,
        "labelling_jobs_finished": 1, "labels_collected": 3, "items_labelled": 2, "limit_hits": 2,
    }
    assert body["previous"] == {
        "new_users": 1, "active_workspaces": 2, "evaluations": 3, "failed_evaluations": 0,
        "test_runs": 0, "model_comparisons": 1, "simulations": 0, "stt": 0, "tts": 1,
        "labelling_evaluator_runs": 1, "cost": None, "traces_received": 1, "traces_scored": 0,
        "labelling_jobs_finished": 0, "labels_collected": 1, "items_labelled": 1, "limit_hits": 0,
    }
    assert body["per_day"] == [
        {**ZERO_DAY, "date": "2026-10-10", "new_users": 1, "active_workspaces": 2,
         "test_runs": 1, "stt": 1, "traces_received": 1},
        {**ZERO_DAY, "date": "2026-10-11", "new_users": 1, "active_workspaces": 1,
         "test_runs": 1, "simulations": 1, "failed_evaluations": 2, "traces_received": 1,
         "traces_scored": 1, "labels_collected": 3, "labelling_jobs_finished": 1},
        {**ZERO_DAY, "date": "2026-10-12", "active_workspaces": 2, "model_comparisons": 1,
         "simulations": 1, "tts": 1, "failed_evaluations": 1},
    ]
    assert body["workspaces"] == [
        {**ZERO_RANGE, "org_uuid": B, "name": "Alice personal", "is_personal": True, "members": 1,
         "last_active_at": "2026-10-12T23:00:00Z", "evaluations": 3, "failed_evaluations": 1,
         "model_comparisons": 1, "stt": 1, "tts": 1, "cost": 1.25},
        {**ZERO_RANGE, "org_uuid": A, "name": "Acme", "is_personal": False, "members": 2,
         "last_active_at": "2026-10-12T09:30:00Z", "evaluations": 4, "failed_evaluations": 2,
         "test_runs": 2, "simulations": 2, "cost": 0.5, "traces_received": 2,
         "traces_scored": 1, "labelling_jobs_finished": 1, "labels_collected": 3,
         "items_labelled": 2, "limit_hits": 2},
        {**ZERO_RANGE, "org_uuid": C, "name": "Bob personal", "is_personal": True, "members": 1,
         "last_active_at": None},
    ]
    assert body["failures_total"] == 4
    assert [(f["kind"], f["error"], f["org_name"]) for f in body["failures"]] == [
        ("trace_scoring", "judge unreachable", "Acme"),
        ("tts", "tts down", "Alice personal"),
        ("simulation", None, "Acme"),
        ("test_run", "boom", "Acme"),
    ]
    assert body["failures"][0]["id"] == failed_scoring
    assert body["failures"][3] == {"id": failed_test, "created_at": "2026-10-11T08:00:00Z",
                                   "org_uuid": A, "org_name": "Acme", "kind": "test_run",
                                   "error": "boom"}
    assert body["limit_hits_total"] == 2
    assert body["limit_hits"] == [
        {"created_at": "2026-10-12T06:00:00Z", "org_uuid": A, "org_name": "Acme",
         "limit_name": "max_scored_traces", "kind": "trace_scoring", "requested": None, "cap": None},
        {"created_at": "2026-10-10T14:00:00Z", "org_uuid": A, "org_name": "Acme",
         "limit_name": "max_rows_per_eval", "kind": "test_run", "requested": 600, "cap": 500},
    ]


def test_time_offset_moves_a_run_to_the_next_day(env):
    client, _ = env
    # The comparison at 23:00 UTC on the 12th is 04:30 on the 13th in India.
    day13 = {"start": "2026-10-13", "end": "2026-10-13"}
    assert _get(client, **day13)["activity"]["model_comparisons"] == 0
    assert _get(client, **day13, tz_offset_minutes=330)["activity"]["model_comparisons"] == 1
    assert _get(client, **Q, tz_offset_minutes=330)["activity"]["model_comparisons"] == 0


def test_all_time_starts_at_the_first_active_day(env):
    client, _ = env
    body = _get(client, end="2026-10-12")
    assert body["previous"] is None
    assert [d["date"] for d in body["per_day"]] == [
        "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09",
        "2026-10-10", "2026-10-11", "2026-10-12",
    ]
    assert body["activity"]["limit_hits"] == 3
    assert body["activity"]["evaluations"] == 10
    assert body["activity"]["new_users"] == 3
    assert body["limit_hits_total"] == 3
