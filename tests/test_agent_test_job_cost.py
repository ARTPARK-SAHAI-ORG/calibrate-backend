"""agent_test_jobs.cost: written with every results write, backfilled once."""

from __future__ import annotations

import json
import uuid

import db

RUN = {"total": 2, "passed": 1, "cost": 0.25}
COMPARISON = {
    "model_results": [
        {"model": "a", "cost": 0.5},
        {"model": "b", "cost": None},
        {"model": "c", "cost": 1.25},
    ]
}


def _cost(job_uuid):
    with db.get_db_connection() as conn:
        return conn.execute(
            "SELECT cost FROM agent_test_jobs WHERE uuid = ?", (job_uuid,)
        ).fetchone()[0]


def test_cost_from_results():
    assert db.agent_test_job_cost(RUN) == 0.25
    assert db.agent_test_job_cost(COMPARISON) == 1.75
    assert db.agent_test_job_cost({"model_results": [{"cost": None}]}) is None
    assert db.agent_test_job_cost({"model_results": []}) is None
    assert db.agent_test_job_cost({"error": "x"}) is None
    assert db.agent_test_job_cost(None) is None


def test_finished_test_run_and_comparison_store_their_cost():
    agent = str(uuid.uuid4())
    run = db.create_agent_test_job(agent_id=agent, job_type="llm-unit-test")
    assert _cost(run) is None
    db.update_agent_test_job(run, status="done", results=RUN)
    assert _cost(run) == 0.25

    comparison = db.create_agent_test_job(agent_id=agent, job_type="llm-benchmark")
    db.update_agent_test_job(comparison, status="done", results=COMPARISON)
    assert _cost(comparison) == 1.75

    # An imported comparison is created finished, with its results.
    imported = db.create_agent_test_job(
        agent_id=agent, job_type="llm-benchmark", status="done", results=COMPARISON
    )
    assert _cost(imported) == 1.75


def test_backfill_fills_old_rows_once():
    agent = str(uuid.uuid4())
    job = db.create_agent_test_job(agent_id=agent, job_type="llm-benchmark")
    with db.get_db_connection() as conn:
        # A row written before the column existed: results but no cost.
        conn.execute(
            "UPDATE agent_test_jobs SET results = ?, cost = NULL WHERE uuid = ?",
            (json.dumps(COMPARISON), job),
        )
        conn.execute(
            "DELETE FROM _schema_migrations WHERE name = ?",
            (db.AGENT_TEST_JOB_COST_BACKFILL_MIGRATION,),
        )
        conn.commit()

    db.init_db()
    assert _cost(job) == 1.75

    with db.get_db_connection() as conn:
        conn.execute("UPDATE agent_test_jobs SET cost = NULL WHERE uuid = ?", (job,))
        conn.commit()
    db.init_db()
    assert _cost(job) is None
