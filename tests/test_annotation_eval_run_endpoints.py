"""Tests for annotation-task evaluator-run endpoints — the start-job +
list + get + delete + visibility flow. Real start is mocked so no calibrate
subprocess spawns."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


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
            "first_name": "E",
            "last_name": "R",
            "email": f"er-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
        "user_uuid": body["user"]["uuid"],
    }


def _llm_ev(client, h):
    evs = client.get("/evaluators", headers=h).json()["items"]
    return next(e for e in evs if e.get("evaluator_type") == "llm")


def test_evaluator_run_lifecycle(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_ev(client, h)
    task_uuid = client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    ).json()["uuid"]
    # Add LLM-compatible items
    client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {
                    "payload": {
                        "name": "i1",
                        "chat_history": [{"role": "user", "content": "hi"}],
                        "agent_response": "hi back",
                    }
                },
                {
                    "payload": {
                        "name": "i2",
                        "chat_history": [{"role": "user", "content": "hi"}],
                        "agent_response": "yes",
                    }
                },
            ]
        },
        headers=h,
    )

    # Start a run (force queue path so no thread spawns)
    with patch(
        "routers.annotation_tasks.can_start_job", return_value=False
    ):
        resp = client.post(
            f"/annotation-tasks/{task_uuid}/evaluator-runs",
            json={
                "evaluators": [{"evaluator_id": llm_ev["uuid"]}],
                "select_all": True,
            },
            headers=h,
        )
    assert resp.status_code == 200
    job_uuid = resp.json()["job_uuid"]
    assert resp.json()["status"] == "queued"

    # Inflight path — start_annotation_eval_job mocked
    with patch(
        "routers.annotation_tasks.can_start_job", return_value=True
    ), patch(
        "routers.annotation_tasks.start_annotation_eval_job"
    ) as start:
        resp2 = client.post(
            f"/annotation-tasks/{task_uuid}/evaluator-runs",
            json={
                "evaluators": [{"evaluator_id": llm_ev["uuid"]}],
                "select_all": True,
            },
            headers=h,
        )
        start.assert_called_once()
    assert resp2.status_code == 200

    # List — slim shape: {evaluators[], runs[{uuid, status, item_count,
    # updated_at, evaluators}]}. Per-row fluff (details, results, user_id,
    # org_uuid, created_at, completed_at, error, share_token, ...) is
    # stripped; evaluator identity/version lives on the top-level block.
    listing = client.get(
        f"/annotation-tasks/{task_uuid}/evaluator-runs", headers=h
    )
    assert listing.status_code == 200
    body = listing.json()
    assert isinstance(body, dict)
    assert "evaluators" in body and "runs" in body
    assert len(body["runs"]) >= 2
    row = body["runs"][0]
    assert set(row.keys()) == {
        "uuid",
        "status",
        "item_count",
        "updated_at",
        "evaluators",
    }
    # Per-row evaluators are FK references only.
    if row["evaluators"]:
        assert set(row["evaluators"][0].keys()) == {
            "evaluator_id",
            "evaluator_version_id",
        }

    # GET job
    got = client.get(
        f"/annotation-tasks/{task_uuid}/evaluator-runs/{job_uuid}", headers=h
    )
    assert got.status_code == 200

    # Visibility — can't share an in-progress / queued run
    bad = client.patch(
        f"/annotation-tasks/{task_uuid}/evaluator-runs/{job_uuid}/visibility",
        json={"is_public": True},
        headers=h,
    )
    assert bad.status_code == 400

    # Off is allowed
    off = client.patch(
        f"/annotation-tasks/{task_uuid}/evaluator-runs/{job_uuid}/visibility",
        json={"is_public": False},
        headers=h,
    )
    assert off.status_code == 200

    # Delete (queued job is deletable)
    deleted = client.delete(
        f"/annotation-tasks/{task_uuid}/evaluator-runs/{job_uuid}", headers=h
    )
    assert deleted.status_code == 200


def test_evaluator_run_bad_evaluator_resolution(client):
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_ev(client, h)
    task_uuid = client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    ).json()["uuid"]
    client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {
                    "payload": {
                        "name": "i1",
                        "chat_history": [],
                        "agent_response": "x",
                    }
                }
            ]
        },
        headers=h,
    )

    # Try to use an evaluator not linked to the task (default-stt-transcription is not linked)
    evaluators = client.get("/evaluators", headers=h).json()["items"]
    other_ev = next(e for e in evaluators if e.get("evaluator_type") == "stt")
    resp = client.post(
        f"/annotation-tasks/{task_uuid}/evaluator-runs",
        json={
            "evaluators": [{"evaluator_id": other_ev["uuid"]}],
            "select_all": True,
        },
        headers=h,
    )
    assert resp.status_code == 400


def test_evaluator_run_detail_shape(client):
    """GET /annotation-tasks/{uuid}/evaluator-runs/{job_uuid} exposes the
    rubric via top-level `evaluators[].output_config.scale` (mirrors the
    labelling-job viewer's shape), strips the slim snapshot from
    `details.evaluators`, and removes the per-run `evaluator` /
    `evaluator_version` blobs — those are keyed back via
    `(evaluator_id, evaluator_version_id)` on each run row."""
    import db as db_mod
    from annotation_eval_runner import ANNOTATION_EVAL_JOB_TYPE

    auth = _signup(client)
    h = auth["headers"]
    user_uuid = auth["user_uuid"]
    org_uuid = db_mod.get_personal_org_for_user(user_uuid)["uuid"]
    llm_ev = _llm_ev(client, h)
    live_version_id = llm_ev["live_version_id"]

    task_uuid = client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    ).json()["uuid"]
    item_ids = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {"payload": {"name": "i1", "chat_history": [], "agent_response": "x"}}
            ]
        },
        headers=h,
    ).json()["item_ids"]

    # Seed a done job + one evaluator-run directly so we exercise the read
    # path without spawning calibrate. The snapshot in `details.evaluators`
    # is the slim shape the runner writes — we assert it's promoted to the
    # enriched top-level `evaluators[]` and dropped from `details`.
    job_uuid = db_mod.create_job(
        job_type=ANNOTATION_EVAL_JOB_TYPE,
        org_uuid=org_uuid,
        user_id=user_uuid,
        status="done",
        details={
            "task_id": task_uuid,
            "evaluators": [
                {
                    "evaluator_id": llm_ev["uuid"],
                    "evaluator_version_id": live_version_id,
                    "name": llm_ev["name"],
                }
            ],
            "item_count": 1,
            "item_ids": item_ids,
        },
    )
    db_mod.create_evaluator_runs(
        [
            {
                "job_id": job_uuid,
                "item_id": item_ids[0],
                "evaluator_id": llm_ev["uuid"],
                "evaluator_version_id": live_version_id,
                "status": "completed",
                "value": {"value": True, "reasoning": "ok"},
            }
        ]
    )

    got = client.get(
        f"/annotation-tasks/{task_uuid}/evaluator-runs/{job_uuid}", headers=h
    )
    assert got.status_code == 200
    body = got.json()

    # Top-level evaluators[] is the new contract — full rubric exposed once.
    assert "evaluators" in body
    assert len(body["evaluators"]) == 1
    ev_entry = body["evaluators"][0]
    assert ev_entry["uuid"] == llm_ev["uuid"]
    assert ev_entry["evaluator_version_id"] == live_version_id
    assert ev_entry["output_type"] in ("binary", "rating")
    assert isinstance(ev_entry.get("output_config"), dict)
    assert isinstance(ev_entry["output_config"].get("scale"), list)
    assert ev_entry["output_config"]["scale"]  # non-empty

    # Slim snapshot is no longer surfaced inside `details` — promoted to the
    # top-level block above.
    assert "evaluators" not in (body.get("details") or {})

    # Runs keep the FK columns but not the duplicated metadata blobs.
    assert body["runs"]
    run = body["runs"][0]
    assert run["evaluator_id"] == llm_ev["uuid"]
    assert run["evaluator_version_id"] == live_version_id
    assert "evaluator" not in run
    assert "evaluator_version" not in run


def test_build_evaluators_block_keeps_stub_for_deleted_evaluator():
    """When an evaluator is soft-deleted after the job ran, the block
    still emits a stub entry so rows[] references stay resolvable on
    the FE — fields the soft-delete strips (description, output_type,
    rubric) come back as null + a `deleted: true` flag."""
    from routers.annotation_tasks import _build_evaluators_block_for_eval_job
    from unittest.mock import patch

    job_details = {
        "evaluators": [
            {
                "evaluator_id": "ev-gone",
                "evaluator_version_id": None,
                "name": "Snapshot Name",
            }
        ]
    }
    raw_runs = [
        {"evaluator_id": "ev-gone", "evaluator_version_id": None}
    ]
    with patch("routers.annotation_tasks.get_evaluator", return_value=None):
        block = _build_evaluators_block_for_eval_job(job_details, raw_runs)
    assert len(block) == 1
    assert block[0]["uuid"] == "ev-gone"
    assert block[0]["name"] == "Snapshot Name"
    assert block[0]["deleted"] is True
    assert block[0]["output_type"] is None


def test_human_agreement_for_run_latest_wins_per_annotator():
    """Eval-run detail's `human_annotations[]` block is latest-wins per
    (item, evaluator, annotator) — matches the summary endpoint. If the
    same annotator labeled the same slot in multiple annotation jobs,
    only the most recent submission surfaces."""
    from routers.annotation_tasks import _human_agreement_for_run
    from unittest.mock import patch

    # Two annotators each labeled the same (item, evaluator) slot
    # twice — once in an older job, once in a newer one. Input is
    # sorted updated_at ASC to match what get_annotations_for_slots
    # returns.
    annotations = [
        {
            "uuid": "ann-aman-old",
            "item_id": "item-1",
            "evaluator_id": "ev-1",
            "annotator_id": "aman",
            "job_id": "job-old",
            "value": {"value": False, "reasoning": "first try"},
            "updated_at": "2026-05-01 10:00:00",
        },
        {
            "uuid": "ann-pri-only",
            "item_id": "item-1",
            "evaluator_id": "ev-1",
            "annotator_id": "pri",
            "job_id": "job-old",
            "value": {"value": True, "reasoning": "p1"},
            "updated_at": "2026-05-01 11:00:00",
        },
        {
            "uuid": "ann-aman-new",
            "item_id": "item-1",
            "evaluator_id": "ev-1",
            "annotator_id": "aman",
            "job_id": "job-new",
            "value": {"value": True, "reasoning": "changed my mind"},
            "updated_at": "2026-05-02 09:00:00",
        },
    ]
    job_runs = [
        {
            "uuid": "run-1",
            "item_id": "item-1",
            "evaluator_id": "ev-1",
            "evaluator_version_id": "v-1",
            "value": {"value": True, "reasoning": "ok"},
        }
    ]
    with patch(
        "routers.annotation_tasks.get_annotations_for_slots",
        return_value=annotations,
    ), patch(
        "routers.annotation_tasks.get_annotators_by_uuids",
        return_value={
            "aman": {"uuid": "aman", "name": "aman"},
            "pri": {"uuid": "pri", "name": "pri"},
        },
    ):
        result = _human_agreement_for_run("task-1", job_runs)

    items = result["items"]
    assert len(items) == 1
    ev_entries = items[0]["evaluators"]
    assert len(ev_entries) == 1
    human_ann = ev_entries[0]["human_annotations"]
    # aman appears once (the newer 'job-new' submission); pri once.
    by_annotator = {h["annotator_id"]: h for h in human_ann}
    assert len(human_ann) == 2
    assert by_annotator["aman"]["annotation_id"] == "ann-aman-new"
    assert by_annotator["aman"]["job_id"] == "job-new"
    assert by_annotator["aman"]["reasoning"] == "changed my mind"
    assert by_annotator["pri"]["annotation_id"] == "ann-pri-only"


def test_build_evaluators_block_applies_binary_default_for_null_rubric():
    """Binary evaluator whose pinned version has output_config=null
    surfaces the Correct/Wrong default — consistent with the other
    evaluator-returning endpoints. Without this, legacy annotation
    eval-run jobs would expose `output_config=null` and the FE would
    have nothing to render labels with (per-run evaluator_version
    blobs were also removed in this PR)."""
    from routers.annotation_tasks import _build_evaluators_block_for_eval_job
    from unittest.mock import patch

    job_details = {
        "evaluators": [
            {
                "evaluator_id": "ev-1",
                "evaluator_version_id": "v-legacy",
                "name": "Safety",
            }
        ]
    }
    raw_runs = [{"evaluator_id": "ev-1", "evaluator_version_id": "v-legacy"}]
    with patch(
        "routers.annotation_tasks.get_evaluator",
        return_value={
            "uuid": "ev-1",
            "name": "Safety",
            "description": "d",
            "output_type": "binary",
            "evaluator_type": "llm",
            "data_type": "text",
        },
    ), patch(
        "routers.annotation_tasks.get_evaluator_version",
        return_value={"uuid": "v-legacy", "version_number": 1, "output_config": None},
    ):
        block = _build_evaluators_block_for_eval_job(job_details, raw_runs)
    assert len(block) == 1
    assert block[0]["output_config"] == {
        "scale": [
            {"value": True, "name": "Correct"},
            {"value": False, "name": "Wrong"},
        ]
    }


def test_evaluator_run_with_specific_item_ids(client):
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_ev(client, h)
    task_uuid = client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    ).json()["uuid"]
    items = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {
                    "payload": {
                        "name": "i1",
                        "chat_history": [],
                        "agent_response": "x",
                    }
                }
            ]
        },
        headers=h,
    ).json()["item_ids"]

    with patch(
        "routers.annotation_tasks.can_start_job", return_value=False
    ):
        resp = client.post(
            f"/annotation-tasks/{task_uuid}/evaluator-runs",
            json={
                "evaluators": [{"evaluator_id": llm_ev["uuid"]}],
                "item_ids": items,
            },
            headers=h,
        )
    assert resp.status_code == 200
