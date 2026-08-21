"""Advanced annotation flow tests — bulk items with annotations, public form
upsert, evaluator-run lifecycle. These exercise the bulkiest remaining
uncovered code paths in routers/annotation_tasks.py and routers/public.py.
"""

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
            "first_name": "A",
            "last_name": "U",
            "email": f"aa-{suffix}@example.com",
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


def _create_task(client, h, llm_ev):
    return client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    ).json()["uuid"]


def _create_annotator(client, h):
    return client.post(
        "/annotators",
        json={"name": f"ann-{uuid.uuid4().hex[:6]}"},
        headers=h,
    ).json()


# ---------------------------------------------------------------------------
# bulk-items with annotations
# ---------------------------------------------------------------------------


def test_bulk_items_with_annotations_validation(client):
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_ev(client, h)
    task_uuid = _create_task(client, h, llm_ev)

    # missing annotator_id
    bad = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {
                    "payload": {"name": "i1"},
                    "annotations": {llm_ev["uuid"]: {"value": True}},
                }
            ]
        },
        headers=h,
    )
    assert bad.status_code == 400

    annotator = _create_annotator(client, h)

    # bad annotator
    bad_ann = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {
                    "payload": {"name": "i1"},
                    "annotations": {llm_ev["uuid"]: {"value": True}},
                }
            ],
            "annotator_id": "00000000-0000-4000-8000-000000000001",
        },
        headers=h,
    )
    assert bad_ann.status_code == 404

    # annotations not a dict
    bad_dict = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {
                    "payload": {"name": "i1"},
                    "annotations": "not-a-dict",
                }
            ],
            "annotator_id": annotator["uuid"],
        },
        headers=h,
    )
    assert bad_dict.status_code == 422  # pydantic rejects the wrong shape

    # Unknown evaluator id
    unknown = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {
                    "payload": {"name": "i1"},
                    "annotations": {"missing-ev": {"value": True}},
                }
            ],
            "annotator_id": annotator["uuid"],
        },
        headers=h,
    )
    assert unknown.status_code == 400

    # Non-dict per-evaluator entry → 400
    bad_value = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {
                    "payload": {"name": "i1"},
                    "annotations": {llm_ev["uuid"]: "not-a-dict"},
                }
            ],
            "annotator_id": annotator["uuid"],
        },
        headers=h,
    )
    assert bad_value.status_code == 400

    # Missing `value` key
    no_value = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {
                    "payload": {"name": "i1"},
                    "annotations": {llm_ev["uuid"]: {"reasoning": "x"}},
                }
            ],
            "annotator_id": annotator["uuid"],
        },
        headers=h,
    )
    assert no_value.status_code == 400


def test_bulk_items_with_annotations_happy(client):
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_ev(client, h)
    task_uuid = _create_task(client, h, llm_ev)
    annotator = _create_annotator(client, h)

    # Submit 2 items, each fully annotated → job auto-completes
    resp = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {
                    "payload": {"name": "i1"},
                    "annotations": {llm_ev["uuid"]: {"value": True}},
                },
                {
                    "payload": {"name": "i2"},
                    "annotations": {llm_ev["uuid"]: {"value": False}},
                },
            ],
            "annotator_id": annotator["uuid"],
        },
        headers=h,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["annotation_job_id"]


def test_bulk_items_partial_annotations(client):
    """One item fully annotated, one without → job moves to in_progress (not completed)."""
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_ev(client, h)
    task_uuid = _create_task(client, h, llm_ev)
    annotator = _create_annotator(client, h)

    resp = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {
                    "payload": {"name": "p1"},
                    "annotations": {llm_ev["uuid"]: {"value": True}},
                },
                {"payload": {"name": "p2"}},  # no annotations
            ],
            "annotator_id": annotator["uuid"],
        },
        headers=h,
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Public annotation-jobs upsert flow
# ---------------------------------------------------------------------------


def test_public_annotation_job_upsert_flow(client):
    """Full annotator-token flow: GET job, POST annotations, completion."""
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_ev(client, h)
    task_uuid = _create_task(client, h, llm_ev)
    annotator = _create_annotator(client, h)
    items = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "i1"}}, {"payload": {"name": "i2"}}]},
        headers=h,
    ).json()["item_ids"]

    # Create a job
    jobs = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"annotator_ids": [annotator["uuid"]], "item_ids": items},
        headers=h,
    ).json()["jobs"]
    public_token = jobs[0]["public_token"]
    job_uuid = jobs[0]["uuid"]

    # GET job
    got = client.get(f"/public/annotation-jobs/{public_token}")
    assert got.status_code == 200
    body = got.json()
    assert body["read_only"] is False

    # POST empty annotations → 400
    bad = client.post(
        f"/public/annotation-jobs/{public_token}/annotations",
        json={"item_id": items[0], "annotations": []},
    )
    assert bad.status_code == 400

    # POST annotation against bad item → 404
    bad_item = client.post(
        f"/public/annotation-jobs/{public_token}/annotations",
        json={
            "item_id": "00000000-0000-4000-8000-000000000001",
            "annotations": [
                {"evaluator_id": llm_ev["uuid"], "value": {"value": True}}
            ],
        },
    )
    assert bad_item.status_code == 404

    # POST annotation with unknown evaluator → 400
    bad_ev = client.post(
        f"/public/annotation-jobs/{public_token}/annotations",
        json={
            "item_id": items[0],
            "annotations": [
                {"evaluator_id": "00000000-0000-4000-8000-000000000001", "value": {"value": True}}
            ],
        },
    )
    assert bad_ev.status_code == 400

    # POST proper annotations for item1
    ok1 = client.post(
        f"/public/annotation-jobs/{public_token}/annotations",
        json={
            "item_id": items[0],
            "annotations": [
                {"evaluator_id": llm_ev["uuid"], "value": {"value": True}},
                {"evaluator_id": None, "value": {"value": "overall"}},
            ],
        },
    )
    assert ok1.status_code == 200
    assert ok1.json()["status"] == "in_progress"

    # POST proper annotations for item2 → completes
    ok2 = client.post(
        f"/public/annotation-jobs/{public_token}/annotations",
        json={
            "item_id": items[1],
            "annotations": [
                {"evaluator_id": llm_ev["uuid"], "value": {"value": False}}
            ],
        },
    )
    assert ok2.status_code == 200
    assert ok2.json()["status"] == "completed"

    # Owner can now toggle view-token sharing (job is completed)
    on = client.patch(
        f"/annotation-tasks/{task_uuid}/jobs/{job_uuid}/visibility",
        json={"is_public": True},
        headers=h,
    )
    assert on.status_code == 200
    view_token = on.json()["view_token"]
    assert view_token

    # View token route returns read_only=True
    view = client.get(f"/public/annotation-jobs/view/{view_token}")
    assert view.status_code == 200
    assert view.json()["read_only"] is True


def test_llm_tool_call_verdict_flow(client):
    """A person labels a tool-call item: mark correct/wrong, and when wrong
    supply the tool calls that should have run. The verdict is stored as the
    item's row-level annotation (evaluator_id=None) with an opaque JSON value —
    no evaluator, no schema change. A job with zero evaluators still completes
    once every item carries that annotation, and the GET returns it to seed."""
    auth = _signup(client)
    h = auth["headers"]

    task_uuid = client.post(
        "/annotation-tasks",
        json={"name": f"tc-{uuid.uuid4().hex[:6]}", "type": "llm-tool-call"},
        headers=h,
    ).json()["uuid"]

    item_id = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {
                    "payload": {
                        "name": "call-1",
                        "chat_history": [{"role": "user", "content": "weather in Paris?"}],
                        "expected_tool_calls": [
                            {"tool": "get_weather", "arguments": {"city": "Paris"}}
                        ],
                        "actual_tool_calls": [
                            {"tool": "get_weather", "arguments": {"city": "London"}}
                        ],
                    }
                }
            ]
        },
        headers=h,
    ).json()["item_ids"][0]

    annotator = _create_annotator(client, h)
    jobs = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"annotator_ids": [annotator["uuid"]], "item_ids": [item_id]},
        headers=h,
    ).json()["jobs"]
    token = jobs[0]["public_token"]

    # Zero evaluators is fine — the form carries none.
    assert client.get(f"/public/annotation-jobs/{token}").json()["evaluators"] == []

    # Boolean pass/fail lives under `value` (the universal binary key) so
    # agreement reads it; the corrected tool calls ride alongside.
    verdict = {
        "value": False,
        "expected_tool_calls": [
            {"tool": "get_weather", "arguments": {"city": "Paris"}}
        ],
    }
    saved = client.post(
        f"/public/annotation-jobs/{token}/annotations",
        json={"item_id": item_id, "annotations": [{"evaluator_id": None, "value": verdict}]},
    )
    assert saved.status_code == 200
    # Single item, single verdict → job auto-completes.
    assert saved.json()["status"] == "completed"

    # Re-save updates in place (no duplicate row-level annotation).
    again = client.post(
        f"/public/annotation-jobs/{token}/annotations",
        json={"item_id": item_id, "annotations": [{"evaluator_id": None, "value": {"value": True}}]},
    )
    assert again.status_code == 200

    # GET returns exactly one row-level annotation, carrying the verdict.
    anns = client.get(f"/public/annotation-jobs/{token}").json()["annotations"]
    row_level = [a for a in anns if a.get("evaluator_id") is None]
    assert len(row_level) == 1
    assert row_level[0]["value"] == {"value": True}


def test_llm_tool_call_verdict_agreement(client):
    """Two people labelling the same tool-call item produce human-vs-human
    pass/fail agreement — the row-level (evaluator_id=None) verdict is its own
    agreement slot, no evaluator involved."""
    auth = _signup(client)
    h = auth["headers"]
    task_uuid = client.post(
        "/annotation-tasks",
        json={"name": f"tc-{uuid.uuid4().hex[:6]}", "type": "llm-tool-call"},
        headers=h,
    ).json()["uuid"]
    item_id = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "call-1"}}]},
        headers=h,
    ).json()["item_ids"][0]

    # Two annotators, same item, opposite verdicts → agreement 0.0 over 1 pair.
    for verdict in (True, False):
        annotator = _create_annotator(client, h)
        token = client.post(
            f"/annotation-tasks/{task_uuid}/jobs",
            json={"annotator_ids": [annotator["uuid"]], "item_ids": [item_id]},
            headers=h,
        ).json()["jobs"][0]["public_token"]
        client.post(
            f"/public/annotation-jobs/{token}/annotations",
            json={"item_id": item_id, "annotations": [{"evaluator_id": None, "value": {"value": verdict}}]},
        )

    hh = client.get(f"/annotation-tasks/{task_uuid}/agreement", headers=h).json()["human_human"]
    assert hh["pair_count"] == 1
    assert hh["current"] == 0.0
