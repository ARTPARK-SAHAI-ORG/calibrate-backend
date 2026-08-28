"""Seeding evaluator scores through POST /annotation-tasks/{uuid}/items."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import main as main_mod

    with patch("main.recover_pending_jobs"):
        with TestClient(main_mod.app) as c:
            yield c


def _signup(client):
    suffix = uuid.uuid4().hex[:8]
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "A",
            "last_name": "U",
            "email": f"er-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _llm_ev(client, h):
    """The org's default next-reply evaluator, a binary judge."""
    evs = client.get("/evaluators", headers=h).json()["items"]
    return next(
        e for e in evs if e.get("source_default_slug") == "default-llm-next-reply"
    )


def _rating_ev(client, h):
    evs = client.get("/evaluators", headers=h).json()["items"]
    return next(
        e for e in evs if e.get("source_default_slug") == "default-helpfulness"
    )


def _task(client, h, ev_uuid):
    return client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [ev_uuid],
        },
        headers=h,
    ).json()["uuid"]


def _item(name="i1", **extra):
    return {
        "payload": {
            "name": name,
            "chat_history": [{"role": "user", "content": "hi"}],
            "agent_response": "hello",
        },
        **extra,
    }


def test_scores_are_stored_against_a_finished_job(client):
    h = _signup(client)
    ev = _llm_ev(client, h)
    task = _task(client, h, ev["uuid"])

    body = client.post(
        f"/annotation-tasks/{task}/items",
        json={
            "items": [
                _item("i1", evaluator_results={ev["uuid"]: {"value": True, "reasoning": "good"}}),
                _item("i2"),
            ]
        },
        headers=h,
    ).json()

    assert body["evaluator_result_count"] == 1
    job_uuid = body["evaluator_run_job_id"]

    run = client.get(
        f"/annotation-tasks/{task}/evaluator-runs/{job_uuid}", headers=h
    ).json()
    assert run["status"] == "completed"
    assert run["details"]["item_count"] == 1
    assert [i["uuid"] for i in run["items"]] == [body["item_ids"][0]]
    assert len(run["runs"]) == 1
    assert [e["uuid"] for e in run["evaluators"]] == [ev["uuid"]]

    from db import get_evaluator_runs_for_job

    rows = get_evaluator_runs_for_job(job_uuid)
    assert len(rows) == 1
    assert rows[0]["value"] == {"value": True, "reasoning": "good"}
    assert rows[0]["evaluator_version_id"] == ev["live_version"]["uuid"]


def test_unlinked_evaluator_rejects_and_writes_nothing(client):
    h = _signup(client)
    ev = _llm_ev(client, h)
    task = _task(client, h, ev["uuid"])
    other = client.get("/evaluators", headers=h).json()["items"]
    unlinked = next(e for e in other if e["uuid"] != ev["uuid"])["uuid"]

    r = client.post(
        f"/annotation-tasks/{task}/items",
        json={"items": [_item("i1", evaluator_results={unlinked: {"value": True}})]},
        headers=h,
    )
    assert r.status_code == 400
    assert "not linked" in r.json()["detail"]
    assert client.get(f"/annotation-tasks/{task}/items", headers=h).json() == []


def test_wrong_value_type_for_binary_is_rejected(client):
    h = _signup(client)
    ev = _llm_ev(client, h)
    task = _task(client, h, ev["uuid"])

    r = client.post(
        f"/annotation-tasks/{task}/items",
        json={"items": [_item("i1", evaluator_results={ev["uuid"]: {"value": 3}})]},
        headers=h,
    )
    assert r.status_code == 400
    assert "must be a bool" in r.json()["detail"]


def test_value_outside_the_rating_scale_is_rejected(client):
    h = _signup(client)
    ev = _rating_ev(client, h)
    task = _task(client, h, ev["uuid"])

    r = client.post(
        f"/annotation-tasks/{task}/items",
        json={"items": [_item("i1", evaluator_results={ev["uuid"]: {"value": 99}})]},
        headers=h,
    )
    assert r.status_code == 400
    assert "outside this evaluator's scale" in r.json()["detail"]

    ok = client.post(
        f"/annotation-tasks/{task}/items",
        json={"items": [_item("i1", evaluator_results={ev["uuid"]: {"value": 3}})]},
        headers=h,
    ).json()
    assert ok["evaluator_result_count"] == 1


def test_unknown_version_number_is_rejected(client):
    h = _signup(client)
    ev = _llm_ev(client, h)
    task = _task(client, h, ev["uuid"])

    r = client.post(
        f"/annotation-tasks/{task}/items",
        json={
            "items": [
                _item("i1", evaluator_results={ev["uuid"]: {"value": True, "version_number": 99}})
            ]
        },
        headers=h,
    )
    assert r.status_code == 400
    assert "no version 99" in r.json()["detail"]


def test_tool_call_row_is_rejected(client):
    h = _signup(client)
    ev = _llm_ev(client, h)
    task = _task(client, h, ev["uuid"])

    r = client.post(
        f"/annotation-tasks/{task}/items",
        json={
            "items": [
                {
                    "payload": {
                        "name": "tc",
                        "chat_history": [{"role": "user", "content": "hi"}],
                        "expected_tool_calls": [{"tool": "search", "arguments": {}}],
                    },
                    "evaluator_results": {ev["uuid"]: {"value": True}},
                }
            ]
        },
        headers=h,
    )
    assert r.status_code == 400
    assert "tool call" in r.json()["detail"]


def test_second_submission_keeps_both_rows(client):
    h = _signup(client)
    ev = _llm_ev(client, h)
    task = _task(client, h, ev["uuid"])
    annotator = client.post(
        "/annotators", json={"name": f"a-{uuid.uuid4().hex[:6]}"}, headers=h
    ).json()["uuid"]

    first = client.post(
        f"/annotation-tasks/{task}/items",
        json={"items": [_item("i1", evaluator_results={ev["uuid"]: {"value": True}})]},
        headers=h,
    ).json()
    # Same name reuses the item; annotations force the reuse path.
    second = client.post(
        f"/annotation-tasks/{task}/items",
        json={
            "items": [
                _item(
                    "i1",
                    annotations={ev["uuid"]: {"value": False}},
                    evaluator_results={ev["uuid"]: {"value": False}},
                )
            ],
            "annotator_id": annotator,
        },
        headers=h,
    ).json()

    assert second["item_ids"] == first["item_ids"]
    assert second["evaluator_run_job_id"] != first["evaluator_run_job_id"]

    from db import get_evaluator_runs_for_task

    rows = get_evaluator_runs_for_task(task)
    assert [r["value"]["value"] for r in rows] == [True, False]


def test_items_without_scores_create_no_job(client):
    h = _signup(client)
    ev = _llm_ev(client, h)
    task = _task(client, h, ev["uuid"])

    body = client.post(
        f"/annotation-tasks/{task}/items",
        json={"items": [_item("i1")]},
        headers=h,
    ).json()
    assert "evaluator_run_job_id" not in body
    assert client.get(f"/annotation-tasks/{task}/evaluator-runs", headers=h).json()[
        "runs"
    ] == []


def test_two_items_on_one_evaluator_share_a_single_job_entry(client):
    h = _signup(client)
    ev = _llm_ev(client, h)
    task = _task(client, h, ev["uuid"])

    body = client.post(
        f"/annotation-tasks/{task}/items",
        json={
            "items": [
                _item("i1", evaluator_results={ev["uuid"]: {"value": True}}),
                _item("i2", evaluator_results={ev["uuid"]: {"value": False}}),
            ]
        },
        headers=h,
    ).json()
    assert body["evaluator_result_count"] == 2

    run = client.get(
        f"/annotation-tasks/{task}/evaluator-runs/{body['evaluator_run_job_id']}",
        headers=h,
    ).json()
    assert len(run["evaluators"]) == 1
    assert run["details"]["item_count"] == 2


# ---- Direct unit tests for the malformed shapes the endpoint can't reach ----


class _FakeItem:
    def __init__(self, evaluator_results, payload=None):
        self.payload = payload or {"name": "i1"}
        self.evaluator_results = evaluator_results


@pytest.fixture
def resolve(monkeypatch):
    from fastapi import HTTPException

    import routers.annotation_tasks as mod

    monkeypatch.setattr(
        mod, "get_evaluators_for_annotation_task", lambda _t: [{"uuid": "ev-1"}]
    )

    def _call(evaluator_results, **overrides):
        for name, value in overrides.items():
            monkeypatch.setattr(mod, name, value)
        try:
            mod._resolve_evaluator_results("task-1", [_FakeItem(evaluator_results)])
        except HTTPException as e:
            return e
        return None

    return _call


def test_evaluator_results_must_be_an_object(resolve):
    err = resolve(["not", "a", "dict"])
    assert err.status_code == 400 and "must be an object" in err.detail


def test_entry_must_be_an_object(resolve):
    err = resolve({"ev-1": "yes"})
    assert err.status_code == 400 and "must be an object" in err.detail


def test_entry_must_carry_a_value(resolve):
    err = resolve({"ev-1": {"reasoning": "why"}})
    assert err.status_code == 400 and "missing required key `value`" in err.detail


def test_missing_evaluator_row_is_rejected(resolve):
    err = resolve({"ev-1": {"value": True}}, get_evaluator=lambda _i: None)
    assert err.status_code == 400 and "not found" in err.detail


def test_evaluator_without_a_live_version_is_rejected(resolve):
    err = resolve(
        {"ev-1": {"value": True}},
        get_evaluator=lambda _i: {"uuid": "ev-1", "name": "E", "output_type": "binary"},
    )
    assert err.status_code == 400 and "no live version" in err.detail


def test_missing_live_version_row_is_rejected(resolve):
    err = resolve(
        {"ev-1": {"value": True}},
        get_evaluator=lambda _i: {
            "uuid": "ev-1",
            "name": "E",
            "output_type": "binary",
            "live_version_id": "v-1",
        },
        get_evaluator_version=lambda _v: None,
    )
    assert err.status_code == 400 and "live version not found" in err.detail


def test_rating_value_must_be_a_number(resolve):
    err = resolve(
        {"ev-1": {"value": "high"}},
        get_evaluator=lambda _i: {
            "uuid": "ev-1",
            "name": "E",
            "output_type": "rating",
            "live_version_id": "v-1",
        },
        get_evaluator_version=lambda _v: {"uuid": "v-1", "output_config": None},
    )
    assert err.status_code == 400 and "must be a number" in err.detail


def test_scores_can_name_an_older_version(client):
    h = _signup(client)
    ev = _llm_ev(client, h)
    v1 = ev["live_version"]["uuid"]
    client.post(
        f"/evaluators/{ev['uuid']}/versions",
        json={
            "judge_model": ev["live_version"]["judge_model"],
            "system_prompt": "second version {{criteria}}",
        },
        headers=h,
    )
    task = _task(client, h, ev["uuid"])

    body = client.post(
        f"/annotation-tasks/{task}/items",
        json={
            "items": [
                _item("i1", evaluator_results={ev["uuid"]: {"value": True, "version_number": 1}})
            ]
        },
        headers=h,
    ).json()

    from db import get_evaluator_runs_for_job

    rows = get_evaluator_runs_for_job(body["evaluator_run_job_id"])
    assert [r["evaluator_version_id"] for r in rows] == [v1]
