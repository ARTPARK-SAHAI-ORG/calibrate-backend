"""Tests for the annotation-task LIST endpoint's all-time `has_agreement` flag."""

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
            "email": f"at-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


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


def test_list_has_agreement_flag(client):
    h = _signup(client)
    llm_ev = _llm_ev(client, h)

    # Task with no annotations / no comparable pairs.
    empty_task = _create_task(client, h, llm_ev)

    # Task with a human-vs-human pair: two annotators label the same
    # (item, evaluator) slot.
    paired_task = _create_task(client, h, llm_ev)
    item_id = client.post(
        f"/annotation-tasks/{paired_task}/items",
        json={"items": [{"payload": {"name": "i1"}}]},
        headers=h,
    ).json()["item_ids"][0]

    tokens = []
    for _ in range(2):
        annotator = _create_annotator(client, h)
        jobs = client.post(
            f"/annotation-tasks/{paired_task}/jobs",
            json={"annotator_ids": [annotator["uuid"]], "item_ids": [item_id]},
            headers=h,
        ).json()["jobs"]
        tokens.append(jobs[0]["public_token"])

    for token in tokens:
        resp = client.post(
            f"/public/annotation-jobs/{token}/annotations",
            json={
                "item_id": item_id,
                "annotations": [
                    {"evaluator_id": llm_ev["uuid"], "value": {"value": True}}
                ],
            },
        )
        assert resp.status_code == 200

    all_items = client.get("/annotation-tasks", headers=h).json()["items"]
    # Field present on every list item.
    assert all(("has_agreement" in t) for t in all_items)

    by_uuid = {t["uuid"]: t for t in all_items}
    assert by_uuid[empty_task]["has_agreement"] is False
    assert by_uuid[paired_task]["has_agreement"] is True
