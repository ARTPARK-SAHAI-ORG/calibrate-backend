"""Per-item evaluator lists on annotation tasks.

An item's evaluators are its own saved list (or the task's list when it has
none), minus anything no longer linked to the task. A saved list is never
rewritten when the task's set changes.
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
            "first_name": "I",
            "last_name": "E",
            "email": f"ie-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
        "user_uuid": body["user"]["uuid"],
    }


def _llm_evs(client, h, count):
    """`count` distinct LLM evaluators, in a stable order."""
    evs = [
        e
        for e in client.get("/evaluators", headers=h).json()["items"]
        if e.get("evaluator_type") == "llm"
    ]
    evs.sort(key=lambda e: e["name"])
    assert len(evs) >= count, f"org has only {len(evs)} llm evaluators"
    return evs[:count]


def _create_task(client, h, evaluator_ids):
    return client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": list(evaluator_ids),
        },
        headers=h,
    ).json()["uuid"]


def _create_annotator(client, h):
    return client.post(
        "/annotators",
        json={"name": f"ann-{uuid.uuid4().hex[:6]}"},
        headers=h,
    ).json()


def _items_by_name(client, h, task_uuid):
    resp = client.get(f"/annotation-tasks/{task_uuid}/items", headers=h)
    assert resp.status_code == 200
    return {it["payload"]["name"]: it for it in resp.json()}


def _set_task_evaluators(client, h, task_uuid, evaluator_ids):
    resp = client.put(
        f"/annotation-tasks/{task_uuid}/evaluators",
        json={"evaluator_ids": list(evaluator_ids)},
        headers=h,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _llm_payload(name):
    return {
        "name": name,
        "chat_history": [{"role": "user", "content": "hi"}],
        "agent_response": "hello",
    }


# ---------------------------------------------------------------------------
# Resolution rule
# ---------------------------------------------------------------------------


def test_saved_list_is_filtered_not_rewritten(client):
    """The worked example: item 5 saves [Tone, Accuracy], item 6 follows the
    task. Removing Accuracy from the task narrows item 5 without touching its
    saved row, and re-adding Accuracy brings it back — while Politeness, which
    item 5 never chose, stays out."""
    h = _signup(client)["headers"]
    tone, accuracy, politeness = [e["uuid"] for e in _llm_evs(client, h, 3)]
    task_uuid = _create_task(client, h, [tone, accuracy, politeness])

    client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {"payload": {"name": "i5"}, "evaluator_ids": [tone, accuracy]},
                {"payload": {"name": "i6"}},
            ]
        },
        headers=h,
    )

    items = _items_by_name(client, h, task_uuid)
    assert items["i5"]["evaluator_ids"] == [tone, accuracy]
    assert items["i5"]["effective_evaluator_ids"] == [tone, accuracy]
    assert items["i6"]["evaluator_ids"] is None
    assert items["i6"]["effective_evaluator_ids"] == [tone, accuracy, politeness]

    _set_task_evaluators(client, h, task_uuid, [tone, politeness])
    items = _items_by_name(client, h, task_uuid)
    assert items["i5"]["effective_evaluator_ids"] == [tone]
    # The saved row survives the task change untouched.
    assert items["i5"]["evaluator_ids"] == [tone, accuracy]
    assert items["i6"]["effective_evaluator_ids"] == [tone, politeness]

    _set_task_evaluators(client, h, task_uuid, [tone, accuracy, politeness])
    items = _items_by_name(client, h, task_uuid)
    assert items["i5"]["effective_evaluator_ids"] == [tone, accuracy]
    assert politeness not in items["i5"]["effective_evaluator_ids"]


def test_effective_list_uses_task_order(client):
    """A saved list's own ordering never leaks out — the item renders in the
    task's display order."""
    h = _signup(client)["headers"]
    a, b, c = [e["uuid"] for e in _llm_evs(client, h, 3)]
    task_uuid = _create_task(client, h, [a, b, c])
    item_uuid = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "i1"}, "evaluator_ids": [c, a]}]},
        headers=h,
    ).json()["item_ids"][0]

    one = client.get(
        f"/annotation-tasks/{task_uuid}/items/{item_uuid}", headers=h
    ).json()
    assert one["evaluator_ids"] == [c, a]
    assert one["effective_evaluator_ids"] == [a, c]

    # Reordering the task reorders the item's effective list too.
    _set_task_evaluators(client, h, task_uuid, [c, b, a])
    one = client.get(
        f"/annotation-tasks/{task_uuid}/items/{item_uuid}", headers=h
    ).json()
    assert one["effective_evaluator_ids"] == [c, a]


def test_null_evaluator_ids_behaves_as_before(client):
    """An item that never saved a list is indistinguishable from a pre-feature
    row: it follows the task everywhere and counts as uncustomised."""
    h = _signup(client)["headers"]
    a, b = [e["uuid"] for e in _llm_evs(client, h, 2)]
    task_uuid = _create_task(client, h, [a, b])
    item_uuid = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "legacy"}}]},
        headers=h,
    ).json()["item_ids"][0]

    detail = client.get(f"/annotation-tasks/{task_uuid}", headers=h).json()
    assert detail["customised_item_count"] == 0
    item = next(it for it in detail["items"] if it["uuid"] == item_uuid)
    assert item["evaluator_ids"] is None
    assert item["effective_evaluator_ids"] == [a, b]

    summary = client.get(f"/annotation-tasks/{task_uuid}/summary", headers=h).json()
    assert sorted(r["evaluator_id"] for r in summary["rows"]) == sorted([a, b])


# ---------------------------------------------------------------------------
# Writing per-item lists
# ---------------------------------------------------------------------------


def test_create_items_with_evaluator_ids(client):
    h = _signup(client)["headers"]
    a, b, unlinked = [e["uuid"] for e in _llm_evs(client, h, 3)]
    task_uuid = _create_task(client, h, [a, b])

    ok = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "narrow"}, "evaluator_ids": [b]}]},
        headers=h,
    )
    assert ok.status_code == 200
    items = _items_by_name(client, h, task_uuid)
    assert items["narrow"]["effective_evaluator_ids"] == [b]

    not_linked = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "bad"}, "evaluator_ids": [unlinked]}]},
        headers=h,
    )
    assert not_linked.status_code == 400
    assert unlinked in not_linked.json()["detail"]

    empty = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "empty"}, "evaluator_ids": []}]},
        headers=h,
    )
    assert empty.status_code == 400

    # Neither rejected request created anything.
    assert set(_items_by_name(client, h, task_uuid)) == {"narrow"}


def test_bulk_update_items_three_way_evaluator_ids(client):
    """`evaluator_ids` absent leaves the saved list alone, explicit null goes
    back to inheriting, a list replaces. `payload` and `evaluator_ids` move
    independently."""
    h = _signup(client)["headers"]
    a, b = [e["uuid"] for e in _llm_evs(client, h, 2)]
    task_uuid = _create_task(client, h, [a, b])
    item_uuid = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "i1"}, "evaluator_ids": [a, b]}]},
        headers=h,
    ).json()["item_ids"][0]

    def _item():
        return client.get(
            f"/annotation-tasks/{task_uuid}/items/{item_uuid}", headers=h
        ).json()

    # Payload only: evaluator list untouched.
    resp = client.put(
        f"/annotation-tasks/{task_uuid}/items",
        json={"updates": [{"uuid": item_uuid, "payload": {"name": "renamed"}}]},
        headers=h,
    )
    assert resp.status_code == 200 and resp.json()["updated_count"] == 1
    assert _item()["payload"] == {"name": "renamed"}
    assert _item()["evaluator_ids"] == [a, b]

    # Evaluator list only: payload untouched.
    resp = client.put(
        f"/annotation-tasks/{task_uuid}/items",
        json={"updates": [{"uuid": item_uuid, "evaluator_ids": [b]}]},
        headers=h,
    )
    assert resp.status_code == 200 and resp.json()["updated_count"] == 1
    assert _item()["payload"] == {"name": "renamed"}
    assert _item()["evaluator_ids"] == [b]

    # Explicit null returns the item to following the task.
    resp = client.put(
        f"/annotation-tasks/{task_uuid}/items",
        json={"updates": [{"uuid": item_uuid, "evaluator_ids": None}]},
        headers=h,
    )
    assert resp.status_code == 200
    assert _item()["evaluator_ids"] is None
    assert _item()["effective_evaluator_ids"] == [a, b]

    # A list replaces.
    resp = client.put(
        f"/annotation-tasks/{task_uuid}/items",
        json={"updates": [{"uuid": item_uuid, "evaluator_ids": [a]}]},
        headers=h,
    )
    assert resp.status_code == 200
    assert _item()["evaluator_ids"] == [a]

    # An id the task is not linked to is rejected and writes nothing.
    other = _llm_evs(client, h, 3)[2]["uuid"]
    bad = client.put(
        f"/annotation-tasks/{task_uuid}/items",
        json={"updates": [{"uuid": item_uuid, "evaluator_ids": [other]}]},
        headers=h,
    )
    assert bad.status_code == 400
    assert _item()["evaluator_ids"] == [a]


# ---------------------------------------------------------------------------
# Bulk add / remove
# ---------------------------------------------------------------------------


def test_bulk_item_evaluators_add(client):
    """`add` unions onto a customised item and skips an inheriting one, which
    already gets everything the task has."""
    h = _signup(client)["headers"]
    a, b, c = [e["uuid"] for e in _llm_evs(client, h, 3)]
    task_uuid = _create_task(client, h, [a, b, c])
    created = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {"payload": {"name": "inherits"}},
                {"payload": {"name": "custom"}, "evaluator_ids": [a]},
            ]
        },
        headers=h,
    ).json()["item_ids"]

    resp = client.post(
        f"/annotation-tasks/{task_uuid}/items/evaluators",
        json={"action": "add", "evaluator_ids": [c], "item_ids": created},
        headers=h,
    )
    assert resp.status_code == 200
    assert resp.json() == {"updated_count": 1}

    items = _items_by_name(client, h, task_uuid)
    # The inheriting item stayed inheriting — no row was written for it.
    assert items["inherits"]["evaluator_ids"] is None
    assert items["custom"]["evaluator_ids"] == [a, c]

    # Adding what the item already has is not a write.
    again = client.post(
        f"/annotation-tasks/{task_uuid}/items/evaluators",
        json={"action": "add", "evaluator_ids": [a], "item_ids": created},
        headers=h,
    )
    assert again.json()["updated_count"] == 0


def test_bulk_item_evaluators_remove_materialises_then_drops(client):
    h = _signup(client)["headers"]
    a, b, c = [e["uuid"] for e in _llm_evs(client, h, 3)]
    task_uuid = _create_task(client, h, [a, b, c])
    item_uuid = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "inherits"}}]},
        headers=h,
    ).json()["item_ids"][0]

    resp = client.post(
        f"/annotation-tasks/{task_uuid}/items/evaluators",
        json={"action": "remove", "evaluator_ids": [b], "item_ids": [item_uuid]},
        headers=h,
    )
    assert resp.status_code == 200
    assert resp.json()["updated_count"] == 1

    item = client.get(
        f"/annotation-tasks/{task_uuid}/items/{item_uuid}", headers=h
    ).json()
    assert item["evaluator_ids"] == [a, c]
    assert item["effective_evaluator_ids"] == [a, c]


def test_bulk_item_evaluators_remove_to_empty_writes_nothing(client):
    h = _signup(client)["headers"]
    a, b = [e["uuid"] for e in _llm_evs(client, h, 2)]
    task_uuid = _create_task(client, h, [a, b])
    created = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {"payload": {"name": "keeps"}, "evaluator_ids": [a, b]},
                {"payload": {"name": "empties"}, "evaluator_ids": [a]},
            ]
        },
        headers=h,
    ).json()["item_ids"]

    resp = client.post(
        f"/annotation-tasks/{task_uuid}/items/evaluators",
        json={"action": "remove", "evaluator_ids": [a], "item_ids": created},
        headers=h,
    )
    assert resp.status_code == 400
    assert "no evaluator" in resp.json()["detail"]

    # The whole request was rejected — the item that could have been narrowed
    # is untouched too.
    items = _items_by_name(client, h, task_uuid)
    assert items["keeps"]["evaluator_ids"] == [a, b]
    assert items["empties"]["evaluator_ids"] == [a]


def test_bulk_item_evaluators_select_all_with_filter(client):
    h = _signup(client)["headers"]
    a, b = [e["uuid"] for e in _llm_evs(client, h, 2)]
    task_uuid = _create_task(client, h, [a, b])
    client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {"payload": {"name": "alpha-one"}},
                {"payload": {"name": "alpha-two"}},
                {"payload": {"name": "beta"}},
            ]
        },
        headers=h,
    )

    resp = client.post(
        f"/annotation-tasks/{task_uuid}/items/evaluators",
        json={
            "action": "remove",
            "evaluator_ids": [b],
            "select_all": True,
            "q": "ALPHA",
        },
        headers=h,
    )
    assert resp.status_code == 200
    assert resp.json()["updated_count"] == 2

    items = _items_by_name(client, h, task_uuid)
    assert items["alpha-one"]["evaluator_ids"] == [a]
    assert items["alpha-two"]["evaluator_ids"] == [a]
    assert items["beta"]["evaluator_ids"] is None

    # A filter that matches nothing is a 400, not a silent no-op.
    nothing = client.post(
        f"/annotation-tasks/{task_uuid}/items/evaluators",
        json={
            "action": "remove",
            "evaluator_ids": [b],
            "select_all": True,
            "q": "zzzz",
        },
        headers=h,
    )
    assert nothing.status_code == 400


def test_customised_item_count(client):
    h = _signup(client)["headers"]
    a, b = [e["uuid"] for e in _llm_evs(client, h, 2)]
    task_uuid = _create_task(client, h, [a, b])
    created = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [{"payload": {"name": "i1"}}, {"payload": {"name": "i2"}}]
        },
        headers=h,
    ).json()["item_ids"]

    assert (
        client.get(f"/annotation-tasks/{task_uuid}", headers=h).json()[
            "customised_item_count"
        ]
        == 0
    )
    assert _set_task_evaluators(client, h, task_uuid, [a, b])[
        "customised_item_count"
    ] == 0

    client.put(
        f"/annotation-tasks/{task_uuid}/items",
        json={"updates": [{"uuid": created[0], "evaluator_ids": [a]}]},
        headers=h,
    )
    assert (
        client.get(f"/annotation-tasks/{task_uuid}", headers=h).json()[
            "customised_item_count"
        ]
        == 1
    )
    assert _set_task_evaluators(client, h, task_uuid, [a, b])[
        "customised_item_count"
    ] == 1


# ---------------------------------------------------------------------------
# Labelling jobs
# ---------------------------------------------------------------------------


def test_job_drops_items_with_no_overlapping_evaluator(client):
    h = _signup(client)["headers"]
    a, b = [e["uuid"] for e in _llm_evs(client, h, 2)]
    task_uuid = _create_task(client, h, [a, b])
    annotator = _create_annotator(client, h)
    created = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {"payload": {"name": "only-a"}, "evaluator_ids": [a]},
                {"payload": {"name": "only-b"}, "evaluator_ids": [b]},
            ]
        },
        headers=h,
    ).json()["item_ids"]
    item_a, item_b = created

    resp = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={
            "annotator_ids": [annotator["uuid"]],
            "item_ids": created,
            "evaluator_ids": [b],
        },
        headers=h,
    )
    assert resp.status_code == 200
    job = resp.json()["jobs"][0]
    assert job["item_ids"] == [item_b]
    assert job["item_count"] == 1
    assert job["skipped_item_count"] == 1

    detail = client.get(
        f"/annotation-tasks/{task_uuid}/jobs/{job['uuid']}", headers=h
    ).json()
    assert [it["evaluator_ids"] for it in detail["items"]] == [[b]]

    # Every item dropping leaves nothing to label.
    none_left = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={
            "annotator_ids": [annotator["uuid"]],
            "item_ids": [item_a],
            "evaluator_ids": [b],
        },
        headers=h,
    )
    assert none_left.status_code == 400
    assert "nothing to label" in none_left.json()["detail"]


def test_public_job_payload_carries_per_item_evaluators(client):
    """The form's `evaluators` block stays the job's full union so columns are
    stable; each item says which of them it actually asks for."""
    h = _signup(client)["headers"]
    a, b = [e["uuid"] for e in _llm_evs(client, h, 2)]
    task_uuid = _create_task(client, h, [a, b])
    annotator = _create_annotator(client, h)
    created = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {"payload": {"name": "narrow"}, "evaluator_ids": [a]},
                {"payload": {"name": "wide"}},
            ]
        },
        headers=h,
    ).json()["item_ids"]

    token = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"annotator_ids": [annotator["uuid"]], "item_ids": created},
        headers=h,
    ).json()["jobs"][0]["public_token"]

    body = client.get(f"/public/annotation-jobs/{token}").json()
    assert [e["uuid"] for e in body["evaluators"]] == [a, b]
    by_name = {it["payload"]["name"]: it for it in body["items"]}
    assert by_name["narrow"]["evaluator_ids"] == [a]
    assert by_name["wide"]["evaluator_ids"] == [a, b]


def test_legacy_job_items_fall_back_to_job_evaluators(client):
    """Job-item rows written before the column existed have a NULL
    `evaluator_ids` and must read as the job's full evaluator snapshot."""
    import db as db_mod

    h = _signup(client)["headers"]
    a, b = [e["uuid"] for e in _llm_evs(client, h, 2)]
    task_uuid = _create_task(client, h, [a, b])
    annotator = _create_annotator(client, h)
    item_uuid = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "old"}, "evaluator_ids": [a]}]},
        headers=h,
    ).json()["item_ids"][0]
    job = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"annotator_ids": [annotator["uuid"]], "item_ids": [item_uuid]},
        headers=h,
    ).json()["jobs"][0]

    with db_mod.get_db_connection() as conn:
        conn.execute(
            "UPDATE annotation_job_items SET evaluator_ids = NULL WHERE job_id = ?",
            (job["uuid"],),
        )
        conn.commit()

    body = client.get(f"/public/annotation-jobs/{job['public_token']}").json()
    assert body["items"][0]["evaluator_ids"] == [a, b]


def test_submit_respects_per_item_evaluators(client):
    """Annotators can only answer the evaluators their item asks for, may save
    partially, and complete the job once every applicable pair is filled."""
    h = _signup(client)["headers"]
    a, b = [e["uuid"] for e in _llm_evs(client, h, 2)]
    task_uuid = _create_task(client, h, [a, b])
    annotator = _create_annotator(client, h)
    created = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {"payload": {"name": "narrow"}, "evaluator_ids": [a]},
                {"payload": {"name": "wide"}},
            ]
        },
        headers=h,
    ).json()["item_ids"]
    narrow, wide = created
    token = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"annotator_ids": [annotator["uuid"]], "item_ids": created},
        headers=h,
    ).json()["jobs"][0]["public_token"]

    def _post(item_id, entries):
        return client.post(
            f"/public/annotation-jobs/{token}/annotations",
            json={"item_id": item_id, "annotations": entries},
        )

    off_item = _post(narrow, [{"evaluator_id": b, "value": {"value": True}}])
    assert off_item.status_code == 422
    assert b in off_item.json()["detail"]

    # The narrow item is complete after one answer; the wide one needs two.
    assert _post(narrow, [{"evaluator_id": a, "value": {"value": True}}]).json()[
        "status"
    ] == "in_progress"
    partial = _post(wide, [{"evaluator_id": a, "value": {"value": True}}])
    assert partial.status_code == 200
    assert partial.json()["status"] == "in_progress"
    done = _post(wide, [{"evaluator_id": b, "value": {"value": False}}])
    assert done.json()["status"] == "completed"


# ---------------------------------------------------------------------------
# Evaluator runs
# ---------------------------------------------------------------------------


def test_evaluator_run_drops_non_applicable_pairs(client):
    h = _signup(client)["headers"]
    a, b = [e["uuid"] for e in _llm_evs(client, h, 2)]
    task_uuid = _create_task(client, h, [a, b])
    client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {"payload": _llm_payload("i1"), "evaluator_ids": [a]},
                {"payload": _llm_payload("i2"), "evaluator_ids": [a]},
            ]
        },
        headers=h,
    )

    with patch("routers.annotation_tasks.can_start_job", return_value=False):
        resp = client.post(
            f"/annotation-tasks/{task_uuid}/evaluator-runs",
            json={
                "evaluators": [{"evaluator_id": a}, {"evaluator_id": b}],
                "select_all": True,
            },
            headers=h,
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["evaluator_count"] == 1
    assert body["item_count"] == 2
    assert body["evaluators_with_no_items"] == [b]
    # 2 items x 2 requested evaluators, 2 pairs survive.
    assert body["skipped_pair_count"] == 2

    with patch("routers.annotation_tasks.can_start_job", return_value=False):
        nothing = client.post(
            f"/annotation-tasks/{task_uuid}/evaluator-runs",
            json={"evaluators": [{"evaluator_id": b}], "select_all": True},
            headers=h,
        )
    assert nothing.status_code == 400
    assert "none of the selected items" in nothing.json()["detail"]


# ---------------------------------------------------------------------------
# Summary and agreement
# ---------------------------------------------------------------------------


def _seed_narrowed_task(client, h, user_uuid):
    """Task with two binary-labelled items, human labels and evaluator runs on
    BOTH evaluators, then item "narrow" cut down to evaluator `a`.

    Returns (task_uuid, item_wide, item_narrow, a, b).
    """
    import db as db_mod
    from annotation_eval_runner import ANNOTATION_EVAL_JOB_TYPE

    evs = _llm_evs(client, h, 2)
    a, b = [e["uuid"] for e in evs]
    task_uuid = _create_task(client, h, [a, b])
    created = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [{"payload": {"name": "wide"}}, {"payload": {"name": "narrow"}}]
        },
        headers=h,
    ).json()["item_ids"]
    item_wide, item_narrow = created

    annotator = _create_annotator(client, h)
    job = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"annotator_ids": [annotator["uuid"]], "item_ids": created},
        headers=h,
    ).json()["jobs"][0]
    for item_id in created:
        for ev_id in (a, b):
            assert (
                client.post(
                    f"/annotation-tasks/{task_uuid}/annotations",
                    json={
                        "job_id": job["uuid"],
                        "item_id": item_id,
                        "evaluator_id": ev_id,
                        "value": {"value": True},
                    },
                    headers=h,
                ).status_code
                == 200
            )

    # Evaluator runs agree on `a` everywhere and disagree on `b` everywhere.
    eval_job = db_mod.create_job(
        job_type=ANNOTATION_EVAL_JOB_TYPE,
        org_uuid=db_mod.get_personal_org_for_user(user_uuid)["uuid"],
        user_id=user_uuid,
        status="done",
        details={"task_id": task_uuid},
    )
    live_by_ev = {e["uuid"]: e["live_version_id"] for e in evs}
    db_mod.create_evaluator_runs(
        [
            {
                "job_id": eval_job,
                "item_id": item_id,
                "evaluator_id": ev_id,
                "evaluator_version_id": live_by_ev[ev_id],
                "status": "completed",
                "value": {"value": ev_id == a},
            }
            for item_id in created
            for ev_id in (a, b)
        ]
    )

    resp = client.put(
        f"/annotation-tasks/{task_uuid}/items",
        json={"updates": [{"uuid": item_narrow, "evaluator_ids": [a]}]},
        headers=h,
    )
    assert resp.status_code == 200
    return task_uuid, item_wide, item_narrow, a, b


def test_summary_skips_non_applicable_pairs(client):
    auth = _signup(client)
    h = auth["headers"]
    task_uuid, item_wide, item_narrow, a, b = _seed_narrowed_task(
        client, h, auth["user_uuid"]
    )

    body = client.get(f"/annotation-tasks/{task_uuid}/summary", headers=h).json()
    pairs = {(r["item_id"], r["evaluator_id"]) for r in body["rows"]}
    assert (item_narrow, a) in pairs
    assert (item_narrow, b) not in pairs
    assert (item_wide, a) in pairs and (item_wide, b) in pairs
    # The columns block still lists everything the task links.
    assert {e["uuid"] for e in body["evaluators"]} == {a, b}

    # Only the wide item still disagrees — the narrow item's disagreeing
    # evaluator no longer applies to it, so it is not on the page at all.
    filtered = client.get(
        f"/annotation-tasks/{task_uuid}/summary",
        params={"disagreement_only": True},
        headers=h,
    ).json()
    assert filtered["pagination"]["total"] == 1
    assert {r["item_id"] for r in filtered["rows"]} == {item_wide}
    assert {r["evaluator_id"] for r in filtered["rows"]} == {b}


def test_agreement_excludes_pairs_that_no_longer_apply(client):
    """An annotation on an evaluator later dropped from its item is kept in
    the table but stops counting, and re-adding the evaluator restores it."""
    h = _signup(client)["headers"]
    a, b = [e["uuid"] for e in _llm_evs(client, h, 2)]
    task_uuid = _create_task(client, h, [a, b])
    item_uuid = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "i1"}}]},
        headers=h,
    ).json()["item_ids"][0]

    # Two annotators label the same item on both evaluators: two comparable
    # slots, one annotator pair each.
    for _ in range(2):
        annotator = _create_annotator(client, h)
        job = client.post(
            f"/annotation-tasks/{task_uuid}/jobs",
            json={"annotator_ids": [annotator["uuid"]], "item_ids": [item_uuid]},
            headers=h,
        ).json()["jobs"][0]
        for ev_id in (a, b):
            client.post(
                f"/annotation-tasks/{task_uuid}/annotations",
                json={
                    "job_id": job["uuid"],
                    "item_id": item_uuid,
                    "evaluator_id": ev_id,
                    "value": {"value": True},
                },
                headers=h,
            )

    def _task_pairs():
        return client.get(
            f"/annotation-tasks/{task_uuid}/agreement", headers=h
        ).json()["human_human"]["pair_count"]

    def _trend_pairs():
        return client.get(
            "/annotation-agreement/trend",
            params={"task_id": task_uuid},
            headers=h,
        ).json()["human_human"]["pair_count"]

    assert _task_pairs() == 2
    assert _trend_pairs() == 2

    client.put(
        f"/annotation-tasks/{task_uuid}/items",
        json={"updates": [{"uuid": item_uuid, "evaluator_ids": [a]}]},
        headers=h,
    )
    assert _task_pairs() == 1
    assert _trend_pairs() == 1
    # The rows themselves are untouched — both evaluators still have labels.
    stored = client.get(
        f"/annotation-tasks/{task_uuid}/items/{item_uuid}/annotations", headers=h
    ).json()
    assert {ann["evaluator_id"] for ann in stored} == {a, b}

    client.put(
        f"/annotation-tasks/{task_uuid}/items",
        json={"updates": [{"uuid": item_uuid, "evaluator_ids": [a, b]}]},
        headers=h,
    )
    assert _task_pairs() == 2
    assert _trend_pairs() == 2
