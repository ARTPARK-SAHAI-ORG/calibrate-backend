"""`score` and `labelled` item filters on the task summary and the select-all
bulk actions."""

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
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "A",
            "last_name": "U",
            "email": f"filt-{uuid.uuid4().hex[:8]}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]["uuid"]


def _seed(client):
    """One binary evaluator E and five items:

    A: evaluator true, rater1 true
    B: evaluator false, rater1 true, rater2 false
    C: evaluator true (an older live run said false, a newer run on another
       version says false), only a comment from rater1
    D: no evaluator run, rater1 labelled it then cleared it
    R: evaluator 4, rater1 3 (a rating)
    """
    import db as db_mod
    from annotation_eval_runner import ANNOTATION_EVAL_JOB_TYPE

    h, user_uuid = _signup(client)
    ev = next(
        e
        for e in client.get("/evaluators", headers=h).json()["items"]
        if e.get("evaluator_type") == "llm"
    )
    ev_id, live = ev["uuid"], ev["live_version_id"]
    task = client.post(
        "/annotation-tasks",
        json={"name": f"t-{uuid.uuid4().hex[:6]}", "type": "llm", "evaluator_ids": [ev_id]},
        headers=h,
    ).json()["uuid"]
    names = ["A", "B", "C", "D", "R"]
    ids = dict(
        zip(
            names,
            client.post(
                f"/annotation-tasks/{task}/items",
                json={"items": [{"payload": {"name": n, "chat_history": [], "agent_response": "hi"}} for n in names]},
                headers=h,
            ).json()["item_ids"],
        )
    )
    jobs = []
    for rater in ("rater1", "rater2"):
        annotator = client.post("/annotators", json={"name": rater}, headers=h).json()
        jobs.append(
            client.post(
                f"/annotation-tasks/{task}/jobs",
                json={"annotator_ids": [annotator["uuid"]], "item_ids": list(ids.values())},
                headers=h,
            ).json()["jobs"][0]["uuid"]
        )

    def annotate(job, item, value, evaluator_id=ev_id):
        r = client.post(
            f"/annotation-tasks/{task}/annotations",
            json={"job_id": job, "item_id": ids[item], "evaluator_id": evaluator_id, "value": value},
            headers=h,
        )
        assert r.status_code == 200, r.text

    annotate(jobs[0], "A", {"value": True})
    annotate(jobs[0], "B", {"value": True})
    annotate(jobs[1], "B", {"value": False})
    annotate(jobs[0], "C", {"comment": "looks fine"}, evaluator_id=None)
    annotate(jobs[0], "D", {"value": False})
    annotate(jobs[0], "D", None)
    annotate(jobs[0], "R", {"value": 3})

    eval_job = db_mod.create_job(
        job_type=ANNOTATION_EVAL_JOB_TYPE,
        org_uuid=db_mod.get_personal_org_for_user(user_uuid)["uuid"],
        user_id=user_uuid,
        status="done",
        details={"task_id": task},
    )

    def run(item, value, version, completed_at):
        (run_uuid,) = db_mod.create_evaluator_runs(
            [
                {
                    "job_id": eval_job,
                    "item_id": ids[item],
                    "evaluator_id": ev_id,
                    "evaluator_version_id": version,
                    "status": "completed",
                    "value": {"value": value},
                }
            ]
        )
        with db_mod.get_db_connection() as conn:
            conn.execute(
                "UPDATE evaluator_runs SET completed_at = ? WHERE uuid = ?",
                (completed_at, run_uuid),
            )
            conn.commit()

    other_version = str(uuid.uuid4())
    run("A", True, live, "2026-01-01 00:00:00")
    run("B", False, live, "2026-01-01 00:00:00")
    run("C", False, live, "2026-01-01 00:00:00")
    run("C", True, live, "2026-01-02 00:00:00")
    run("C", False, other_version, "2026-01-03 00:00:00")
    run("R", 4, live, "2026-01-01 00:00:00")
    return {"h": h, "task": task, "ev": ev_id, "ids": ids}


@pytest.fixture(scope="module")
def seeded(client):
    return _seed(client)


def _names(client, seeded, query):
    body = client.get(
        f"/annotation-tasks/{seeded['task']}/summary?live_only=true&{query}",
        headers=seeded["h"],
    )
    assert body.status_code == 200, body.text
    by_id = {v: k for k, v in seeded["ids"].items()}
    return sorted({by_id[r["item_id"]] for r in body.json()["rows"]})


@pytest.mark.parametrize(
    "value, expected",
    [
        ("true:evaluator", ["A", "C"]),
        ("false:evaluator", ["B"]),
        ("1:evaluator", ["A", "C"]),
        ("true.false:evaluator", ["A", "B", "C"]),
        ("true:human", ["A", "B"]),
        ("false:human", ["B"]),
        ("0:human", ["B"]),
        ("true:either", ["A", "B", "C"]),
        ("true", ["A", "B", "C"]),
        ("false", ["B"]),
        ("true:both", ["A"]),
        ("false:both", ["B"]),
        ("4:evaluator", ["R"]),
        ("3:human", ["R"]),
        ("3:evaluator", []),
        ("3.4:both", ["R"]),
    ],
)
def test_score_filter(client, seeded, value, expected):
    assert _names(client, seeded, f"score={seeded['ev']}:{value}") == expected


def test_score_filters_all_must_hold(client, seeded):
    ev = seeded["ev"]
    assert _names(client, seeded, f"score={ev}:true:human&score={ev}:false:evaluator") == ["B"]


@pytest.mark.parametrize("value, expected", [("true", ["A", "B", "R"]), ("false", ["C", "D"])])
def test_labelled_filter(client, seeded, value, expected):
    assert _names(client, seeded, f"labelled={value}") == expected


def test_score_and_labelled_combine(client, seeded):
    assert _names(client, seeded, f"score={seeded['ev']}:true:evaluator&labelled=true") == ["A"]


def test_total_counts_filtered_items(client, seeded):
    body = client.get(
        f"/annotation-tasks/{seeded['task']}/summary?live_only=true&labelled=true&limit=1",
        headers=seeded["h"],
    ).json()
    assert body["pagination"]["total"] == 3
    assert len(body["rows"]) == 1


@pytest.mark.parametrize(
    "value",
    ["{ev}", "{ev}:", "{ev}:maybe", "{ev}:1..2", "{ev}:nan", "{ev}:true:someone", "{ev}:true:human:x", ":true"],
)
def test_malformed_score_is_422(client, seeded, value):
    raw = value.format(ev=seeded["ev"])
    r = client.get(
        f"/annotation-tasks/{seeded['task']}/summary", params={"score": raw}, headers=seeded["h"]
    )
    assert r.status_code == 422
    assert raw in r.json()["detail"]


def test_unlinked_evaluator_is_422(client, seeded):
    raw = f"{uuid.uuid4()}:true"
    r = client.get(
        f"/annotation-tasks/{seeded['task']}/summary", params={"score": raw}, headers=seeded["h"]
    )
    assert r.status_code == 422
    assert "not linked" in r.json()["detail"]


def test_too_many_score_filters_is_422(client, seeded):
    r = client.get(
        f"/annotation-tasks/{seeded['task']}/summary",
        params={"score": [f"{seeded['ev']}:true"] * 21},
        headers=seeded["h"],
    )
    assert r.status_code == 422
    assert "at most 20" in r.json()["detail"]
    ok = client.get(
        f"/annotation-tasks/{seeded['task']}/summary",
        params={"score": [f"{seeded['ev']}:true"] * 20},
        headers=seeded["h"],
    )
    assert ok.status_code == 200


def test_select_all_jobs_honours_filters(client, seeded):
    annotator = client.post("/annotators", json={"name": "rater3"}, headers=seeded["h"]).json()
    r = client.post(
        f"/annotation-tasks/{seeded['task']}/jobs",
        json={
            "annotator_ids": [annotator["uuid"]],
            "select_all": True,
            "score": [f"{seeded['ev']}:true:evaluator"],
            "labelled": False,
        },
        headers=seeded["h"],
    )
    assert r.status_code == 200, r.text
    assert r.json()["jobs"][0]["item_ids"] == [seeded["ids"]["C"]]


def test_select_all_evaluator_run_honours_filters(client, seeded):
    with patch("routers.annotation_tasks.can_start_job", return_value=False):
        r = client.post(
            f"/annotation-tasks/{seeded['task']}/evaluator-runs",
            json={
                "evaluators": [{"evaluator_id": seeded["ev"]}],
                "select_all": True,
                "score": [f"{seeded['ev']}:true:human"],
            },
            headers=seeded["h"],
        )
    assert r.status_code == 200, r.text
    assert r.json()["item_count"] == 2


def test_select_all_bulk_delete_honours_filters(client):
    s = _seed(client)
    r = client.request(
        "DELETE",
        f"/annotation-tasks/{s['task']}/items",
        json={"select_all": True, "score": [f"{s['ev']}:true"], "labelled": True},
        headers=s["h"],
    )
    assert r.status_code == 200, r.text
    assert r.json()["deleted_count"] == 2
    assert _names(client, s, "") == ["C", "D", "R"]

    bad = client.request(
        "DELETE",
        f"/annotation-tasks/{s['task']}/items",
        json={"select_all": True, "score": [f"{s['ev']}:maybe"]},
        headers=s["h"],
    )
    assert bad.status_code == 422


def test_deleted_annotator_labels_are_ignored(client):
    """The summary hides a deleted annotator's labels, so the filters do too."""
    import db as db_mod

    s = _seed(client)
    annotators = client.get(
        f"/annotation-tasks/{s['task']}/summary", headers=s["h"]
    ).json()["annotators"]
    rater2 = next(a["uuid"] for a in annotators if a["name"] == "rater2")
    assert _names(client, s, f"score={s['ev']}:false:human") == ["B"]

    assert db_mod.delete_annotator(rater2)
    assert _names(client, s, f"score={s['ev']}:false:human") == []
    r = client.request(
        "DELETE",
        f"/annotation-tasks/{s['task']}/items",
        json={"select_all": True, "score": [f"{s['ev']}:false:human"]},
        headers=s["h"],
    )
    # Nothing matches, which the bulk actions answer with a 400.
    assert r.status_code == 400, r.text


def test_text_labels_read_as_verdicts():
    """A label uploaded as "yes" / "no" matches the way the Items tab shows it."""
    from annotation_item_filters import filter_items

    items = [{"uuid": "yes"}, {"uuid": "no"}, {"uuid": "maybe"}]
    kept = filter_items(
        items,
        scores=[("ev", {1.0}, "human")],
        labelled=None,
        evaluators=[{"uuid": "ev", "live_version_id": "v"}],
        runs=[],
        annotations=[
            {"item_id": "yes", "evaluator_id": "ev", "annotator_id": "a", "value": {"value": "Yes"}},
            {"item_id": "no", "evaluator_id": "ev", "annotator_id": "a", "value": {"value": "no"}},
            {"item_id": "maybe", "evaluator_id": "ev", "annotator_id": "a", "value": {"value": "maybe"}},
        ],
        annotator_ids={"a"},
    )
    assert [it["uuid"] for it in kept] == ["yes"]
