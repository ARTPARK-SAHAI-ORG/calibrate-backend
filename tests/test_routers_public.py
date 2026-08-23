"""Integration tests for /public routes.

The /public routes do not require auth. They resolve share_tokens against
the various job tables. We seed jobs directly via db.* and toggle visibility
via update_job_visibility / update_agent_test_job_visibility / etc.
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


def _user(client):
    """Sign up a fresh user and return `(user_uuid, org_uuid)`.

    Most tests don't care about distinguishing the two, but the multi-tenant
    DB layer requires `org_uuid` on every `create_*` call. Returning both keeps
    each test self-contained.
    """
    import db as _db

    suffix = uuid.uuid4().hex[:8]
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "P",
            "last_name": "U",
            "email": f"pub-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    user_uuid = body["user"]["uuid"]
    org = _db.get_personal_org_for_user(user_uuid)
    return {"user_uuid": user_uuid, "org_uuid": org["uuid"]}


# ---------------------------------------------------------------------------
# Defaults / share token gate
# ---------------------------------------------------------------------------


def test_public_evaluators_defaults_requires_valid_token(client):
    resp = client.get(
        "/public/evaluators/defaults", params={"share_token": "missing"}
    )
    assert resp.status_code == 404


def test_public_evaluators_defaults_token_validation(client):
    """When the token is valid we should get the seed list. Use an STT job we
    just made public."""
    import db as db_mod

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    job_uuid = db_mod.create_job(
        job_type="stt-eval",
        org_uuid=org_uuid,
        user_id=user_id,
        status="done",
        details={"providers": ["openai"], "language": "en"},
        results={"provider_results": []},
    )
    token = uuid.uuid4().hex
    db_mod.update_job_visibility(job_uuid, True, token)

    resp = client.get(
        "/public/evaluators/defaults", params={"share_token": token}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list) and body

    # Filtered by types
    filtered = client.get(
        "/public/evaluators/defaults",
        params={"share_token": token, "types": "stt,llm"},
    )
    assert filtered.status_code == 200

    # `llm-general` is an accepted filter value and returns the seeded
    # default-llm-general evaluator.
    general = client.get(
        "/public/evaluators/defaults",
        params={"share_token": token, "types": "llm-general"},
    )
    assert general.status_code == 200
    general_body = general.json()
    assert general_body
    assert all(e["evaluator_type"] == "llm-general" for e in general_body)
    assert any(e["name"] == "Output correctness" for e in general_body)

    # `tool-call` is an accepted filter value.
    tool_call = client.get(
        "/public/evaluators/defaults",
        params={"share_token": token, "types": "tool-call"},
    )
    assert tool_call.status_code == 200

    # Invalid types value → 400
    bad = client.get(
        "/public/evaluators/defaults",
        params={"share_token": token, "types": "bogus"},
    )
    assert bad.status_code == 400

    # Empty types value passes (returns full list)
    empty = client.get(
        "/public/evaluators/defaults",
        params={"share_token": token, "types": ""},
    )
    assert empty.status_code == 200


# ---------------------------------------------------------------------------
# Public STT / TTS / annotation-eval / sim / agent-test / benchmark — token unknown
# ---------------------------------------------------------------------------


def test_public_unknown_token_404(client):
    for path in [
        "/public/stt/none",
        "/public/tts/none",
        "/public/test-run/none",
        "/public/benchmark/none",
        "/public/simulation-run/none",
        "/public/annotation-eval/none",
        "/public/annotation-jobs/none",
        "/public/annotation-jobs/view/none",
    ]:
        r = client.get(path)
        assert r.status_code == 404, path


# ---------------------------------------------------------------------------
# Public STT / TTS with valid token
# ---------------------------------------------------------------------------


def test_public_stt_valid_token(client):
    import db as db_mod

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    job_uuid = db_mod.create_job(
        job_type="stt-eval",
        org_uuid=org_uuid,
        user_id=user_id,
        status="done",
        details={
            "providers": ["openai"],
            "language": "en",
            "audio_paths": [],
        },
        results={"provider_results": [], "leaderboard_summary": None},
    )
    token = uuid.uuid4().hex
    db_mod.update_job_visibility(job_uuid, True, token)
    resp = client.get(f"/public/stt/{token}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == job_uuid


def test_public_tts_valid_token(client):
    import db as db_mod

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    job_uuid = db_mod.create_job(
        job_type="tts-eval",
        org_uuid=org_uuid,
        user_id=user_id,
        status="done",
        details={"providers": ["openai"], "language": "en"},
        results={"provider_results": [], "leaderboard_summary": None},
    )
    token = uuid.uuid4().hex
    db_mod.update_job_visibility(job_uuid, True, token)
    resp = client.get(f"/public/tts/{token}")
    assert resp.status_code == 200


def test_public_test_run_valid_token(client):
    import db as db_mod

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    agent_uuid = db_mod.create_agent(
        name=f"a-{uuid.uuid4().hex[:6]}", org_uuid=org_uuid, user_id=user_id
    )
    job_uuid = db_mod.create_agent_test_job(
        agent_id=agent_uuid, job_type="llm-unit-test", status="done"
    )
    token = uuid.uuid4().hex
    db_mod.update_agent_test_job_visibility(job_uuid, True, token)
    resp = client.get(f"/public/test-run/{token}")
    assert resp.status_code == 200


def test_public_benchmark_valid_token(client):
    import db as db_mod

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    agent_uuid = db_mod.create_agent(
        name=f"a-{uuid.uuid4().hex[:6]}", org_uuid=org_uuid, user_id=user_id
    )
    job_uuid = db_mod.create_agent_test_job(
        agent_id=agent_uuid, job_type="llm-benchmark", status="done"
    )
    token = uuid.uuid4().hex
    db_mod.update_agent_test_job_visibility(job_uuid, True, token)
    resp = client.get(f"/public/benchmark/{token}")
    assert resp.status_code == 200


def test_public_simulation_run_valid_token(client):
    import db as db_mod

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    sim_uuid = db_mod.create_simulation(
        name=f"sim-{uuid.uuid4().hex[:6]}", org_uuid=org_uuid, user_id=user_id
    )
    job_uuid = db_mod.create_simulation_job(
        simulation_id=sim_uuid, job_type="text", status="done"
    )
    token = uuid.uuid4().hex
    db_mod.update_simulation_job_visibility(job_uuid, True, token)
    resp = client.get(f"/public/simulation-run/{token}")
    assert resp.status_code == 200


def test_public_annotation_eval_must_be_done(client):
    """The endpoint refuses to serve in-progress or failed annotation-eval jobs."""
    import db as db_mod
    from annotation_eval_runner import ANNOTATION_EVAL_JOB_TYPE

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    # Create a task to host the job
    task_uuid = db_mod.create_annotation_task(
        name=f"t-{uuid.uuid4().hex[:6]}",
        type="llm",
        org_uuid=org_uuid,
        user_id=user_id,
    )
    job_uuid = db_mod.create_job(
        job_type=ANNOTATION_EVAL_JOB_TYPE,
        org_uuid=org_uuid,
        user_id=user_id,
        status="in_progress",
        details={"task_id": task_uuid, "evaluators": []},
    )
    token = uuid.uuid4().hex
    db_mod.update_job_visibility(job_uuid, True, token)
    # Not done → 404
    resp = client.get(f"/public/annotation-eval/{token}")
    assert resp.status_code == 404

    # Mark done → 200
    db_mod.update_job(job_uuid, status="done")
    resp = client.get(f"/public/annotation-eval/{token}")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Labelling form: tool-call rows
# ---------------------------------------------------------------------------


_TOOL_CALL_PAYLOAD = {
    "chat_history": [{"role": "user", "content": "book it"}],
    "tool_calls": [{"tool": "book", "arguments": {"day": "mon"}}],
}
_TEXT_PAYLOAD = {
    "chat_history": [{"role": "user", "content": "hi"}],
    "agent_response": "hello",
}


def _labelling_job(client, payloads, evaluator_types):
    """Seed a task with `payloads` as items and one evaluator per entry in
    `evaluator_types`, then snapshot both into a labelling job. Returns the
    annotator token, item uuids, and evaluator uuids in the order given."""
    import db as db_mod

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    task_uuid = db_mod.create_annotation_task(
        name=f"t-{uuid.uuid4().hex[:6]}",
        type="llm",
        org_uuid=org_uuid,
        user_id=user_id,
    )
    item_uuids = db_mod.create_annotation_items(
        task_uuid, [{"payload": p} for p in payloads]
    )
    evaluator_uuids = []
    for ev_type in evaluator_types:
        ev_uuid = db_mod.create_evaluator(
            name=f"ev-{uuid.uuid4().hex[:6]}",
            evaluator_type=ev_type,
            org_uuid=org_uuid,
            owner_user_id=user_id,
        )
        db_mod.add_evaluator_to_annotation_task(task_uuid, ev_uuid)
        evaluator_uuids.append(ev_uuid)
    annotator_uuid = db_mod.create_annotator(
        name=f"a-{uuid.uuid4().hex[:6]}", org_uuid=org_uuid, user_id=user_id
    )
    token = uuid.uuid4().hex
    db_mod.create_annotation_job(
        task_id=task_uuid,
        annotator_id=annotator_uuid,
        item_uuids=item_uuids,
        public_token=token,
    )
    return token, item_uuids, evaluator_uuids


def _save(client, token, item_uuid, evaluator_uuid):
    resp = client.post(
        f"/public/annotation-jobs/{token}/annotations",
        json={
            "item_id": item_uuid,
            "annotations": [
                {"evaluator_id": evaluator_uuid, "value": {"value": True}}
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["status"]


def test_public_annotation_job_marks_tool_call_items(client):
    token, item_uuids, evaluator_uuids = _labelling_job(
        client, [_TOOL_CALL_PAYLOAD, _TEXT_PAYLOAD], ["tool-call", "llm"]
    )

    body = client.get(f"/public/annotation-jobs/{token}").json()
    by_uuid = {it["uuid"]: it for it in body["items"]}
    assert by_uuid[item_uuids[0]]["is_tool_call"] is True
    assert by_uuid[item_uuids[1]]["is_tool_call"] is False
    # The evaluator entries carry the type the browser pairs the flag against.
    assert [ev["evaluator_type"] for ev in body["evaluators"]] == [
        "tool-call",
        "llm",
    ]


def test_mixed_job_completes_on_the_evaluator_each_row_shows(client):
    token, item_uuids, evaluator_uuids = _labelling_job(
        client, [_TOOL_CALL_PAYLOAD, _TEXT_PAYLOAD], ["tool-call", "llm"]
    )
    tool_item, text_item = item_uuids
    tool_ev, llm_ev = evaluator_uuids

    assert _save(client, token, text_item, llm_ev) == "in_progress"
    # The text evaluator is not required on the tool-call row, so answering it
    # there leaves the row's own tool-call slot open.
    assert _save(client, token, tool_item, llm_ev) == "in_progress"
    assert _save(client, token, tool_item, tool_ev) == "completed"


def test_tool_call_answer_does_not_substitute_on_a_text_row(client):
    token, item_uuids, evaluator_uuids = _labelling_job(
        client, [_TEXT_PAYLOAD], ["tool-call", "llm"]
    )
    tool_ev, llm_ev = evaluator_uuids

    assert _save(client, token, item_uuids[0], tool_ev) == "in_progress"
    assert _save(client, token, item_uuids[0], llm_ev) == "completed"


def test_all_tool_call_job_completes_on_the_tool_call_evaluator_alone(client):
    token, item_uuids, evaluator_uuids = _labelling_job(
        client, [_TOOL_CALL_PAYLOAD, _TOOL_CALL_PAYLOAD], ["tool-call", "llm"]
    )
    tool_ev = evaluator_uuids[0]

    assert _save(client, token, item_uuids[0], tool_ev) == "in_progress"
    assert _save(client, token, item_uuids[1], tool_ev) == "completed"


def test_job_completes_when_a_row_has_nothing_to_answer(client):
    """A task carrying only the tool-call evaluator draws nothing on its text
    rows, so those rows must not hold the job open. Answering every tool-call
    row finishes it."""
    token, item_uuids, evaluator_uuids = _labelling_job(
        client, [_TOOL_CALL_PAYLOAD, _TEXT_PAYLOAD], ["tool-call"]
    )
    tool_call_item, text_item = item_uuids

    body = client.get(f"/public/annotation-jobs/{token}").json()
    assert [ev["evaluator_type"] for ev in body["evaluators"]] == ["tool-call"]

    assert _save(client, token, tool_call_item, evaluator_uuids[0]) == "completed"
    # The text row was never touched and could not have been.
    annotated_items = {
        a["item_id"]
        for a in client.get(f"/public/annotation-jobs/{token}").json()["annotations"]
    }
    assert text_item not in annotated_items
