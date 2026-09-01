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
    assert resp.json()["name"] == "Run 1"

    db_mod.set_agent_test_job_name(job_uuid, "Nightly regression")
    shared = client.get(f"/public/test-run/{token}")
    assert shared.json()["name"] == "Nightly regression"


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
    assert resp.json()["name"] == "Benchmark 1"

    db_mod.set_agent_test_job_name(job_uuid, "Model bake-off")
    shared = client.get(f"/public/benchmark/{token}")
    assert shared.json()["name"] == "Model bake-off"


def _shared_run(client, job_type="llm-unit-test"):
    """A shared run (or benchmark) holding one pass and one fail, returning its
    share token."""
    import db as db_mod

    auth = _user(client)
    agent_uuid = db_mod.create_agent(
        name=f"a-{uuid.uuid4().hex[:6]}",
        org_uuid=auth["org_uuid"],
        user_id=auth["user_uuid"],
    )
    job_uuid = db_mod.create_agent_test_job(
        agent_id=agent_uuid, job_type=job_type, status="done"
    )
    rows = [
        {
            "name": "tc_pass",
            "test_case_id": "tc_pass",
            "passed": True,
            "reasoning": "looks good",
            "output": {"response": "hi"},
            "test_case": {"name": "tc_pass", "history": []},
            "judge_results": None,
        },
        {
            "name": "tc_fail",
            "test_case_id": "tc_fail",
            "passed": False,
            "reasoning": "wrong answer",
            "output": {"response": "nope"},
            "test_case": {"name": "tc_fail", "history": []},
            "judge_results": None,
        },
    ]
    results = (
        {"total_tests": 2, "passed": 1, "failed": 1, "test_results": rows}
        if job_type == "llm-unit-test"
        else {
            "model_results": [
                {
                    "model": "openai/gpt-4.1",
                    "success": True,
                    "message": "Completed",
                    "total_tests": 2,
                    "passed": 1,
                    "failed": 1,
                    "test_results": rows,
                }
            ]
        }
    )
    db_mod.update_agent_test_job(job_uuid, status="done", results=results)
    token = uuid.uuid4().hex
    db_mod.update_agent_test_job_visibility(job_uuid, True, token)
    return token


_SHARED_HEAVY_FIELDS = ("test_case", "output", "judge_results", "inputs")


def test_public_test_run_summary_mode_drops_heavy_keys(client):
    """A shared link carries the same payload a run detail does, so it trims the
    same way."""
    token = _shared_run(client)

    full = client.get(f"/public/test-run/{token}").json()
    assert full["results"][0]["output"] == {"response": "hi"}

    body = client.get(f"/public/test-run/{token}", params={"mode": "summary"}).json()
    assert len(body["results"]) == 2
    for case in body["results"]:
        for field in _SHARED_HEAVY_FIELDS:
            assert field not in case
        assert "test_case_id" in case and "passed" in case
    assert body["results"][1]["reasoning"] == "wrong answer"
    assert body["total_tests"] == 2


def test_public_benchmark_summary_mode_drops_heavy_keys(client):
    token = _shared_run(client, job_type="llm-benchmark")

    model = client.get(
        f"/public/benchmark/{token}", params={"mode": "summary"}
    ).json()["model_results"][0]
    for case in model["test_results"]:
        for field in _SHARED_HEAVY_FIELDS:
            assert field not in case
    assert model["passed"] == 1


def test_public_test_run_reports_totals_and_test_type(client):
    """A shared link opens on the same Summary tab, so it carries the same
    per-evaluator totals and case types."""
    import db as db_mod

    auth = _user(client)
    agent_uuid = db_mod.create_agent(
        name=f"a-{uuid.uuid4().hex[:6]}",
        org_uuid=auth["org_uuid"],
        user_id=auth["user_uuid"],
    )
    evaluator_uuid = str(uuid.uuid4())
    snapshot = [
        {"uuid": evaluator_uuid, "name": "Correctness", "output_type": "binary"}
    ]
    job_uuid = db_mod.create_agent_test_job(
        agent_id=agent_uuid,
        job_type="llm-unit-test",
        details={"evaluators_by_test_id": {"tc_a": snapshot, "tc_b": snapshot}},
    )
    db_mod.update_agent_test_job(
        job_uuid,
        status="done",
        results={
            "test_results": [
                {
                    "name": "tc_a",
                    "test_case_id": "tc_a",
                    "passed": True,
                    "test_case": {"evaluation": {"type": "response"}},
                    "judge_results": [
                        {"evaluator_uuid": evaluator_uuid, "match": True}
                    ],
                },
                {
                    "name": "tc_b",
                    "test_case_id": "tc_b",
                    "passed": False,
                    "test_case": {"evaluation": {"type": "general"}},
                    "judge_results": [
                        {"evaluator_uuid": evaluator_uuid, "match": False}
                    ],
                },
            ]
        },
    )
    token = uuid.uuid4().hex
    db_mod.update_agent_test_job_visibility(job_uuid, True, token)

    body = client.get(
        f"/public/test-run/{token}", params={"mode": "summary"}
    ).json()
    assert [c["test_type"] for c in body["results"]] == ["response", "general"]
    totals = body["evaluator_summary"][0]
    assert (totals["passed"], totals["total"], totals["pass_rate"]) == (1, 2, 50.0)


def test_public_test_run_case_returns_one_case_in_full(client):
    token = _shared_run(client)

    resp = client.get(f"/public/test-run/{token}/results/tc_fail")
    assert resp.status_code == 200
    case = resp.json()
    assert case["output"]["response"] == "nope"
    assert case["test_case"] == {"name": "tc_fail", "history": []}
    assert case["reasoning"] == "wrong answer"

    assert client.get(f"/public/test-run/{token}/results/nope").status_code == 404
    assert client.get("/public/test-run/bad-token/results/tc_fail").status_code == 404


def test_public_benchmark_case_needs_a_model(client):
    token = _shared_run(client, job_type="llm-benchmark")

    assert (
        client.get(f"/public/benchmark/{token}/results/tc_fail").status_code == 400
    )

    ok = client.get(
        f"/public/benchmark/{token}/results/tc_fail",
        params={"model": "openai/gpt-4.1"},
    )
    assert ok.status_code == 200
    assert ok.json()["output"]["response"] == "nope"

    missing = client.get(
        f"/public/benchmark/{token}/results/tc_fail", params={"model": "openai/gpt-9"}
    )
    assert missing.status_code == 404


def test_public_run_case_does_not_cross_job_types(client):
    """A benchmark token must not open the run route, or a share link would read
    a job it was not minted for."""
    token = _shared_run(client, job_type="llm-benchmark")

    assert client.get(f"/public/test-run/{token}/results/tc_fail").status_code == 404


def test_public_test_run_shows_a_tool_call_verdict(client):
    """A tool-call test is judged by diffing the calls, so its verdict is built
    at read time from the evaluator frozen onto the run. A shared link must show
    the same verdict the owner sees, not a blank."""
    import db as db_mod

    auth = _user(client)
    agent_uuid = db_mod.create_agent(
        name=f"a-{uuid.uuid4().hex[:6]}",
        org_uuid=auth["org_uuid"],
        user_id=auth["user_uuid"],
    )
    evaluator_uuid = str(uuid.uuid4())
    job_uuid = db_mod.create_agent_test_job(
        agent_id=agent_uuid,
        job_type="llm-unit-test",
        status="done",
        details={
            "tool_call_evaluator": {
                "uuid": evaluator_uuid,
                "name": "Tool call correctness",
                "output_type": "binary",
                "output_config": {
                    "scale": [
                        {"value": False, "name": "Wrong"},
                        {"value": True, "name": "Correct"},
                    ]
                },
            }
        },
    )
    db_mod.update_agent_test_job(
        job_uuid,
        status="done",
        results={
            "test_results": [
                {
                    "name": "tc_tool",
                    "test_case_id": "tc_tool",
                    "passed": True,
                    "reasoning": "calls match",
                    "output": {"tool_calls": [{"tool": "search"}]},
                    "test_case": {"evaluation": {"type": "tool_call"}},
                    "judge_results": None,
                }
            ]
        },
    )
    token = uuid.uuid4().hex
    db_mod.update_agent_test_job_visibility(job_uuid, True, token)

    body = client.get(f"/public/test-run/{token}").json()
    verdict = body["results"][0]["judge_results"][0]
    assert verdict["evaluator_uuid"] == evaluator_uuid
    assert verdict["value_name"] == "Correct"
    assert [e["name"] for e in body["evaluators"]] == ["Tool call correctness"]


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
