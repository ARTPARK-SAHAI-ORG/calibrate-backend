"""Every run endpoint refuses a request above the workspace's max rows per eval."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from conftest import NONEXISTENT_UUID


@pytest.fixture(scope="module")
def app():
    import main as main_mod

    return main_mod.app


@pytest.fixture(scope="module")
def client(app):
    with patch("main.recover_pending_jobs"):
        with TestClient(app) as c:
            yield c


@pytest.fixture(autouse=True)
def cap_of_one(monkeypatch):
    """Cap every workspace at one row so two rows is always over the limit."""
    import routers.org_limits as org_limits

    monkeypatch.setattr(org_limits, "DEFAULT_MAX_ROWS_PER_EVAL", 1)


def _signup(client):
    suffix = uuid.uuid4().hex[:8]
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "L",
            "last_name": "M",
            "email": f"lim-{suffix}@example.com",
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


def _create_agent(client, h):
    name = f"a-{uuid.uuid4().hex[:6]}"
    created = client.post(
        "/agents", json={"name": name, "type": "agent"}, headers=h
    ).json()
    return {**created, "name": name}


def _create_test(client, h):
    return client.post(
        "/tests",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "response",
            "config": {"history": [], "evaluation": {"type": "response"}},
            "evaluators": [{"evaluator_uuid": _llm_ev(client, h)["uuid"]}],
        },
        headers=h,
    ).json()


def _agent_with_tests(client, h, count):
    agent = _create_agent(client, h)
    tests = [_create_test(client, h) for _ in range(count)]
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [t["uuid"] for t in tests]},
        headers=h,
    )
    return agent, tests


def test_helper_allows_up_to_the_cap_and_rejects_above_it(monkeypatch):
    from fastapi import HTTPException
    import routers.org_limits as org_limits

    monkeypatch.setattr(org_limits, "DEFAULT_MAX_ROWS_PER_EVAL", 3)
    org_limits.enforce_max_rows_per_eval("org-uuid", 3)
    with pytest.raises(HTTPException) as exc:
        org_limits.enforce_max_rows_per_eval("org-uuid", 4)
    assert exc.value.status_code == 400
    assert "4 rows" in exc.value.detail and "limit of 3" in exc.value.detail


def test_helper_prefers_the_workspace_limit_over_the_default(client, monkeypatch):
    import db as db_mod
    import routers.org_limits as org_limits
    from routers.org_limits import OrgLimits

    auth = _signup(client)
    org_uuid = db_mod.get_personal_org_for_user(auth["user_uuid"])["uuid"]
    db_mod.create_org_limits(org_uuid=org_uuid, limits=OrgLimits(max_rows_per_eval=5))
    assert org_limits.effective_max_rows_per_eval(org_uuid) == 5
    org_limits.enforce_max_rows_per_eval(org_uuid, 5)


def test_run_counts_linked_tests(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    agent, tests = _agent_with_tests(client, h, 2)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")

    resp = client.post(f"/agent-tests/agent/{agent['uuid']}/run", json={}, headers=h)
    assert resp.status_code == 400
    assert "limit of 1" in resp.json()["detail"]

    # One explicit test is within the cap. Started rather than queued, so this
    # leaves no queued job behind for the shared job-queue tests to pick up.
    with patch("routers.agent_tests.can_start_agent_test_job", return_value=True), patch(
        "threading.Thread"
    ):
        ok = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run",
            json={"test_uuids": [tests[0]["uuid"]]},
            headers=h,
        )
    assert ok.status_code == 200


def test_benchmark_counts_tests_times_models(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    agent, tests = _agent_with_tests(client, h, 1)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")

    resp = client.post(
        f"/agent-tests/agent/{agent['uuid']}/benchmark",
        json={"models": ["openai/gpt-4.1", "openai/gpt-4o"]},
        headers=h,
    )
    assert resp.status_code == 400
    assert "would process 2 rows" in resp.json()["detail"]


def test_stt_evaluate_counts_texts(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    resp = client.post(
        "/stt/evaluate",
        json={
            "providers": ["openai"],
            "language": "en",
            "audio_paths": ["s3://b/1.wav", "s3://b/2.wav"],
            "texts": ["one", "two"],
        },
        headers=auth["headers"],
    )
    assert resp.status_code == 400
    assert "limit of 1" in resp.json()["detail"]


def test_tts_evaluate_counts_texts(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    resp = client.post(
        "/tts/evaluate",
        json={"providers": ["openai"], "language": "en", "texts": ["one", "two"]},
        headers=auth["headers"],
    )
    assert resp.status_code == 400
    assert "limit of 1" in resp.json()["detail"]


def test_simulation_run_counts_personas_times_scenarios(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    personas = [
        client.post(
            "/personas",
            json={"name": f"p-{uuid.uuid4().hex[:6]}", "description": "d"},
            headers=h,
        ).json()
        for _ in range(2)
    ]
    scenario = client.post(
        "/scenarios",
        json={"name": f"s-{uuid.uuid4().hex[:6]}", "description": "d"},
        headers=h,
    ).json()
    simulation = client.post(
        "/simulations",
        json={
            "name": f"sim-{uuid.uuid4().hex[:6]}",
            "agent_uuid": agent["uuid"],
            "persona_uuids": [p["uuid"] for p in personas],
            "scenario_uuids": [scenario["uuid"]],
        },
        headers=h,
    ).json()

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    resp = client.post(
        f"/simulations/{simulation['uuid']}/run", json={"type": "text"}, headers=h
    )
    assert resp.status_code == 400
    assert "would process 2 rows" in resp.json()["detail"]


def test_evaluator_run_counts_items_times_evaluators(client):
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
                        "name": f"i{i}",
                        "chat_history": [{"role": "user", "content": "hi"}],
                        "agent_response": "hi back",
                    }
                }
                for i in range(2)
            ]
        },
        headers=h,
    )

    resp = client.post(
        f"/annotation-tasks/{task_uuid}/evaluator-runs",
        json={"evaluators": [{"evaluator_id": llm_ev["uuid"]}], "select_all": True},
        headers=h,
    )
    assert resp.status_code == 400
    assert "would process 2 rows" in resp.json()["detail"]


def _failed_eval_job(auth, *, job_type, details):
    import db as db_mod

    org_uuid = db_mod.get_personal_org_for_user(auth["user_uuid"])["uuid"]
    return db_mod.create_job(
        job_type=job_type,
        org_uuid=org_uuid,
        user_id=auth["user_uuid"],
        status="failed",
        details=details,
        results={"error": "failed"},
    )


@pytest.mark.parametrize(
    "prefix, job_type, details",
    [
        (
            "stt",
            "stt-eval",
            {
                "providers": ["openai"],
                "language": "en",
                "texts": ["one", "two"],
                "audio_paths": ["s3://b/1.wav", "s3://b/2.wav"],
            },
        ),
        (
            "tts",
            "tts-eval",
            {"providers": ["openai"], "language": "en", "texts": ["one", "two"]},
        ),
    ],
)
def test_retry_counts_the_rows_it_would_rerun(
    client, monkeypatch, prefix, job_type, details
):
    import db as db_mod

    auth = _signup(client)
    task_id = _failed_eval_job(auth, job_type=job_type, details=details)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")

    resp = client.post(
        f"/{prefix}/evaluate/{task_id}/retry", headers=auth["headers"]
    )
    assert resp.status_code == 400
    assert "limit of 1" in resp.json()["detail"]

    # The refused retry leaves the original run as it was.
    job = db_mod.get_job(task_id)
    assert job["status"] == "failed"
    assert job["results"] == {"error": "failed"}


def test_batch_run_skips_an_agent_over_the_limit(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    over, _ = _agent_with_tests(client, h, 2)
    under, _ = _agent_with_tests(client, h, 1)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")

    with patch("routers.agent_tests.can_start_agent_test_job", return_value=True), patch(
        "threading.Thread"
    ):
        resp = client.post(
            "/agent-tests/run",
            json={"agent_names": [over["name"], under["name"]]},
            headers=h,
        )
    assert resp.status_code == 200
    body = resp.json()
    # The agent within the limit still runs; only the oversized one is skipped.
    assert [r["agent_uuid"] for r in body["runs"]] == [under["uuid"]]
    assert body["skipped"] == [
        {
            "agent_name": over["name"],
            "agent_uuid": over["uuid"],
            "reason": "over_row_limit",
        }
    ]


def test_run_counts_a_repeated_test_once(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    agent, tests = _agent_with_tests(client, h, 1)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")

    with patch("routers.agent_tests.can_start_agent_test_job", return_value=True), patch(
        "threading.Thread"
    ):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run",
            json={"test_uuids": [tests[0]["uuid"], tests[0]["uuid"]]},
            headers=h,
        )
    assert resp.status_code == 200

    # An unknown uuid is still refused, repeat or not.
    missing = client.post(
        f"/agent-tests/agent/{agent['uuid']}/run",
        json={"test_uuids": [tests[0]["uuid"], NONEXISTENT_UUID]},
        headers=h,
    )
    assert missing.status_code == 404


# ---------------------------------------------------------------------------
# The cap lifts when the workspace pays for the run with its own provider keys
# ---------------------------------------------------------------------------


@pytest.fixture
def own_keys(monkeypatch):
    """Store this workspace's own keys for the providers a test names."""
    from cryptography.fernet import Fernet

    import provider_keys as pk

    monkeypatch.setenv(pk.ENCRYPTION_KEY_ENV, Fernet.generate_key().decode())

    def _store(client, auth, *providers):
        for provider in providers:
            values = {
                env_var: (
                    '{"client_email": "r@p.iam.gserviceaccount.com"}'
                    if pk.env_var_kind(env_var) == pk.FILE
                    else f"wk-{env_var.lower()}"
                )
                for env_var in pk.provider_env_vars(provider)
            }
            resp = client.put(
                f"/provider-keys/{provider}",
                json={"values": values},
                headers=auth["headers"],
            )
            assert resp.status_code == 200, resp.text

    return _store


def test_stt_stays_capped_when_only_the_transcriber_is_paid_for(
    client, monkeypatch, own_keys
):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    own_keys(client, auth, "openai")

    # The Sarvam judge bundle bills OpenRouter, which this workspace has not paid for.
    resp = client.post(
        "/stt/evaluate",
        json={
            "providers": ["openai"],
            "language": "en",
            "audio_paths": ["s3://b/1.wav", "s3://b/2.wav"],
            "texts": ["one", "two"],
            "sarvam_judges": True,
        },
        headers=auth["headers"],
    )
    assert resp.status_code == 400


def test_stt_is_uncapped_once_every_provider_it_calls_is_paid_for(
    client, monkeypatch, own_keys
):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    own_keys(client, auth, "openai", "openrouter")

    resp = client.post(
        "/stt/evaluate",
        json={
            "providers": ["openai"],
            "language": "en",
            "audio_paths": ["s3://b/1.wav", "s3://b/2.wav"],
            "texts": ["one", "two"],
            "sarvam_judges": True,
        },
        headers=auth["headers"],
    )
    assert resp.status_code == 200, resp.text


def test_tts_is_uncapped_once_its_providers_are_paid_for(client, monkeypatch, own_keys):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    own_keys(client, auth, "openai", "openrouter")

    resp = client.post(
        "/tts/evaluate",
        json={"providers": ["openai"], "language": "en", "texts": ["one", "two"]},
        headers=auth["headers"],
    )
    assert resp.status_code == 200, resp.text


def test_the_reported_cap_reflects_the_providers_a_run_would_call(
    client, monkeypatch, own_keys
):
    auth = _signup(client)
    own_keys(client, auth, "sarvam")

    def cap(*providers):
        query = "".join(f"?providers={p}" if i == 0 else f"&providers={p}"
                        for i, p in enumerate(providers))
        return client.get(
            f"/org-limits/me/max-rows-per-eval{query}", headers=auth["headers"]
        ).json()["max_rows_per_eval"]

    assert cap() == 1
    assert cap("sarvam") is None
    assert cap("sarvam", "openrouter") == 1
