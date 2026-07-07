"""Integration tests for STT and TTS evaluation routers.

Forces the job queue path (`can_start_job` returns False) so the heavy
background subprocess never spawns. That covers the entire request-validation
and job-creation surface without needing to mock S3 or calibrate CLI.
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
            "first_name": "S",
            "last_name": "U",
            "email": f"sttts-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
        "user_uuid": body["user"]["uuid"],
    }


# ---------------------------------------------------------------------------
# STT /evaluate
# ---------------------------------------------------------------------------


def test_stt_evaluate_no_providers(client):
    auth = _signup(client)
    resp = client.post(
        "/stt/evaluate",
        json={"providers": [], "language": "en"},
        headers=auth["headers"],
    )
    assert resp.status_code == 400


def test_stt_evaluate_legacy_evaluators_field_rejected(client):
    """The model has `extra=forbid` to reject the dropped `evaluators` shape."""
    auth = _signup(client)
    resp = client.post(
        "/stt/evaluate",
        json={
            "providers": ["openai"],
            "language": "en",
            "audio_paths": ["s3://b/k.wav"],
            "texts": ["hi"],
            "evaluators": [{"name": "bogus"}],
        },
        headers=auth["headers"],
    )
    assert resp.status_code == 422


def test_stt_evaluate_queued_path(client, monkeypatch):
    """Force the queue path so no background thread spawns. Hits resolve_evaluators,
    dataset_inputs, create_job."""
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.stt.can_start_job", return_value=False), patch(
        "threading.Thread"
    ):
        resp = client.post(
            "/stt/evaluate",
            json={
                "providers": ["openai"],
                "language": "en",
                "audio_paths": ["s3://b/k.wav"],
                "texts": ["hello"],
            },
            headers=auth["headers"],
        )
        assert resp.status_code == 200
        body = resp.json()
        task_id = body["task_id"]
        assert body["status"] == "queued"

        # GET the queued job
        got = client.get(f"/stt/evaluate/{task_id}", headers=auth["headers"])
        assert got.status_code == 200
        assert got.json()["status"] == "queued"


def test_stt_evaluate_inflight_path(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.stt.can_start_job", return_value=True), patch(
        "routers.stt.threading.Thread"
    ) as thread_mock:
        resp = client.post(
            "/stt/evaluate",
            json={
                "providers": ["openai", "deepgram"],
                "language": "en",
                "audio_paths": ["s3://b/k.wav"],
                "texts": ["hi"],
            },
            headers=auth["headers"],
        )
        assert resp.status_code == 200
        # Thread started but never joined
        thread_mock.return_value.start.assert_called_once()
        body = resp.json()
        # GET in-progress (no output_dir yet → all providers show queued message)
        got = client.get(
            f"/stt/evaluate/{body['task_id']}", headers=auth["headers"]
        )
        assert got.status_code == 200


def test_stt_evaluate_missing_bucket(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.delenv("S3_OUTPUT_BUCKET", raising=False)
    monkeypatch.delenv("OBJECT_STORAGE_MODE", raising=False)
    resp = client.post(
        "/stt/evaluate",
        json={
            "providers": ["openai"],
            "language": "en",
            "audio_paths": ["s3://b/k.wav"],
            "texts": ["hi"],
        },
        headers=auth["headers"],
    )
    assert resp.status_code == 500


def test_stt_evaluate_local_storage_without_bucket(client, monkeypatch, tmp_path):
    auth = _signup(client)
    monkeypatch.setenv("OBJECT_STORAGE_MODE", "local")
    monkeypatch.delenv("S3_OUTPUT_BUCKET", raising=False)
    monkeypatch.setenv("LOCAL_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    with patch("routers.stt.can_start_job", return_value=False), patch(
        "routers.stt.threading.Thread"
    ):
        resp = client.post(
            "/stt/evaluate",
            json={
                "providers": ["openai"],
                "language": "en",
                "audio_paths": ["s3://local-dev-artifacts/stt/media/input.wav"],
                "texts": ["hi"],
            },
            headers=auth["headers"],
        )

    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"


def test_stt_evaluate_invalid_evaluator(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    resp = client.post(
        "/stt/evaluate",
        json={
            "providers": ["openai"],
            "language": "en",
            "audio_paths": ["s3://b/k.wav"],
            "texts": ["hi"],
            "evaluator_uuids": ["00000000-0000-4000-8000-000000000001"],
        },
        headers=auth["headers"],
    )
    assert resp.status_code == 404


def test_stt_evaluate_wrong_evaluator_type(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    # Use the LLM default evaluator; that should be rejected for STT
    evaluators = client.get("/evaluators", headers=auth["headers"]).json()
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")
    resp = client.post(
        "/stt/evaluate",
        json={
            "providers": ["openai"],
            "language": "en",
            "audio_paths": ["s3://b/k.wav"],
            "texts": ["hi"],
            "evaluator_uuids": [llm_ev["uuid"]],
        },
        headers=auth["headers"],
    )
    assert resp.status_code == 400


def test_stt_visibility_toggle(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.stt.can_start_job", return_value=False), patch(
        "threading.Thread"
    ):
        resp = client.post(
            "/stt/evaluate",
            json={
                "providers": ["openai"],
                "language": "en",
                "audio_paths": ["s3://b/k.wav"],
                "texts": ["hi"],
            },
            headers=auth["headers"],
        )
        task_id = resp.json()["task_id"]

    # Toggle on
    on = client.patch(
        f"/stt/evaluate/{task_id}/visibility",
        json={"is_public": True},
        headers=auth["headers"],
    )
    assert on.status_code == 200
    assert on.json()["is_public"] is True
    assert on.json()["share_token"]

    # Toggle off
    off = client.patch(
        f"/stt/evaluate/{task_id}/visibility",
        json={"is_public": False},
        headers=auth["headers"],
    )
    assert off.status_code == 200

    # Unknown task
    missing = client.patch(
        "/stt/evaluate/does-not-exist/visibility",
        json={"is_public": True},
        headers=auth["headers"],
    )
    assert missing.status_code == 404


def test_stt_get_status_unknown(client):
    auth = _signup(client)
    resp = client.get("/stt/evaluate/missing", headers=auth["headers"])
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TTS /evaluate (parallel set)
# ---------------------------------------------------------------------------


def test_tts_evaluate_no_providers(client):
    auth = _signup(client)
    resp = client.post(
        "/tts/evaluate",
        json={"providers": [], "language": "en"},
        headers=auth["headers"],
    )
    assert resp.status_code == 400


def test_tts_evaluate_queued_path(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.tts.can_start_job", return_value=False), patch(
        "threading.Thread"
    ):
        resp = client.post(
            "/tts/evaluate",
            json={
                "providers": ["openai"],
                "language": "en",
                "texts": ["hello", "world"],
            },
            headers=auth["headers"],
        )
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]
        assert resp.json()["status"] == "queued"

    got = client.get(f"/tts/evaluate/{task_id}", headers=auth["headers"])
    assert got.status_code == 200


def test_tts_evaluate_inflight_path(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.tts.can_start_job", return_value=True), patch(
        "routers.tts.threading.Thread"
    ) as thread_mock:
        resp = client.post(
            "/tts/evaluate",
            json={
                "providers": ["openai"],
                "language": "en",
                "texts": ["hello"],
            },
            headers=auth["headers"],
        )
        assert resp.status_code == 200
        thread_mock.return_value.start.assert_called_once()


def test_tts_evaluate_missing_bucket(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.delenv("S3_OUTPUT_BUCKET", raising=False)
    resp = client.post(
        "/tts/evaluate",
        json={
            "providers": ["openai"],
            "language": "en",
            "texts": ["hello"],
        },
        headers=auth["headers"],
    )
    assert resp.status_code == 500


def test_tts_evaluate_legacy_field_rejected(client):
    auth = _signup(client)
    resp = client.post(
        "/tts/evaluate",
        json={
            "providers": ["openai"],
            "language": "en",
            "texts": ["hi"],
            "evaluators": [{"x": 1}],
        },
        headers=auth["headers"],
    )
    assert resp.status_code == 422


def test_tts_visibility_toggle(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.tts.can_start_job", return_value=False), patch(
        "threading.Thread"
    ):
        resp = client.post(
            "/tts/evaluate",
            json={"providers": ["openai"], "language": "en", "texts": ["hi"]},
            headers=auth["headers"],
        )
        task_id = resp.json()["task_id"]

    on = client.patch(
        f"/tts/evaluate/{task_id}/visibility",
        json={"is_public": True},
        headers=auth["headers"],
    )
    assert on.status_code == 200
    off = client.patch(
        f"/tts/evaluate/{task_id}/visibility",
        json={"is_public": False},
        headers=auth["headers"],
    )
    assert off.status_code == 200
    assert (
        client.patch(
            "/tts/evaluate/missing/visibility",
            json={"is_public": True},
            headers=auth["headers"],
        ).status_code
        == 404
    )


def test_tts_get_status_unknown(client):
    auth = _signup(client)
    resp = client.get("/tts/evaluate/missing", headers=auth["headers"])
    assert resp.status_code == 404
