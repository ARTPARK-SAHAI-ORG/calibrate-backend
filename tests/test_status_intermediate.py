"""Tests for the intermediate-result reading branches in stt/tts/sim GET endpoints.

We create a job directly via db, drop a simulated output_dir on disk with
provider results, then hit the GET endpoint to drive the in-progress reader.
"""

from __future__ import annotations

import csv
import json
import os
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import db


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
            "first_name": "Z",
            "last_name": "Z",
            "email": f"zz-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    user_uuid = body["user"]["uuid"]
    org_uuid = db.get_personal_org_for_user(user_uuid)["uuid"]
    return {
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
        "user_uuid": user_uuid,
        "org_uuid": org_uuid,
    }


def _seed_stt_output(tmp_path: Path, providers, total: int = 1):
    for p in providers:
        sub = tmp_path / f"{p}_results"
        sub.mkdir(parents=True)
        with open(sub / "results.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "gt", "pred"])
            for i in range(total):
                writer.writerow([f"audio_{i+1}", "ref", "pred"])
        with open(sub / "metrics.json", "w") as f:
            json.dump({"wer": 0.1}, f)


def test_stt_get_in_progress_reads_intermediate(client, tmp_path):
    auth = _signup(client)
    h = auth["headers"]
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    # Pre-seed output_dir on disk
    _seed_stt_output(tmp_path, ["openai"], total=1)

    job_uuid = db.create_job(
        job_type="stt-eval",
        org_uuid=org_uuid,
        user_id=user_id,
        status="in_progress",
        details={
            "providers": ["openai"],
            "language": "en",
            "audio_paths": ["s3://bucket/k.wav"],
            "texts": ["hi"],
            "s3_bucket": "bucket",
            "output_dir": str(tmp_path),
            "evaluators": [],
        },
    )
    resp = client.get(f"/stt/evaluate/{job_uuid}", headers=h)
    assert resp.status_code == 200
    body = resp.json()
    # Should have provider_results
    assert body["provider_results"]


def test_stt_get_timeout_path(client, tmp_path):
    """An old updated_at triggers the timeout branch which kills the process,
    reads intermediate, merges, marks failed."""
    auth = _signup(client)
    h = auth["headers"]
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    _seed_stt_output(tmp_path, ["openai"], total=1)

    # Manually create a job and force updated_at to be old
    job_uuid = db.create_job(
        job_type="stt-eval",
        org_uuid=org_uuid,
        user_id=user_id,
        status="in_progress",
        details={
            "providers": ["openai"],
            "language": "en",
            "audio_paths": ["s3://bucket/k.wav"],
            "texts": ["hi"],
            "s3_bucket": "bucket",
            "output_dir": str(tmp_path),
            "pid": 99999,
            "evaluators": [],
        },
    )

    with patch("routers.stt.is_job_timed_out", return_value=True), patch(
        "routers.stt.kill_process_group"
    ), patch("routers.stt.try_start_queued_job"):
        resp = client.get(f"/stt/evaluate/{job_uuid}", headers=h)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"


def _seed_tts_output(tmp_path: Path, providers, total: int = 1):
    for p in providers:
        sub = tmp_path / f"{p}_results"
        sub.mkdir(parents=True)
        with open(sub / "results.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "text", "audio_path"])
            for i in range(total):
                writer.writerow([f"row_{i+1}", "hi", str(sub / f"audio_{i+1}.wav")])
                (sub / f"audio_{i+1}.wav").write_bytes(b"x")
        with open(sub / "metrics.json", "w") as f:
            json.dump({"ttfb": 0.5}, f)


def test_tts_get_in_progress_reads_intermediate(client, tmp_path):
    auth = _signup(client)
    h = auth["headers"]
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    _seed_tts_output(tmp_path, ["openai"], total=1)
    job_uuid = db.create_job(
        job_type="tts-eval",
        org_uuid=org_uuid,
        user_id=user_id,
        status="in_progress",
        details={
            "providers": ["openai"],
            "language": "en",
            "texts": ["hi"],
            "s3_bucket": "bucket",
            "output_dir": str(tmp_path),
            "evaluators": [],
        },
    )
    with patch("routers.tts.upload_file_to_s3"), patch(
        "routers.tts.get_s3_client"
    ), patch(
        "routers.tts.generate_presigned_download_url",
        return_value="https://signed",
    ):
        resp = client.get(f"/tts/evaluate/{job_uuid}", headers=h)
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider_results"]
    # Raw S3 key is exposed alongside the presigned audio_path so it can be
    # copied into a TTS annotation item.
    row = body["provider_results"][0]["results"][0]
    assert row["audio_path"] == "https://signed"
    assert row["audio_s3_path"] == f"tts/evals/{job_uuid}/outputs/openai/audio_1.wav"


def test_tts_get_done_exposes_raw_audio_key(client):
    auth = _signup(client)
    h = auth["headers"]
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    raw_key = "tts/evals/done-job/outputs/openai/audio_1.wav"
    job_uuid = db.create_job(
        job_type="tts-eval",
        org_uuid=org_uuid,
        user_id=user_id,
        status="done",
        details={
            "providers": ["openai"],
            "language": "en",
            "texts": ["hi"],
            "s3_bucket": "bucket",
            "evaluators": [],
        },
        results={
            "provider_results": [
                {
                    "provider": "openai",
                    "success": True,
                    "results": [{"id": "row_1", "text": "hi", "audio_path": raw_key}],
                }
            ],
        },
    )
    with patch(
        "utils.generate_presigned_download_url",
        return_value="https://signed",
    ):
        resp = client.get(f"/tts/evaluate/{job_uuid}", headers=h)
    assert resp.status_code == 200
    row = resp.json()["provider_results"][0]["results"][0]
    assert row["audio_path"] == "https://signed"
    assert row["audio_s3_path"] == raw_key


def test_tts_get_timeout_path(client, tmp_path):
    auth = _signup(client)
    h = auth["headers"]
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    _seed_tts_output(tmp_path, ["openai"], total=1)
    job_uuid = db.create_job(
        job_type="tts-eval",
        org_uuid=org_uuid,
        user_id=user_id,
        status="in_progress",
        details={
            "providers": ["openai"],
            "language": "en",
            "texts": ["hi"],
            "s3_bucket": "bucket",
            "output_dir": str(tmp_path),
            "pid": 99999,
            "evaluators": [],
        },
    )
    with patch("routers.tts.is_job_timed_out", return_value=True), patch(
        "routers.tts.kill_process_group"
    ), patch("routers.tts.try_start_queued_job"), patch(
        "routers.tts.upload_file_to_s3"
    ), patch("routers.tts.get_s3_client"):
        resp = client.get(f"/tts/evaluate/{job_uuid}", headers=h)
    assert resp.status_code == 200
    assert resp.json()["status"] == "failed"


def test_simulation_run_status_voice_done(client):
    """Voice simulation in DONE status with simulation_results triggers presigned-URL on-the-fly path."""
    auth = _signup(client)
    h = auth["headers"]
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    sim_uuid = db.create_simulation(
        name=f"s-{uuid.uuid4().hex[:6]}", org_uuid=org_uuid, user_id=user_id
    )
    job_uuid = db.create_simulation_job(
        simulation_id=sim_uuid, job_type="voice", status="done"
    )
    db.update_simulation_job(
        job_uuid,
        results={
            "simulation_results": [
                {
                    "simulation_name": "sim_1",
                    "audios_s3_path": "prefix/audios",
                    "conversation_wav_s3_key": "prefix/conversation.wav",
                    "transcript": [],
                }
            ],
            "total_simulations": 1,
        },
    )
    with patch(
        "routers.simulations._get_audio_urls_from_s3_key",
        return_value=["https://signed/x"],
    ), patch(
        "routers.simulations.generate_presigned_download_url",
        return_value="https://signed/y",
    ), patch(
        "routers.simulations.get_s3_output_config", return_value="bucket"
    ):
        resp = client.get(f"/simulations/run/{job_uuid}", headers=h)
    assert resp.status_code == 200
    body = resp.json()
    assert body["simulation_results"][0]["audio_urls"] == ["https://signed/x"]


def test_simulation_run_status_in_progress_timeout(client):
    auth = _signup(client)
    h = auth["headers"]
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    sim_uuid = db.create_simulation(
        name=f"s-{uuid.uuid4().hex[:6]}", org_uuid=org_uuid, user_id=user_id
    )
    job_uuid = db.create_simulation_job(
        simulation_id=sim_uuid, job_type="text", status="in_progress"
    )
    db.update_simulation_job(job_uuid, details={"pid": 9999})
    with patch(
        "routers.simulations.is_job_timed_out", return_value=True
    ), patch(
        "routers.simulations.kill_process_group"
    ), patch(
        "routers.simulations.try_start_queued_simulation_job"
    ):
        resp = client.get(f"/simulations/run/{job_uuid}", headers=h)
    assert resp.status_code == 200
    assert resp.json()["status"] == "failed"
