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


def _failed_eval_job(db_mod, auth, *, job_type, details):
    org_uuid = db_mod.get_personal_org_for_user(auth["user_uuid"])["uuid"]
    task_id = db_mod.create_job(
        job_type=job_type,
        org_uuid=org_uuid,
        user_id=auth["user_uuid"],
        status="failed",
        details=details,
        results={"error": "failed"},
    )
    return task_id, org_uuid


# ---------------------------------------------------------------------------
# STT /evaluate/import
# ---------------------------------------------------------------------------


def _run_archive(providers, *, root="output"):
    """Tar a calibrate STT output folder. `providers` maps a name to its rows."""
    import io
    import json
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:

        def add(name, payload):
            data = payload.encode("utf-8")
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        for provider, rows in providers.items():
            header = "id,gt,pred,wer,sarvam_intent_score\n"
            body = "".join(
                f"{r['id']},{r['gt']},{r['pred']},0.5,1\n" for r in rows
            )
            add(f"{provider}/results.csv", header + body)
            add(f"{provider}/metrics.json", json.dumps({"wer": 0.5}))
            # Never read, and the reason the endpoint filters members by name.
            add(f"{provider}/logs", "x" * 2048)
    return buf.getvalue()


def _stt_dataset(db_mod, auth, texts, name="import-ds"):
    org_uuid = db_mod.get_personal_org_for_user(auth["user_uuid"])["uuid"]
    ds_uuid = db_mod.create_dataset(
        name=name,
        dataset_type="stt",
        org_uuid=org_uuid,
        user_id=auth["user_uuid"],
    )
    db_mod.add_dataset_items(
        ds_uuid,
        [
            {"text": text, "audio_path": f"s3://bucket/stt/media/{i}.wav"}
            for i, text in enumerate(texts)
        ],
    )
    return ds_uuid


def _post_import(client, auth, ds_uuid, archive, language="english"):
    return client.post(
        "/stt/evaluate/import",
        headers=auth["headers"],
        data={"dataset_id": ds_uuid, "language": language},
        files={"archive": ("output.tar", archive, "application/x-tar")},
    )


def test_stt_import_stores_a_finished_run(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    ds_uuid = _stt_dataset(db_mod, auth, ["hello there", "second line"])
    archive = _run_archive(
        {
            "deepgram": [
                {"id": "drive-a", "gt": "hello there", "pred": "hello their"},
                {"id": "drive-b", "gt": "second line", "pred": "second lime"},
            ],
            "sarvam": [
                {"id": "drive-a", "gt": "hello there", "pred": "hello there"},
                {"id": "drive-b", "gt": "second line", "pred": "second line"},
            ],
        }
    )

    resp = _post_import(client, auth, ds_uuid, archive)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "done"
    assert body["providers"] == ["deepgram", "sarvam"]
    assert body["row_count"] == 2

    job = db_mod.get_job(body["task_id"])
    assert job["status"] == "done"
    assert job["details"]["providers"] == ["deepgram", "sarvam"]
    assert job["details"]["dataset_id"] == ds_uuid
    assert job["details"]["evaluators"] == []
    assert job["results"]["error"] is None

    rows = job["results"]["provider_results"][0]["results"]
    # Renumbered so the run page can pair each row with the dataset's audio,
    # with the caller's own id kept alongside.
    assert [r["id"] for r in rows] == ["audio_1", "audio_2"]
    assert [r["source_id"] for r in rows] == ["drive-a", "drive-b"]


def test_stt_import_run_reads_back_with_audio_and_numbers(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    ds_uuid = _stt_dataset(db_mod, auth, ["only line"], name="import-read-ds")
    archive = _run_archive(
        {"deepgram": [{"id": "drive-a", "gt": "only line", "pred": "only lion"}]}
    )
    task_id = _post_import(client, auth, ds_uuid, archive).json()["task_id"]

    with patch("routers.stt.presign_audio_path", return_value="https://signed/a.wav"):
        resp = client.get(f"/stt/evaluate/{task_id}", headers=auth["headers"])

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["dataset_name"] == "import-read-ds"
    row = body["provider_results"][0]["results"][0]
    assert row["audio_url"] == "https://signed/a.wav"
    # Both are text in the CSV and have to reach the page as numbers.
    assert row["wer"] == 0.5
    assert row["sarvam_intent_score"] == 1.0


def test_stt_import_rejects_row_count_mismatch(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    ds_uuid = _stt_dataset(db_mod, auth, ["a", "b"], name="import-count-ds")
    archive = _run_archive({"deepgram": [{"id": "x", "gt": "a", "pred": "a"}]})

    resp = _post_import(client, auth, ds_uuid, archive)
    assert resp.status_code == 400
    assert "1 rows but the dataset has 2 items" in resp.json()["detail"]


def test_stt_import_rejects_rows_out_of_dataset_order(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    ds_uuid = _stt_dataset(db_mod, auth, ["first", "second"], name="import-order-ds")
    archive = _run_archive(
        {
            "deepgram": [
                {"id": "x", "gt": "second", "pred": "second"},
                {"id": "y", "gt": "first", "pred": "first"},
            ]
        }
    )

    resp = _post_import(client, auth, ds_uuid, archive)
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "do not match the ground truth" in detail
    assert "same order" in detail


def test_stt_import_rejects_archive_without_results(client, monkeypatch):
    import io
    import tarfile

    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    ds_uuid = _stt_dataset(db_mod, auth, ["a"], name="import-empty-ds")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        data = b"nothing useful"
        info = tarfile.TarInfo("output/deepgram/logs")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    resp = _post_import(client, auth, ds_uuid, buf.getvalue())
    assert resp.status_code == 400
    assert "results.csv" in resp.json()["detail"]


def test_stt_import_rejects_an_unreadable_archive(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    ds_uuid = _stt_dataset(db_mod, auth, ["a"], name="import-bad-tar-ds")

    resp = _post_import(client, auth, ds_uuid, b"this is not a tar")
    assert resp.status_code == 400
    assert "Could not read the archive" in resp.json()["detail"]


def test_stt_import_rejects_an_oversized_archive(client, monkeypatch):
    import db as db_mod
    import routers.stt as stt_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    monkeypatch.setattr(stt_mod, "_MAX_IMPORT_ARCHIVE_BYTES", 16)
    ds_uuid = _stt_dataset(db_mod, auth, ["a"], name="import-big-ds")
    archive = _run_archive({"deepgram": [{"id": "x", "gt": "a", "pred": "a"}]})

    resp = _post_import(client, auth, ds_uuid, archive)
    assert resp.status_code == 413
    assert "larger than" in resp.json()["detail"]


def test_stt_import_unknown_dataset(client, monkeypatch):
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    auth = _signup(client)
    archive = _run_archive({"deepgram": [{"id": "x", "gt": "a", "pred": "a"}]})

    resp = _post_import(client, auth, "00000000-0000-4000-8000-000000000001", archive)
    assert resp.status_code == 404


def test_stt_import_rejects_another_workspaces_dataset(client, monkeypatch):
    import db as db_mod

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    owner = _signup(client)
    ds_uuid = _stt_dataset(db_mod, owner, ["a"], name="import-other-ds")
    archive = _run_archive({"deepgram": [{"id": "x", "gt": "a", "pred": "a"}]})

    outsider = _signup(client)
    resp = _post_import(client, outsider, ds_uuid, archive)
    assert resp.status_code == 404


def test_stt_import_rejects_two_run_folders(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    ds_uuid = _stt_dataset(db_mod, auth, ["a"], name="import-two-runs-ds")
    rows = {"deepgram": [{"id": "x", "gt": "a", "pred": "a"}]}
    first = _run_archive(rows, root="run-one")
    second = _run_archive(rows, root="run-two")

    import io
    import tarfile

    merged = io.BytesIO()
    with tarfile.open(fileobj=merged, mode="w") as out:
        for blob in (first, second):
            with tarfile.open(fileobj=io.BytesIO(blob)) as src:
                for member in src.getmembers():
                    out.addfile(member, src.extractfile(member))

    resp = _post_import(client, auth, ds_uuid, merged.getvalue())
    assert resp.status_code == 400
    assert "one run at a time" in resp.json()["detail"]


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


@pytest.mark.parametrize(
    "payload_extra, expected",
    [({}, True), ({"sarvam_judges": True}, True), ({"sarvam_judges": False}, False)],
)
def test_stt_evaluate_snapshots_sarvam_judges(
    client, monkeypatch, payload_extra, expected
):
    """The sarvam_judges toggle (default True) is stored in job details so the
    runner and retry path remember the run's mode."""
    import db as db_mod

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
                **payload_extra,
            },
            headers=auth["headers"],
        )
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

    job = db_mod.get_job(task_id)
    assert job["details"]["sarvam_judges"] is expected


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


def test_stt_evaluate_no_evaluators_snapshots_empty(client, monkeypatch):
    """Omitting evaluator_uuids runs transcription metrics only — no default judge."""
    import db

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
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]
    job = db.get_job(task_id)
    assert job["details"]["evaluators"] == []


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
    evaluators = client.get("/evaluators", headers=auth["headers"]).json()["items"]
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


def test_stt_retry_reruns_same_job(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.stt.can_start_job", return_value=False), patch(
        "threading.Thread"
    ):
        original = client.post(
            "/stt/evaluate",
            json={
                "providers": ["openai"],
                "language": "en",
                "audio_paths": ["s3://b/k.wav"],
                "texts": ["hello"],
            },
            headers=auth["headers"],
        )
        assert original.status_code == 200
        original_id = original.json()["task_id"]

        retry = client.post(
            f"/stt/evaluate/{original_id}/retry",
            headers=auth["headers"],
        )
        assert retry.status_code == 200
        body = retry.json()
        assert body["task_id"] == original_id
        assert body["status"] == "queued"


def test_stt_retry_rejects_in_progress(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.stt.can_start_job", return_value=True), patch(
        "routers.stt.threading.Thread"
    ):
        original = client.post(
            "/stt/evaluate",
            json={
                "providers": ["openai"],
                "language": "en",
                "audio_paths": ["s3://b/k.wav"],
                "texts": ["hello"],
            },
            headers=auth["headers"],
        )
        original_id = original.json()["task_id"]
        assert original.json()["status"] == "in_progress"

    resp = client.post(
        f"/stt/evaluate/{original_id}/retry",
        headers=auth["headers"],
    )
    assert resp.status_code == 400


def test_stt_retry_not_found(client):
    auth = _signup(client)
    resp = client.post(
        "/stt/evaluate/missing/retry",
        headers=auth["headers"],
    )
    assert resp.status_code == 404


def test_stt_retry_wrong_job_type(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.tts.can_start_job", return_value=False), patch(
        "threading.Thread"
    ):
        tts = client.post(
            "/tts/evaluate",
            json={
                "providers": ["openai"],
                "language": "en",
                "texts": ["hello"],
            },
            headers=auth["headers"],
        )
        tts_id = tts.json()["task_id"]

    resp = client.post(
        f"/stt/evaluate/{tts_id}/retry",
        headers=auth["headers"],
    )
    assert resp.status_code == 404


def test_stt_retry_rereads_linked_dataset(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    org_uuid = db_mod.get_personal_org_for_user(auth["user_uuid"])["uuid"]
    ds_uuid = db_mod.create_dataset(
        name="retry-ds",
        dataset_type="stt",
        org_uuid=org_uuid,
        user_id=auth["user_uuid"],
    )
    item_ids = db_mod.add_dataset_items(
        ds_uuid,
        [{"text": "hi", "audio_path": "s3://b/broken.wav"}],
    )
    task_id = db_mod.create_job(
        job_type="stt-eval",
        org_uuid=org_uuid,
        user_id=auth["user_uuid"],
        status="failed",
        details={
            "audio_paths": ["s3://b/broken.wav"],
            "texts": ["hi"],
            "providers": ["openai"],
            "language": "en",
            "dataset_id": ds_uuid,
            "dataset_name": "retry-ds",
            "dataset_item_ids": item_ids,
            "evaluators": [],
        },
        results={"error": "failed"},
    )
    db_mod.update_dataset_item(item_ids[0], ds_uuid, audio_path="s3://b/fixed.wav")

    with patch("routers.stt.can_start_job", return_value=False), patch(
        "threading.Thread"
    ):
        resp = client.post(
            f"/stt/evaluate/{task_id}/retry",
            headers=auth["headers"],
        )
    assert resp.status_code == 200
    job = db_mod.get_job(task_id, org_uuid=org_uuid)
    assert job["details"]["audio_paths"] == ["s3://b/fixed.wav"]


def test_stt_retry_starts_immediately(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    task_id, _org = _failed_eval_job(
        db_mod,
        auth,
        job_type="stt-eval",
        details={
            "audio_paths": ["s3://b/k.wav"],
            "texts": ["hi"],
            "providers": ["openai"],
            "language": "en",
            "evaluators": [],
        },
    )
    with patch("routers.stt.can_start_job", return_value=True), patch(
        "routers.stt.threading.Thread"
    ) as thread_mock:
        resp = client.post(
            f"/stt/evaluate/{task_id}/retry",
            headers=auth["headers"],
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "in_progress"
    thread_mock.assert_called_once()


def test_stt_retry_missing_providers(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    task_id, _org = _failed_eval_job(
        db_mod,
        auth,
        job_type="stt-eval",
        details={
            "audio_paths": ["s3://b/k.wav"],
            "texts": ["hi"],
            "providers": [],
            "language": "en",
        },
    )
    resp = client.post(
        f"/stt/evaluate/{task_id}/retry",
        headers=auth["headers"],
    )
    assert resp.status_code == 400


def test_stt_retry_missing_bucket(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    task_id, _org = _failed_eval_job(
        db_mod,
        auth,
        job_type="stt-eval",
        details={
            "audio_paths": ["s3://b/k.wav"],
            "texts": ["hi"],
            "providers": ["openai"],
            "language": "en",
        },
    )
    monkeypatch.delenv("S3_OUTPUT_BUCKET", raising=False)
    monkeypatch.delenv("OBJECT_STORAGE_MODE", raising=False)
    resp = client.post(
        f"/stt/evaluate/{task_id}/retry",
        headers=auth["headers"],
    )
    assert resp.status_code == 500


def _dataset_backed_stt_job(db_mod, auth):
    org_uuid = db_mod.get_personal_org_for_user(auth["user_uuid"])["uuid"]
    ds_uuid = db_mod.create_dataset(
        name="gone-ds",
        dataset_type="stt",
        org_uuid=org_uuid,
        user_id=auth["user_uuid"],
    )
    item_ids = db_mod.add_dataset_items(
        ds_uuid, [{"text": "hi", "audio_path": "s3://b/k.wav"}]
    )
    task_id = db_mod.create_job(
        job_type="stt-eval",
        org_uuid=org_uuid,
        user_id=auth["user_uuid"],
        status="failed",
        details={
            "audio_paths": ["s3://b/k.wav"],
            "texts": ["hi"],
            "providers": ["openai"],
            "language": "en",
            "dataset_id": ds_uuid,
            "dataset_name": "gone-ds",
            "dataset_item_ids": item_ids,
            "evaluators": [],
        },
        results={"error": "failed"},
    )
    return task_id, org_uuid, ds_uuid


def test_stt_status_nulls_dataset_when_deleted(client):
    import db as db_mod

    auth = _signup(client)
    task_id, org_uuid, ds_uuid = _dataset_backed_stt_job(db_mod, auth)

    before = client.get(f"/stt/evaluate/{task_id}", headers=auth["headers"]).json()
    assert before["dataset_id"] == ds_uuid
    assert before["dataset_name"] == "gone-ds"

    db_mod.delete_dataset(ds_uuid, org_uuid)

    after = client.get(f"/stt/evaluate/{task_id}", headers=auth["headers"]).json()
    assert after["dataset_id"] is None
    assert after["dataset_name"] is None


def test_stt_retry_deleted_dataset_returns_clear_error(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    task_id, org_uuid, ds_uuid = _dataset_backed_stt_job(db_mod, auth)
    db_mod.delete_dataset(ds_uuid, org_uuid)

    resp = client.post(
        f"/stt/evaluate/{task_id}/retry",
        headers=auth["headers"],
    )
    assert resp.status_code == 400
    assert "no longer exists" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# _refresh_evaluators_to_live (re-hydrate snapshot to live version at run time)
# ---------------------------------------------------------------------------


def _make_snapshot(ev_uuid: str, version_uuid: str, *, evaluator_type: str,
                   data_type: str, system_prompt: str) -> dict:
    """Mimic `_resolve_evaluators_for_job` submit-time snapshot output."""
    return {
        "uuid": ev_uuid,
        "name": "ev",
        "evaluator_type": evaluator_type,
        "data_type": data_type,
        "kind": "single",
        "output_type": "binary",
        "evaluator_version_id": version_uuid,
        "judge_model": "m",
        "system_prompt": system_prompt,
        "output_config": None,
        "variables": None,
        "variable_values": {"foo": "bar"},
    }


def test_stt_refresh_evaluators_picks_up_live_version(client):
    import db
    from llm_judge import refresh_evaluators_to_live as stt_refresh

    ev_uuid = db.create_evaluator(
        name=f"stt-ev-{uuid.uuid4().hex[:8]}",
        evaluator_type="stt",
        data_type="text",
        output_type="binary",
    )
    v1 = db.create_evaluator_version(
        evaluator_uuid=ev_uuid, judge_model="m", system_prompt="STT V1"
    )
    db.set_evaluator_live_version(ev_uuid, v1["uuid"])

    snapshot = _make_snapshot(
        ev_uuid, v1["uuid"], evaluator_type="stt", data_type="text",
        system_prompt="STT V1",
    )

    # Create v2 and make it live; no re-link, no new snapshot.
    v2 = db.create_evaluator_version(
        evaluator_uuid=ev_uuid, judge_model="m", system_prompt="STT V2"
    )
    db.set_evaluator_live_version(ev_uuid, v2["uuid"])

    refreshed = stt_refresh([snapshot])
    assert len(refreshed) == 1
    entry = refreshed[0]
    assert entry["system_prompt"] == "STT V2"
    assert entry["evaluator_version_id"] == v2["uuid"]
    # Pinned per-job config preserved.
    assert entry["variable_values"] == {"foo": "bar"}


def test_stt_refresh_keeps_snapshot_when_no_live_version(client):
    from llm_judge import refresh_evaluators_to_live as stt_refresh

    snap = _make_snapshot(
        str(uuid.uuid4()), str(uuid.uuid4()), evaluator_type="stt",
        data_type="text", system_prompt="STT V1",
    )
    refreshed = stt_refresh([snap])
    assert len(refreshed) == 1
    # Unknown evaluator → snapshot returned unchanged.
    assert refreshed[0] == snap


def test_tts_refresh_evaluators_picks_up_live_version(client):
    import db
    from llm_judge import refresh_evaluators_to_live as tts_refresh

    ev_uuid = db.create_evaluator(
        name=f"tts-ev-{uuid.uuid4().hex[:8]}",
        evaluator_type="tts",
        data_type="audio",
        output_type="binary",
    )
    v1 = db.create_evaluator_version(
        evaluator_uuid=ev_uuid, judge_model="m", system_prompt="TTS V1"
    )
    db.set_evaluator_live_version(ev_uuid, v1["uuid"])

    snapshot = _make_snapshot(
        ev_uuid, v1["uuid"], evaluator_type="tts", data_type="audio",
        system_prompt="TTS V1",
    )

    v2 = db.create_evaluator_version(
        evaluator_uuid=ev_uuid, judge_model="m", system_prompt="TTS V2"
    )
    db.set_evaluator_live_version(ev_uuid, v2["uuid"])

    refreshed = tts_refresh([snapshot])
    assert len(refreshed) == 1
    entry = refreshed[0]
    assert entry["system_prompt"] == "TTS V2"
    assert entry["evaluator_version_id"] == v2["uuid"]
    assert entry["variable_values"] == {"foo": "bar"}


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


def test_tts_retry_reruns_same_job(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.tts.can_start_job", return_value=False), patch(
        "threading.Thread"
    ):
        original = client.post(
            "/tts/evaluate",
            json={
                "providers": ["openai"],
                "language": "en",
                "texts": ["hello"],
            },
            headers=auth["headers"],
        )
        assert original.status_code == 200
        original_id = original.json()["task_id"]

        retry = client.post(
            f"/tts/evaluate/{original_id}/retry",
            headers=auth["headers"],
        )
        assert retry.status_code == 200
        body = retry.json()
        assert body["task_id"] == original_id
        assert body["status"] == "queued"


def test_tts_retry_rejects_in_progress(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.tts.can_start_job", return_value=True), patch(
        "routers.tts.threading.Thread"
    ):
        original = client.post(
            "/tts/evaluate",
            json={
                "providers": ["openai"],
                "language": "en",
                "texts": ["hello"],
            },
            headers=auth["headers"],
        )
        original_id = original.json()["task_id"]
        assert original.json()["status"] == "in_progress"

    resp = client.post(
        f"/tts/evaluate/{original_id}/retry",
        headers=auth["headers"],
    )
    assert resp.status_code == 400


def test_tts_retry_not_found(client):
    auth = _signup(client)
    resp = client.post(
        "/tts/evaluate/missing/retry",
        headers=auth["headers"],
    )
    assert resp.status_code == 404


def test_tts_retry_wrong_job_type(client, monkeypatch):
    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.stt.can_start_job", return_value=False), patch(
        "threading.Thread"
    ):
        stt = client.post(
            "/stt/evaluate",
            json={
                "providers": ["openai"],
                "language": "en",
                "audio_paths": ["s3://b/k.wav"],
                "texts": ["hi"],
            },
            headers=auth["headers"],
        )
        stt_id = stt.json()["task_id"]

    resp = client.post(
        f"/tts/evaluate/{stt_id}/retry",
        headers=auth["headers"],
    )
    assert resp.status_code == 404


def test_tts_retry_starts_immediately(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    task_id, _org = _failed_eval_job(
        db_mod,
        auth,
        job_type="tts-eval",
        details={
            "texts": ["hello"],
            "providers": ["openai"],
            "language": "en",
            "evaluators": [],
        },
    )
    with patch("routers.tts.can_start_job", return_value=True), patch(
        "routers.tts.threading.Thread"
    ) as thread_mock:
        resp = client.post(
            f"/tts/evaluate/{task_id}/retry",
            headers=auth["headers"],
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "in_progress"
    thread_mock.assert_called_once()


def test_tts_retry_missing_providers(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    task_id, _org = _failed_eval_job(
        db_mod,
        auth,
        job_type="tts-eval",
        details={"texts": ["hello"], "providers": [], "language": "en"},
    )
    resp = client.post(
        f"/tts/evaluate/{task_id}/retry",
        headers=auth["headers"],
    )
    assert resp.status_code == 400


def test_tts_retry_missing_bucket(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    task_id, _org = _failed_eval_job(
        db_mod,
        auth,
        job_type="tts-eval",
        details={
            "texts": ["hello"],
            "providers": ["openai"],
            "language": "en",
        },
    )
    monkeypatch.delenv("S3_OUTPUT_BUCKET", raising=False)
    monkeypatch.delenv("OBJECT_STORAGE_MODE", raising=False)
    resp = client.post(
        f"/tts/evaluate/{task_id}/retry",
        headers=auth["headers"],
    )
    assert resp.status_code == 500


def test_tts_retry_rereads_linked_dataset(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    org_uuid = db_mod.get_personal_org_for_user(auth["user_uuid"])["uuid"]
    ds_uuid = db_mod.create_dataset(
        name="retry-tts-ds",
        dataset_type="tts",
        org_uuid=org_uuid,
        user_id=auth["user_uuid"],
    )
    item_ids = db_mod.add_dataset_items(ds_uuid, [{"text": "old line"}])
    task_id = db_mod.create_job(
        job_type="tts-eval",
        org_uuid=org_uuid,
        user_id=auth["user_uuid"],
        status="failed",
        details={
            "texts": ["old line"],
            "providers": ["openai"],
            "language": "en",
            "dataset_id": ds_uuid,
            "dataset_name": "retry-tts-ds",
            "dataset_item_ids": item_ids,
            "evaluators": [],
        },
        results={"error": "failed"},
    )
    db_mod.update_dataset_item(item_ids[0], ds_uuid, text="fixed line")

    with patch("routers.tts.can_start_job", return_value=False), patch(
        "threading.Thread"
    ):
        resp = client.post(
            f"/tts/evaluate/{task_id}/retry",
            headers=auth["headers"],
        )
    assert resp.status_code == 200
    job = db_mod.get_job(task_id, org_uuid=org_uuid)
    assert job["details"]["texts"] == ["fixed line"]


def _dataset_backed_tts_job(db_mod, auth):
    org_uuid = db_mod.get_personal_org_for_user(auth["user_uuid"])["uuid"]
    ds_uuid = db_mod.create_dataset(
        name="gone-tts-ds",
        dataset_type="tts",
        org_uuid=org_uuid,
        user_id=auth["user_uuid"],
    )
    item_ids = db_mod.add_dataset_items(ds_uuid, [{"text": "line"}])
    task_id = db_mod.create_job(
        job_type="tts-eval",
        org_uuid=org_uuid,
        user_id=auth["user_uuid"],
        status="failed",
        details={
            "texts": ["line"],
            "providers": ["openai"],
            "language": "en",
            "dataset_id": ds_uuid,
            "dataset_name": "gone-tts-ds",
            "dataset_item_ids": item_ids,
            "evaluators": [],
        },
        results={"error": "failed"},
    )
    return task_id, org_uuid, ds_uuid


def test_tts_status_nulls_dataset_when_deleted(client):
    import db as db_mod

    auth = _signup(client)
    task_id, org_uuid, ds_uuid = _dataset_backed_tts_job(db_mod, auth)

    before = client.get(f"/tts/evaluate/{task_id}", headers=auth["headers"]).json()
    assert before["dataset_id"] == ds_uuid
    assert before["dataset_name"] == "gone-tts-ds"

    db_mod.delete_dataset(ds_uuid, org_uuid)

    after = client.get(f"/tts/evaluate/{task_id}", headers=auth["headers"]).json()
    assert after["dataset_id"] is None
    assert after["dataset_name"] is None


def test_tts_retry_deleted_dataset_returns_clear_error(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    task_id, org_uuid, ds_uuid = _dataset_backed_tts_job(db_mod, auth)
    db_mod.delete_dataset(ds_uuid, org_uuid)

    resp = client.post(
        f"/tts/evaluate/{task_id}/retry",
        headers=auth["headers"],
    )
    assert resp.status_code == 400
    assert "no longer exists" in resp.json()["detail"]


def _lock_held_by_another_thread() -> bool:
    """True when the job-queue lock is currently held.

    Probed from a separate thread because the lock is reentrant: the calling
    thread would re-acquire its own lock and learn nothing.
    """
    import threading

    import utils

    result = []

    def probe():
        got = utils._job_queue_lock.acquire(timeout=0.05)
        result.append(got)
        if got:
            utils._job_queue_lock.release()

    t = threading.Thread(target=probe)
    t.start()
    t.join()
    return not result[0]


def test_stt_evaluate_holds_queue_lock_across_check_and_create(client, monkeypatch):
    """Capacity check and job INSERT must happen under one lock hold.

    Without it, two submissions can both see a free slot and both start,
    exceeding MAX_CONCURRENT_JOBS.
    """
    import db as db_mod
    import routers.stt as stt_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")

    held_during = {}
    real_create_job = db_mod.create_job

    def capacity_check(*args, **kwargs):
        held_during["check"] = _lock_held_by_another_thread()
        return False  # queue the job so no worker thread spawns

    def create_job(*args, **kwargs):
        held_during["create"] = _lock_held_by_another_thread()
        return real_create_job(*args, **kwargs)

    monkeypatch.setattr(stt_mod, "can_start_job", capacity_check)
    monkeypatch.setattr(stt_mod, "create_job", create_job)

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
    assert held_during == {"check": True, "create": True}


def test_stt_import_rejects_a_provider_folder_with_no_rows(client, monkeypatch):
    import io
    import json
    import tarfile

    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    ds_uuid = _stt_dataset(db_mod, auth, ["a"], name="import-norows-ds")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, payload in (
            ("output/deepgram/results.csv", "id,gt,pred\n"),
            ("output/deepgram/metrics.json", json.dumps({"wer": 0.1})),
        ):
            data = payload.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    resp = _post_import(client, auth, ds_uuid, buf.getvalue())
    assert resp.status_code == 400
    assert "No rows could be read for: deepgram" in resp.json()["detail"]


def test_stt_import_needs_the_artifact_bucket(client, monkeypatch):
    import db as db_mod

    auth = _signup(client)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    ds_uuid = _stt_dataset(db_mod, auth, ["a"], name="import-nobucket-ds")
    archive = _run_archive({"deepgram": [{"id": "x", "gt": "a", "pred": "a"}]})
    monkeypatch.delenv("S3_OUTPUT_BUCKET", raising=False)

    resp = _post_import(client, auth, ds_uuid, archive)
    assert resp.status_code == 500
