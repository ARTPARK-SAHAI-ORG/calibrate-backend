"""API-key access to STT datasets: reading one and uploading an audio item."""

from __future__ import annotations

import io
import uuid
import wave
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from routers import datasets as datasets_router


@pytest.fixture(scope="module")
def client():
    import main as main_mod

    with patch("main.recover_pending_jobs"):
        with TestClient(main_mod.app) as c:
            yield c


@pytest.fixture
def uploads(monkeypatch):
    stored = []
    monkeypatch.setattr(datasets_router, "get_s3_client", lambda: None)
    monkeypatch.setattr(
        datasets_router,
        "upload_file_to_s3",
        lambda _s3, path, bucket, key: stored.append((open(path, "rb").read(), bucket, key)),
    )
    monkeypatch.setattr(datasets_router, "presign_audio_path", lambda p: p)
    return stored


def _signup(client):
    suffix = uuid.uuid4().hex[:8]
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "Ds",
            "last_name": "U",
            "email": f"ds-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _api_key(client, h):
    created = client.post("/api-keys", json={"name": "formbharo"}, headers=h)
    assert created.status_code == 201, created.text
    return {"X-API-Key": created.json()["key"]}


def _dataset(client, h, dataset_type="stt"):
    resp = client.post(
        "/datasets",
        json={"name": f"ds-{uuid.uuid4().hex[:8]}", "dataset_type": dataset_type},
        headers=h,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["uuid"]


def _wav(seconds: float, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


def _upload(client, headers, dataset_id, data, text="hello"):
    return client.post(
        f"/datasets/{dataset_id}/items/upload",
        files={"audio": ("clip.wav", data, "audio/wav")},
        data={"text": text},
        headers=headers,
    )


def test_api_key_uploads_item_and_reads_dataset(client, uploads):
    h = _signup(client)
    key = _api_key(client, h)
    ds = _dataset(client, h)
    client.post(
        f"/datasets/{ds}/items",
        json=[{"audio_path": "s3://test-bucket/stt/media/first.wav", "text": "first"}],
        headers=h,
    )
    clip = _wav(2, rate=16000)

    resp = _upload(client, key, ds, clip, text="corrected text")

    assert resp.status_code == 201, resp.text
    item = resp.json()
    assert item["text"] == "corrected text"
    assert item["order_index"] == 1
    [(data, bucket, s3_key)] = uploads
    assert data == clip
    assert bucket == "test-bucket"
    assert s3_key.startswith("stt/media/") and s3_key.endswith(".wav")
    assert item["audio_path"] == f"s3://test-bucket/{s3_key}"

    detail = client.get(f"/datasets/{ds}", headers=key)
    assert detail.status_code == 200, detail.text
    assert [i["text"] for i in detail.json()["items"]] == ["first", "corrected text"]


def test_api_key_creates_dataset_in_its_workspace(client, uploads):
    h = _signup(client)
    key = _api_key(client, h)
    name = f"agent-{uuid.uuid4().hex[:8]}"

    resp = client.post("/datasets", json={"name": name, "dataset_type": "stt"}, headers=key)

    assert resp.status_code == 201, resp.text
    ds = resp.json()
    assert (ds["name"], ds["dataset_type"], ds["item_count"]) == (name, "stt", 0)
    assert client.get(f"/datasets/{ds['uuid']}", headers=h).status_code == 200
    assert _upload(client, key, ds["uuid"], _wav(1)).status_code == 201
    again = client.post("/datasets", json={"name": name, "dataset_type": "stt"}, headers=key)
    assert again.status_code == 409


def test_other_workspace_dataset_is_refused(client, uploads):
    other_ds = _dataset(client, _signup(client))
    key = _api_key(client, _signup(client))

    for resp in (
        client.get(f"/datasets/{other_ds}", headers=key),
        _upload(client, key, other_ds, _wav(1)),
    ):
        assert resp.status_code == 403
        assert "organization_uuid" not in resp.json()
    assert uploads == []
    missing = "00000000-0000-4000-8000-000000000001"
    assert client.get(f"/datasets/{missing}", headers=key).status_code == 404


def test_tts_dataset_refuses_upload(client, uploads):
    h = _signup(client)
    resp = _upload(client, _api_key(client, h), _dataset(client, h, "tts"), _wav(1))
    assert resp.status_code == 400
    assert "STT" in resp.json()["detail"]
    assert uploads == []


def test_non_wav_is_refused(client, uploads):
    h = _signup(client)
    resp = _upload(client, _api_key(client, h), _dataset(client, h), b"ID3not a wav at all")
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Audio file must be a WAV file"
    assert uploads == []


def test_over_size_limit_is_refused(client, uploads):
    h = _signup(client)
    data = b"\x00" * (datasets_router.MAX_UPLOAD_AUDIO_BYTES + 1)
    resp = _upload(client, _api_key(client, h), _dataset(client, h), data)
    assert resp.status_code == 413
    assert uploads == []


def test_over_length_limit_is_refused(client, uploads):
    h = _signup(client)
    ds = _dataset(client, h)
    key = _api_key(client, h)
    # 8 kHz mono 16-bit keeps 10 minutes under 10 MB, so the length check is what refuses it.
    assert _upload(client, key, ds, _wav(datasets_router.MAX_UPLOAD_AUDIO_SECONDS + 1)).status_code == 400
    assert uploads == []
    assert _upload(client, key, ds, _wav(datasets_router.MAX_UPLOAD_AUDIO_SECONDS)).status_code == 201


def test_missing_bucket_config_is_a_server_error(client, uploads, monkeypatch):
    def _no_bucket():
        raise ValueError("S3_OUTPUT_BUCKET environment variable is required")

    monkeypatch.setattr(datasets_router, "get_s3_output_config", _no_bucket)
    h = _signup(client)
    resp = _upload(client, _api_key(client, h), _dataset(client, h), _wav(1))
    assert resp.status_code == 500
    assert uploads == []


def test_wav_with_zero_sample_rate_is_refused(tmp_path):
    path = tmp_path / "zero.wav"
    path.write_bytes(_wav(1))
    data = bytearray(path.read_bytes())
    data[24:28] = (0).to_bytes(4, "little")
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        datasets_router._wav_duration_seconds(str(path))
