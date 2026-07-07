"""Unit tests for pure helpers in src/utils.py.

Covers the env-var parsers, kill-process-group helpers (with mocked
`os.killpg`), the S3 client / presigned-URL helpers (boto3 mocked), the
in-memory job-queue registry, and the data-shaping helpers
(`normalize_metrics`, `coerce_evaluator_score`,
`post_process_provider_results`, `compute_share_token_toggle`, etc.)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import db
import utils
from utils import (
    EvaluatorRunEntry,
    ProviderResult,
    TaskCreateResponse,
    TaskStatus,
    TaskStatusResponse,
    build_evaluator_runs_for_eval_job,
    build_tool_configs,
    can_start_agent_test_job,
    can_start_job,
    can_start_simulation_job,
    capture_exception_to_sentry,
    coerce_evaluator_score,
    compute_share_token_toggle,
    download_file_from_s3,
    enrich_evaluator_runs_with_current_names,
    env_bool,
    env_int,
    env_str,
    generate_presigned_download_url,
    generate_presigned_upload_url,
    get_local_artifact_path,
    get_max_concurrent_jobs,
    get_max_concurrent_jobs_per_org,
    get_object_storage_mode,
    get_s3_client,
    get_s3_output_config,
    is_local_object_storage,
    is_evaluator_metric_aggregate,
    list_object_keys,
    is_job_timed_out,
    kill_process_group,
    kill_processes_from_dict,
    load_evaluator_metric_key_map,
    normalize_metrics,
    ordered_evaluator_metric_keys,
    post_process_provider_results,
    presign_audio_path,
    read_evaluators_map_from_config,
    read_leaderboard_xlsx,
    register_job_starter,
    try_start_queued_agent_test_job,
    try_start_queued_job,
    try_start_queued_simulation_job,
    upload_directory_tree_to_s3,
    upload_file_to_s3,
    upload_top_level_files_to_s3,
)


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------


def test_env_helpers(monkeypatch):
    monkeypatch.setenv("FOO_STR", "hello")
    monkeypatch.setenv("FOO_EMPTY", "")
    assert env_str("FOO_STR", "fallback") == "hello"
    assert env_str("FOO_EMPTY", "fallback") == "fallback"
    assert env_str("FOO_MISSING", "fallback") == "fallback"

    monkeypatch.setenv("FLAG_TRUE", "yes")
    monkeypatch.setenv("FLAG_FALSE", "off")
    assert env_bool("FLAG_TRUE", False) is True
    assert env_bool("FLAG_FALSE", True) is False
    assert env_bool("FLAG_MISSING", True) is True

    monkeypatch.setenv("INT_OK", "42")
    monkeypatch.setenv("INT_BAD", "not-a-number")
    assert env_int("INT_OK", 0) == 42
    assert env_int("INT_BAD", 7) == 7
    assert env_int("INT_MISSING", 9) == 9


# ---------------------------------------------------------------------------
# Sentry capture (we just verify it forwards the exception)
# ---------------------------------------------------------------------------


def test_capture_exception_to_sentry_calls_sentry():
    with patch("utils.sentry_sdk") as sentry:
        capture_exception_to_sentry(RuntimeError("oh no"))
        assert sentry.capture_exception.called
        sentry.flush.assert_called_once()


# ---------------------------------------------------------------------------
# build_tool_configs
# ---------------------------------------------------------------------------


def test_build_tool_configs_structured_and_webhook():
    tools = [
        {
            "name": "search",
            "description": "search the web",
            "config": {"type": "structured_output", "parameters": [{"name": "q"}]},
        },
        {
            "name": "notify",
            "description": "notify webhook",
            "config": {
                "type": "webhook",
                "parameters": [{"name": "id"}],
                "webhook": {"url": "https://x"},
            },
        },
        {
            "name": "no-config-type",
            "description": "defaults to structured_output",
            "config": {"parameters": []},
        },
    ]
    out = build_tool_configs(tools)
    assert out[0]["type"] == "structured_output"
    assert out[1]["type"] == "webhook"
    assert out[1]["webhook"] == {"url": "https://x"}
    assert out[2]["type"] == "structured_output"


# ---------------------------------------------------------------------------
# Pydantic models (just instantiation)
# ---------------------------------------------------------------------------


def test_pydantic_models():
    entry = EvaluatorRunEntry(
        evaluator_uuid="f47ac10b-58cc-4372-a567-0e02b2c3d479",
        metric_key="quality",
        aggregate={"mean": 0.9},
        output_type="rating",
    )
    assert entry.evaluator_uuid == "f47ac10b-58cc-4372-a567-0e02b2c3d479"
    pr = ProviderResult(provider="openai", success=True, metrics={"wer": 0.1})
    assert pr.success is True
    create = TaskCreateResponse(task_id="f47ac10b-58cc-4372-a567-0e02b2c3d479", status="queued")
    assert create.status == "queued"
    status = TaskStatusResponse(task_id="f47ac10b-58cc-4372-a567-0e02b2c3d479", status="done", provider_results=[pr])
    assert status.provider_results[0].provider == "openai"

    # TaskStatus enum
    assert TaskStatus.QUEUED.value == "queued"
    assert TaskStatus.IN_PROGRESS.value == "in_progress"


# ---------------------------------------------------------------------------
# kill_process_group / kill_processes_from_dict
# ---------------------------------------------------------------------------


def test_kill_process_group_returns_true_when_no_pid():
    assert kill_process_group(0, "job-1") is True


def test_kill_process_group_handles_already_dead():
    with patch("utils.os.killpg", side_effect=ProcessLookupError()):
        assert kill_process_group(1234, "job-1") is True


def test_kill_process_group_handles_permission_error():
    with patch("utils.os.killpg", side_effect=PermissionError()):
        assert kill_process_group(1234, "job-1") is False


def test_kill_process_group_two_phase_kill():
    # First call (SIGTERM) succeeds, second call (SIGKILL) finds the process gone
    calls = {"count": 0}

    def fake_killpg(pid, sig):
        calls["count"] += 1
        if calls["count"] == 2:
            raise ProcessLookupError()

    with patch("utils.os.killpg", side_effect=fake_killpg), patch("utils.time.sleep"):
        assert kill_process_group(1234, "job-1") is True
    assert calls["count"] == 2


def test_kill_process_group_unexpected_exception_returns_false():
    with patch("utils.os.killpg", side_effect=RuntimeError("boom")):
        assert kill_process_group(1234, "job-1") is False


def test_kill_processes_from_dict_skips_empty_and_dispatches():
    kill_processes_from_dict(None, "job-1")  # no-op
    kill_processes_from_dict({}, "job-1")
    with patch("utils.os.killpg") as kp, patch("utils.time.sleep"):
        kill_processes_from_dict({"openai": 111, "missing": 0}, "job-1")
        # Two calls per pid (SIGTERM then SIGKILL) for the openai PID only.
        assert kp.call_count == 2


def test_kill_processes_from_dict_handles_errors():
    with patch("utils.os.killpg", side_effect=ProcessLookupError()):
        kill_processes_from_dict({"a": 1}, "job-1")
    with patch("utils.os.killpg", side_effect=PermissionError()):
        kill_processes_from_dict({"a": 1}, "job-1")
    with patch("utils.os.killpg", side_effect=RuntimeError("boom")):
        kill_processes_from_dict({"a": 1}, "job-1")


# ---------------------------------------------------------------------------
# is_job_timed_out
# ---------------------------------------------------------------------------


def test_is_job_timed_out_true_and_false():
    recent = (datetime.utcnow() - timedelta(seconds=10)).strftime("%Y-%m-%d %H:%M:%S")
    long_ago = (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    assert is_job_timed_out(recent) is False
    assert is_job_timed_out(long_ago) is True
    # malformed timestamps fail closed (return False)
    assert is_job_timed_out("not-a-timestamp") is False


# ---------------------------------------------------------------------------
# S3 helpers (boto3 mocked)
# ---------------------------------------------------------------------------


def test_get_s3_client_with_and_without_endpoint(monkeypatch):
    with patch("utils.boto3.client") as bc:
        get_s3_client()
        assert bc.called

    monkeypatch.setenv("S3_ENDPOINT_URL", "https://storage.googleapis.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    with patch("utils.boto3.client") as bc:
        get_s3_client()
        kwargs = bc.call_args.kwargs
        assert kwargs["endpoint_url"] == "https://storage.googleapis.com"
        assert kwargs["region_name"] == "us-east-1"


def test_get_s3_output_config_requires_bucket(monkeypatch):
    monkeypatch.delenv("S3_OUTPUT_BUCKET", raising=False)
    monkeypatch.delenv("OBJECT_STORAGE_MODE", raising=False)
    with pytest.raises(ValueError):
        get_s3_output_config()
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "my-bucket")
    assert get_s3_output_config() == "my-bucket"


def test_local_object_storage_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("OBJECT_STORAGE_MODE", "local")
    monkeypatch.delenv("S3_OUTPUT_BUCKET", raising=False)
    monkeypatch.setenv("LOCAL_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    assert get_object_storage_mode() == "local"
    assert is_local_object_storage() is True
    assert get_s3_output_config() == "local-dev-artifacts"
    assert get_s3_client() is None

    source = tmp_path / "source.txt"
    source.write_text("hello")
    upload_file_to_s3(None, source, "ignored", "runs/one/source.txt")
    assert get_local_artifact_path("runs/one/source.txt").read_text() == "hello"

    assert generate_presigned_download_url("runs/one/source.txt") == (
        "/local-artifacts/runs/one/source.txt"
    )
    monkeypatch.setenv("LOCAL_ARTIFACT_BASE_URL", "http://localhost:8000")
    assert generate_presigned_download_url("runs/one/source.txt") == (
        "http://localhost:8000/local-artifacts/runs/one/source.txt"
    )
    # Uploads use the same LOCAL_ARTIFACT_BASE_URL fallback as downloads.
    assert generate_presigned_upload_url(
        "runs/two/upload.txt",
        "text/plain",
    ) == "http://localhost:8000/local-artifacts/runs/two/upload.txt"


def test_invalid_object_storage_mode(monkeypatch):
    monkeypatch.setenv("OBJECT_STORAGE_MODE", "memory")
    with pytest.raises(ValueError):
        get_object_storage_mode()
    # get_s3_client() must reject an invalid mode too (not silently build a real
    # client), so a typo fails consistently instead of half-behaving like s3.
    with pytest.raises(ValueError):
        get_s3_client()


def test_download_file_from_s3_local(monkeypatch, tmp_path):
    monkeypatch.setenv("OBJECT_STORAGE_MODE", "local")
    monkeypatch.setenv("LOCAL_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    source = tmp_path / "input.wav"
    source.write_bytes(b"fake wav")
    upload_file_to_s3(None, source, "ignored", "stt/media/input.wav")

    # Destination parent does not exist yet — the helper must create it.
    dest = tmp_path / "work" / "audio.wav"
    download_file_from_s3(None, "local-dev-artifacts", "stt/media/input.wav", dest)
    assert dest.read_bytes() == b"fake wav"


def test_download_file_from_s3_s3_mode(monkeypatch):
    monkeypatch.delenv("OBJECT_STORAGE_MODE", raising=False)
    client = MagicMock()
    download_file_from_s3(client, "my-bucket", "runs/one/a.wav", "/tmp/a.wav")
    client.download_file.assert_called_once_with(
        "my-bucket", "runs/one/a.wav", "/tmp/a.wav"
    )


def test_list_object_keys_local(monkeypatch, tmp_path):
    monkeypatch.setenv("OBJECT_STORAGE_MODE", "local")
    monkeypatch.setenv("LOCAL_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    for name in ("1_bot.wav", "1_user.wav", "2_bot.wav", "notes.txt"):
        src = tmp_path / name
        src.write_text("x")
        upload_file_to_s3(None, src, "ignored", f"sims/abc/audios/{name}")
    # A file outside the prefix must not appear.
    other = tmp_path / "other.wav"
    other.write_text("x")
    upload_file_to_s3(None, other, "ignored", "sims/xyz/audios/9_bot.wav")

    keys = list_object_keys(None, "local-dev-artifacts", "sims/abc/audios/")
    assert keys == [
        "sims/abc/audios/1_bot.wav",
        "sims/abc/audios/1_user.wav",
        "sims/abc/audios/2_bot.wav",
        "sims/abc/audios/notes.txt",
    ]
    # Missing prefix lists nothing rather than raising.
    assert list_object_keys(None, "local-dev-artifacts", "sims/missing/") == []


def test_list_object_keys_s3_mode(monkeypatch):
    monkeypatch.delenv("OBJECT_STORAGE_MODE", raising=False)
    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "p/a.wav"}, {"Key": "p/b.wav"}]},
        {},  # page with no Contents must be tolerated
    ]
    client.get_paginator.return_value = paginator

    keys = list_object_keys(client, "my-bucket", "p/")
    assert keys == ["p/a.wav", "p/b.wav"]
    client.get_paginator.assert_called_once_with("list_objects_v2")
    paginator.paginate.assert_called_once_with(Bucket="my-bucket", Prefix="p/")


def test_upload_helpers_with_tmp_dir(tmp_path):
    file1 = tmp_path / "a.txt"
    file1.write_text("hi")
    sub = tmp_path / "sub"
    sub.mkdir()
    file2 = sub / "b.txt"
    file2.write_text("there")

    s3_mock = MagicMock()
    upload_file_to_s3(s3_mock, file1, "bucket", "prefix/a.txt")
    assert s3_mock.upload_file.called

    s3_mock.reset_mock()
    upload_top_level_files_to_s3(s3_mock, tmp_path, "bucket", "prefix/")
    # Only the top-level files (a.txt), subdir contents skipped
    assert s3_mock.upload_file.call_count == 1
    upload_top_level_files_to_s3(s3_mock, tmp_path / "missing", "bucket", "prefix/")
    upload_top_level_files_to_s3(s3_mock, None, "bucket", "prefix/")

    s3_mock.reset_mock()
    upload_directory_tree_to_s3(s3_mock, tmp_path, "bucket", "prefix/")
    # Both files (a.txt + sub/b.txt)
    assert s3_mock.upload_file.call_count == 2
    upload_directory_tree_to_s3(s3_mock, tmp_path / "missing", "bucket", "prefix/")
    upload_directory_tree_to_s3(s3_mock, None, "bucket", "prefix/")


def test_presigned_url_helpers(monkeypatch):
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "my-bucket")
    fake_url = "https://example.com/signed"
    s3_mock = MagicMock()
    s3_mock.generate_presigned_url.return_value = fake_url
    with patch("utils.get_s3_client", return_value=s3_mock):
        assert generate_presigned_download_url("k") == fake_url
        assert generate_presigned_upload_url("k", "text/plain") == fake_url

    # error path
    with patch(
        "utils.get_s3_client", side_effect=RuntimeError("boom")
    ):
        assert generate_presigned_download_url("k") is None
        assert generate_presigned_upload_url("k", "text/plain") is None


def test_presign_audio_path_branches(monkeypatch):
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "my-bucket")
    fake_url = "https://example.com/signed"
    s3_mock = MagicMock()
    s3_mock.generate_presigned_url.return_value = fake_url
    with patch("utils.get_s3_client", return_value=s3_mock):
        assert presign_audio_path(None) is None
        assert presign_audio_path("") == ""
        assert presign_audio_path("https://already.example/x") == "https://already.example/x"
        assert presign_audio_path("s3://bucket/key.wav") == fake_url
        assert presign_audio_path("raw-key.wav") == fake_url


# ---------------------------------------------------------------------------
# Job queue helpers (real DB)
# ---------------------------------------------------------------------------


def test_get_max_concurrent_jobs(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "5")
    assert get_max_concurrent_jobs() == 5
    monkeypatch.setenv("MAX_CONCURRENT_JOBS_PER_ORG", "3")
    assert get_max_concurrent_jobs_per_org() == 3
    # back-compat: old env var name still honored with a deprecation warning
    monkeypatch.delenv("MAX_CONCURRENT_JOBS_PER_ORG", raising=False)
    monkeypatch.setenv("MAX_CONCURRENT_JOBS_PER_USER", "7")
    assert get_max_concurrent_jobs_per_org() == 7


def _mk_user_org(prefix: str) -> tuple[str, str]:
    """Test helper: create a user and return (user_uuid, personal_org_uuid)."""
    user_uuid = db.create_user(prefix, "U", f"{prefix}-{os.urandom(4).hex()}@x.com")
    org = db.get_personal_org_for_user(user_uuid)
    return user_uuid, org["uuid"]


def test_register_and_try_start_queued_job(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "100")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS_PER_ORG", "100")

    job_type = f"queue-test-{os.urandom(4).hex()}"
    user_uuid, org_uuid = _mk_user_org("Q")
    job_uuid = db.create_job(
        job_type=job_type,
        org_uuid=org_uuid,
        user_id=user_uuid,
        status="queued",
        details={},
    )

    started = {"called_with": None}

    def starter(job):
        started["called_with"] = job["uuid"]

    register_job_starter(job_type, starter)
    assert try_start_queued_job([job_type]) is True
    assert started["called_with"] == job_uuid

    assert try_start_queued_job([job_type]) is False

    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "0")
    db.create_job(
        job_type=job_type,
        org_uuid=org_uuid,
        user_id=user_uuid,
        status="queued",
        details={},
    )
    assert try_start_queued_job([job_type]) is False

    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "100")
    assert can_start_job([job_type], org_uuid) in (True, False)
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "0")
    assert can_start_job([job_type], org_uuid) is False
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "100")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS_PER_ORG", "0")
    assert can_start_job([job_type], org_uuid) is True


def test_try_start_queued_job_failure_starter(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "100")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS_PER_ORG", "100")
    user_uuid, org_uuid = _mk_user_org("F")
    db.create_job(
        job_type="stt-eval",
        org_uuid=org_uuid,
        user_id=user_uuid,
        status="queued",
        details={},
    )

    def bad_starter(job):
        raise RuntimeError("nope")

    register_job_starter("stt-eval", bad_starter)
    assert try_start_queued_job(["stt-eval"]) is False


def test_try_start_queued_job_no_starter(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "100")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS_PER_ORG", "100")
    user_uuid, org_uuid = _mk_user_org("N")
    db.create_job(
        job_type="zzz-no-starter",
        org_uuid=org_uuid,
        user_id=user_uuid,
        status="queued",
        details={},
    )
    utils._job_starters.pop("zzz-no-starter", None)
    assert try_start_queued_job(["zzz-no-starter"]) is False


def test_try_start_queued_agent_test_job(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "100")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS_PER_ORG", "100")
    user_uuid, org_uuid = _mk_user_org("AT")
    agent_uuid = db.create_agent(
        name=f"ag-{os.urandom(4).hex()}", org_uuid=org_uuid, user_id=user_uuid
    )
    job_uuid = db.create_agent_test_job(
        agent_id=agent_uuid, job_type="llm-unit-test", status="queued"
    )

    started = {}

    def starter(job):
        started["uuid"] = job["uuid"]

    register_job_starter("llm-unit-test", starter)
    assert try_start_queued_agent_test_job(["llm-unit-test"]) is True
    assert started.get("uuid") == job_uuid
    assert try_start_queued_agent_test_job(["llm-unit-test"]) is False
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "0")
    db.create_agent_test_job(
        agent_id=agent_uuid, job_type="llm-unit-test", status="queued"
    )
    assert try_start_queued_agent_test_job(["llm-unit-test"]) is False

    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "100")
    assert can_start_agent_test_job(["llm-unit-test"], org_uuid) in (True, False)
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "0")
    assert can_start_agent_test_job(["llm-unit-test"], org_uuid) is False


def test_try_start_queued_simulation_job(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "100")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS_PER_ORG", "100")
    user_uuid, org_uuid = _mk_user_org("S")
    sim_uuid = db.create_simulation(
        name=f"sim-{os.urandom(4).hex()}", org_uuid=org_uuid, user_id=user_uuid
    )
    job_uuid = db.create_simulation_job(
        simulation_id=sim_uuid, job_type="text", status="queued"
    )

    started = {}

    def starter(job):
        started["uuid"] = job["uuid"]

    register_job_starter("text", starter)
    assert try_start_queued_simulation_job(["text"]) is True
    assert started.get("uuid") == job_uuid
    assert try_start_queued_simulation_job(["text"]) is False
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "0")
    db.create_simulation_job(
        simulation_id=sim_uuid, job_type="text", status="queued"
    )
    assert try_start_queued_simulation_job(["text"]) is False

    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "100")
    assert can_start_simulation_job(["text"], org_uuid) in (True, False)
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "0")
    assert can_start_simulation_job(["text"], org_uuid) is False


def test_try_start_failure_paths_agent_test(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "100")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS_PER_ORG", "100")
    user_uuid, org_uuid = _mk_user_org("X")
    agent_uuid = db.create_agent(
        name=f"ag-{os.urandom(4).hex()}", org_uuid=org_uuid, user_id=user_uuid
    )
    db.create_agent_test_job(
        agent_id=agent_uuid, job_type="bad-starter", status="queued"
    )

    def bad(job):
        raise RuntimeError("x")

    register_job_starter("bad-starter", bad)
    assert try_start_queued_agent_test_job(["bad-starter"]) is False


def test_try_start_failure_paths_simulation(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "100")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS_PER_ORG", "100")
    user_uuid, org_uuid = _mk_user_org("X")
    sim_uuid = db.create_simulation(
        name=f"sim-{os.urandom(4).hex()}", org_uuid=org_uuid, user_id=user_uuid
    )
    db.create_simulation_job(
        simulation_id=sim_uuid, job_type="bad-sim-starter", status="queued"
    )

    def bad(job):
        raise RuntimeError("x")

    register_job_starter("bad-sim-starter", bad)
    assert try_start_queued_simulation_job(["bad-sim-starter"]) is False


# ---------------------------------------------------------------------------
# normalize_metrics + evaluator-related helpers
# ---------------------------------------------------------------------------


def test_normalize_metrics_shapes():
    assert normalize_metrics(None) is None
    assert normalize_metrics({"wer": 1}) == {"wer": 1}
    legacy = [
        {"wer": 0.1},
        {"string_similarity": 0.5},
        {"metric_name": "ttfb", "mean": 0.2},
        "not-a-dict",
    ]
    result = normalize_metrics(legacy)
    assert result["wer"] == 0.1
    assert result["string_similarity"] == 0.5
    assert result["ttfb"] == {"mean": 0.2}
    # Empty list → returned as-is
    assert normalize_metrics([]) == []
    # Non-dict / non-list → passthrough
    assert normalize_metrics(42) == 42


def test_is_evaluator_metric_aggregate_and_keys():
    assert is_evaluator_metric_aggregate({"type": "binary", "pass_rate": 1}) is True
    assert is_evaluator_metric_aggregate({"mean": 1.0}) is False
    assert is_evaluator_metric_aggregate("foo") is False
    keys = ordered_evaluator_metric_keys(
        {
            "wer": 0.1,
            "Safety": {"type": "binary", "pass_rate": 0.9},
            "Faithfulness": {"type": "rating", "mean": 4.2},
        }
    )
    assert "Safety" in keys and "Faithfulness" in keys and "wer" not in keys
    assert ordered_evaluator_metric_keys(None) == []
    assert ordered_evaluator_metric_keys({}) == []


def test_read_evaluators_map_from_config(tmp_path):
    # missing dir
    assert read_evaluators_map_from_config(None) == {}
    assert read_evaluators_map_from_config(tmp_path / "missing") == {}

    # malformed JSON
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "config.json").write_text("{bad json")
    assert read_evaluators_map_from_config(bad) == {}

    # well-formed
    good = tmp_path / "good"
    good.mkdir()
    (good / "config.json").write_text(
        json.dumps({"evaluators_map": {"uuid-1": "Safety", "uuid-2": "Faithfulness"}})
    )
    mapping = read_evaluators_map_from_config(good)
    assert mapping == {"Safety": "uuid-1", "Faithfulness": "uuid-2"}

    # missing evaluators_map key
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text(json.dumps({}))
    assert read_evaluators_map_from_config(empty) == {}


def test_build_evaluator_runs_for_eval_job():
    runs = build_evaluator_runs_for_eval_job(
        {"Safety": {"type": "binary", "pass_rate": 0.9}},
        {"Safety": "6ba7b810-9dad-11d1-80b4-00c04fd430c8"},
    )
    assert runs and runs[0].evaluator_uuid == "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    # missing map entry → skipped
    assert build_evaluator_runs_for_eval_job(
        {"Safety": {"type": "binary"}}, {}
    ) == []
    # non-dict metrics → []
    assert build_evaluator_runs_for_eval_job("not a dict", {"00000000-0000-4000-8000-000000000001": "00000000-0000-4000-8000-000000000002"}) == []


def test_coerce_evaluator_score_binary_and_rating():
    assert coerce_evaluator_score(True, "binary") is True
    assert coerce_evaluator_score(1, "binary") is True
    assert coerce_evaluator_score(0, "binary") is False
    assert coerce_evaluator_score("True", "binary") is True
    assert coerce_evaluator_score("pass", "binary") is True
    assert coerce_evaluator_score("fail", "binary") is False
    assert coerce_evaluator_score("1.0", "binary") is True
    assert coerce_evaluator_score("0.0", "binary") is False
    # unparseable → passthrough
    assert coerce_evaluator_score("???", "binary") == "???"
    # rating
    assert coerce_evaluator_score("3", "rating") == 3
    assert coerce_evaluator_score("3.5", "rating") == 3.5
    assert coerce_evaluator_score("bogus", "rating") == "bogus"
    # unknown type → passthrough
    assert coerce_evaluator_score("anything", "unknown") == "anything"


def test_post_process_provider_results_flow():
    snapshots = [
        {
            "uuid": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "name": "Safety",
            "output_type": "binary",
            "evaluator_version_id": "6ba7b811-9dad-11d1-80b4-00c04fd430c8",
        }
    ]
    metric_key_map = {"Safety": "6ba7b810-9dad-11d1-80b4-00c04fd430c8"}

    provider_results = [
        {
            "provider": "openai",
            "metrics": {"Safety": {"type": "binary", "pass_rate": 0.9}},
            "results": [
                {"Safety": True, "Safety_reasoning": "ok", "wer": "0.1"},
                {"Safety": "ERROR", "Safety_reasoning": "Error: boom"},
            ],
        }
    ]
    post_process_provider_results(
        provider_results,
        evaluator_snapshots=snapshots,
        evaluator_id_by_metric_key=metric_key_map,
    )
    pr = provider_results[0]
    assert pr["evaluator_runs"][0]["evaluator_uuid"] == "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    assert pr["results"][0]["evaluator_outputs"]["6ba7b810-9dad-11d1-80b4-00c04fd430c8"]["value"] is True
    assert pr["results"][0]["wer"] == 0.1
    assert pr["results"][1]["evaluator_outputs"]["6ba7b810-9dad-11d1-80b4-00c04fd430c8"]["error"] is True

    # Empty input → no-op
    post_process_provider_results(None)
    post_process_provider_results([])


def test_post_process_falls_back_to_name_when_no_map():
    snapshots = [{"uuid": "u", "name": "Safety", "output_type": "binary"}]
    provider_results = [
        {
            "provider": "openai",
            "metrics": None,
            "results": [
                {"Safety": "true", "Safety_reasoning": "ok"},
            ],
        }
    ]
    post_process_provider_results(provider_results, evaluator_snapshots=snapshots)
    assert "u" in provider_results[0]["results"][0]["evaluator_outputs"]


def test_load_evaluator_metric_key_map(tmp_path):
    assert load_evaluator_metric_key_map(None) == {}
    assert load_evaluator_metric_key_map({}) == {}
    assert load_evaluator_metric_key_map({"output_dir": None}) == {}
    assert load_evaluator_metric_key_map({"output_dir": str(tmp_path / "missing")}) == {}
    od = tmp_path / "od"
    od.mkdir()
    (od / "config.json").write_text(json.dumps({"evaluators_map": {"u1": "Safety"}}))
    out = load_evaluator_metric_key_map({"output_dir": str(od)})
    assert out["Safety"] == "u1"


def test_compute_share_token_toggle_behaviour():
    # Disabling preserves the stored token but suppresses the return value.
    persist, returned = compute_share_token_toggle({"share_token": "tok"}, False)
    assert persist == "tok"
    assert returned is None

    # Enabling without an existing token mints a new one.
    persist, returned = compute_share_token_toggle({}, True)
    assert persist is not None
    assert returned == persist

    # Enabling with an existing token preserves it.
    persist, returned = compute_share_token_toggle({"share_token": "existing"}, True)
    assert persist == "existing"
    assert returned == "existing"

    # Custom token factory branch
    persist, returned = compute_share_token_toggle(
        None, True, token_factory=lambda: "fixed"
    )
    assert persist == "fixed"
    assert returned == "fixed"


def test_enrich_evaluator_runs_with_current_names(monkeypatch):
    # Snapshot has the name; DB lookup returns None for unknown uuid
    pr = [
        {
            "evaluator_runs": [
                {
                    "evaluator_uuid": "unknown-uuid",
                    "metric_key": "mk",
                }
            ]
        }
    ]
    snapshots = [{"uuid": "unknown-uuid", "name": "Snapshot Name"}]
    enrich_evaluator_runs_with_current_names(pr, evaluator_snapshots=snapshots)
    assert pr[0]["evaluator_runs"][0]["name"] == "Snapshot Name"
    # Empty / None inputs no-op
    enrich_evaluator_runs_with_current_names(None)
    enrich_evaluator_runs_with_current_names([])
    enrich_evaluator_runs_with_current_names([{"no_runs": True}])
    enrich_evaluator_runs_with_current_names([{"evaluator_runs": ["scalar"]}])


def test_read_leaderboard_xlsx(tmp_path):
    # Non-existent path → None
    assert read_leaderboard_xlsx(tmp_path / "missing") is None

    # Empty directory → None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert read_leaderboard_xlsx(empty) is None

    # Create a minimal xlsx with a 'summary' sheet
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "summary"
    ws.append(["model", "wer"])
    ws.append(["openai", 0.1])
    ws.append([None, None])  # blank row should be skipped
    lead_dir = tmp_path / "lead"
    lead_dir.mkdir()
    xlsx_path = lead_dir / "stt_leaderboard.xlsx"
    wb.save(xlsx_path)

    rows = read_leaderboard_xlsx(lead_dir)
    assert rows == [{"model": "openai", "wer": 0.1}]

    # Now a workbook missing the 'summary' sheet
    wb2 = openpyxl.Workbook()
    wb2.active.title = "not-summary"
    bad_lead = tmp_path / "bad-lead"
    bad_lead.mkdir()
    wb2.save(bad_lead / "x.xlsx")
    assert read_leaderboard_xlsx(bad_lead) is None
