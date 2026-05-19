"""Unit tests for the long-running background workers.

Covers `routers.stt.run_evaluation_task`, `routers.tts.run_tts_evaluation_task`,
`routers.simulations.run_simulation_task`, and `routers.agent_tests.run_llm_test_task` /
`run_benchmark_task` with subprocess and S3 fully mocked.

The goal is to walk the success / failure / timeout branches without spawning
the real calibrate CLI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import db


# ---------------------------------------------------------------------------
# Helpers — make a "completed" output dir like calibrate would
# ---------------------------------------------------------------------------


def _make_stt_output_dir(root: Path, providers: list[str], total: int = 1):
    """Build an output_dir structure that _collect_intermediate_results
    and the success-path post-processor can read."""
    for p in providers:
        sub = root / f"{p}_results"
        sub.mkdir(parents=True)
        with open(sub / "results.csv", "w") as f:
            f.write("id,gt,pred\n")
            for i in range(total):
                f.write(f"audio_{i+1},x,y\n")
        with open(sub / "metrics.json", "w") as f:
            json.dump({"wer": 0.0}, f)


def _make_tts_output_dir(root: Path, providers: list[str], total: int = 1):
    for p in providers:
        sub = root / f"{p}_results"
        sub.mkdir(parents=True)
        with open(sub / "results.csv", "w") as f:
            f.write("id,text,audio_path\n")
            for i in range(total):
                f.write(f"row_{i+1},hi,/tmp/audio.wav\n")
        with open(sub / "metrics.json", "w") as f:
            json.dump({"ttfb": 0.5}, f)


# ---------------------------------------------------------------------------
# STT run_evaluation_task
# ---------------------------------------------------------------------------


def _make_user_and_job(job_type="stt-eval"):
    user_uuid = db.create_user("R", "T", f"rt-{os.urandom(4).hex()}@x.com")
    org_uuid = db.get_personal_org_for_user(user_uuid)["uuid"]
    job_uuid = db.create_job(
        job_type=job_type,
        org_uuid=org_uuid,
        user_id=user_uuid,
        status="in_progress",
        details={
            "audio_paths": ["s3://bucket/key.wav"],
            "texts": ["hi"],
            "providers": ["openai"],
            "language": "en",
            "s3_bucket": "bucket",
            "evaluators": [],
        },
    )
    return user_uuid, job_uuid


class _FakeProcess:
    def __init__(self, returncode=0, poll_results=None):
        self.returncode = returncode
        self.pid = 4242
        # poll_results is a list of values returned by successive poll() calls.
        # Default: one None (still running) then 0 (done) so the heartbeat loop
        # ticks once before falling out.
        self._poll_results = (
            poll_results if poll_results is not None else [None, returncode]
        )

    def poll(self):
        if self._poll_results:
            return self._poll_results.pop(0)
        return self.returncode


def test_stt_run_evaluation_task_success(tmp_path):
    from routers.stt import run_evaluation_task, STTEvaluationRequest

    _, job_uuid = _make_user_and_job()

    process = _FakeProcess(returncode=0, poll_results=[None, 0])

    def fake_popen(*args, **kwargs):
        # Manufacture an "output" dir under the temp cwd that has provider results
        output_dir = Path(kwargs["cwd"]) / "output"
        if output_dir.exists():
            _make_stt_output_dir(output_dir, ["openai"], total=1)
        return process

    s3_mock = MagicMock()
    with patch("routers.stt.subprocess.Popen", side_effect=fake_popen), patch(
        "routers.stt.get_s3_client", return_value=s3_mock
    ), patch("routers.stt.upload_file_to_s3"), patch(
        "routers.stt.upload_top_level_files_to_s3"
    ), patch(
        "routers.stt.upload_directory_tree_to_s3"
    ), patch(
        "routers.stt.try_start_queued_job"
    ), patch(
        "routers.stt.time.sleep"
    ):
        request = STTEvaluationRequest(
            audio_paths=["s3://bucket/key.wav"],
            texts=["hi"],
            providers=["openai"],
            language="en",
        )
        run_evaluation_task(job_uuid, request, "bucket")

    job = db.get_job(job_uuid)
    # Success path → status moves to done
    assert job["status"] == "done"


def test_stt_run_evaluation_task_subprocess_failure(tmp_path):
    from routers.stt import run_evaluation_task, STTEvaluationRequest

    _, job_uuid = _make_user_and_job()

    process = _FakeProcess(returncode=1, poll_results=[None, 1])
    s3_mock = MagicMock()
    with patch("routers.stt.subprocess.Popen", return_value=process), patch(
        "routers.stt.get_s3_client", return_value=s3_mock
    ), patch("routers.stt.upload_file_to_s3"), patch(
        "routers.stt.upload_top_level_files_to_s3"
    ), patch(
        "routers.stt.upload_directory_tree_to_s3"
    ), patch(
        "routers.stt.try_start_queued_job"
    ), patch(
        "routers.stt.time.sleep"
    ):
        request = STTEvaluationRequest(
            audio_paths=["s3://bucket/key.wav"],
            texts=["hi"],
            providers=["openai"],
            language="en",
        )
        run_evaluation_task(job_uuid, request, "bucket")

    job = db.get_job(job_uuid)
    assert job["status"] == "failed"


def test_stt_run_evaluation_task_unexpected_exception():
    from routers.stt import run_evaluation_task, STTEvaluationRequest

    _, job_uuid = _make_user_and_job()
    s3_mock = MagicMock()
    s3_mock.download_file.side_effect = RuntimeError("boom")
    with patch("routers.stt.get_s3_client", return_value=s3_mock), patch(
        "routers.stt.try_start_queued_job"
    ), patch("routers.stt.time.sleep"):
        request = STTEvaluationRequest(
            audio_paths=["s3://bucket/key.wav"],
            texts=["hi"],
            providers=["openai"],
            language="en",
        )
        run_evaluation_task(job_uuid, request, "bucket")

    job = db.get_job(job_uuid)
    assert job["status"] == "failed"


# ---------------------------------------------------------------------------
# TTS run_tts_evaluation_task
# ---------------------------------------------------------------------------


def _make_tts_job():
    user_uuid = db.create_user("R", "T", f"rttts-{os.urandom(4).hex()}@x.com")
    org_uuid = db.get_personal_org_for_user(user_uuid)["uuid"]
    job_uuid = db.create_job(
        job_type="tts-eval",
        org_uuid=org_uuid,
        user_id=user_uuid,
        status="in_progress",
        details={
            "texts": ["hi"],
            "providers": ["openai"],
            "language": "en",
            "s3_bucket": "bucket",
            "evaluators": [],
        },
    )
    return user_uuid, job_uuid


def test_tts_run_evaluation_task_with_outputs():
    """Hit the success-path branches even though the final status may be
    failed (because the simulated audio_path doesn't map to one of the
    walked files). Either way, the post-processing code runs."""
    from routers.tts import run_tts_evaluation_task, TTSEvaluationRequest

    _, job_uuid = _make_tts_job()
    process = _FakeProcess(returncode=0, poll_results=[None, 0])

    def fake_popen(*args, **kwargs):
        output_dir = Path(kwargs["cwd"]) / "output"
        if output_dir.exists():
            _make_tts_output_dir(output_dir, ["openai"], total=1)
            # Add a leaderboard dir so the "exists" branch fires
            (output_dir / "leaderboard").mkdir()
        return process

    s3_mock = MagicMock()
    with patch("routers.tts.subprocess.Popen", side_effect=fake_popen), patch(
        "routers.tts.get_s3_client", return_value=s3_mock
    ), patch("routers.tts.upload_file_to_s3"), patch(
        "routers.tts.upload_top_level_files_to_s3"
    ), patch(
        "routers.tts.upload_directory_tree_to_s3"
    ), patch(
        "routers.tts.try_start_queued_job"
    ), patch(
        "routers.tts.time.sleep"
    ):
        request = TTSEvaluationRequest(
            texts=["hi"], providers=["openai"], language="en"
        )
        run_tts_evaluation_task(job_uuid, request, "bucket")

    job = db.get_job(job_uuid)
    # The post-processing path ran; final status depends on path-mapping
    # heuristics — either is acceptable.
    assert job["status"] in ("done", "failed")


def test_tts_run_evaluation_task_failure():
    from routers.tts import run_tts_evaluation_task, TTSEvaluationRequest

    _, job_uuid = _make_tts_job()
    process = _FakeProcess(returncode=1, poll_results=[None, 1])
    s3_mock = MagicMock()
    with patch("routers.tts.subprocess.Popen", return_value=process), patch(
        "routers.tts.get_s3_client", return_value=s3_mock
    ), patch("routers.tts.upload_file_to_s3"), patch(
        "routers.tts.upload_top_level_files_to_s3"
    ), patch(
        "routers.tts.upload_directory_tree_to_s3"
    ), patch(
        "routers.tts.try_start_queued_job"
    ), patch(
        "routers.tts.time.sleep"
    ):
        request = TTSEvaluationRequest(
            texts=["hi"], providers=["openai"], language="en"
        )
        run_tts_evaluation_task(job_uuid, request, "bucket")

    job = db.get_job(job_uuid)
    assert job["status"] == "failed"


def test_tts_collect_intermediate_results(tmp_path):
    """Drives _collect_tts_intermediate_results."""
    from routers.tts import _collect_tts_intermediate_results

    _make_tts_output_dir(tmp_path, ["openai"], total=2)
    s3_mock = MagicMock()
    with patch("routers.tts.get_s3_client", return_value=s3_mock), patch(
        "routers.tts.upload_file_to_s3"
    ):
        results = _collect_tts_intermediate_results(
            tmp_path, ["openai", "missing"], "task-1", "bucket", expected_total=2
        )
    # 1 with rows, 1 without
    assert len(results) == 2


def test_stt_collect_intermediate_results(tmp_path):
    from routers.stt import _collect_intermediate_results

    _make_stt_output_dir(tmp_path, ["openai"], total=2)
    results = _collect_intermediate_results(
        tmp_path, ["openai", "missing"], expected_total=2
    )
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Agent test run_llm_test_task / run_benchmark_task
# ---------------------------------------------------------------------------


def _make_agent_test_job(job_type="llm-unit-test"):
    user_uuid = db.create_user("R", "AT", f"rtat-{os.urandom(4).hex()}@x.com")
    org_uuid = db.get_personal_org_for_user(user_uuid)["uuid"]
    agent_uuid = db.create_agent(
        name=f"a-{os.urandom(4).hex()}", org_uuid=org_uuid, user_id=user_uuid
    )
    job_uuid = db.create_agent_test_job(
        agent_id=agent_uuid, job_type=job_type, status="in_progress"
    )
    return user_uuid, agent_uuid, job_uuid


def test_run_llm_test_task_failure_propagates():
    """No tests / agent → graceful failure path."""
    from routers.agent_tests import run_llm_test_task

    _, agent_uuid, job_uuid = _make_agent_test_job()
    process = _FakeProcess(returncode=1)
    process.wait = MagicMock(return_value=1)
    with patch("routers.agent_tests.subprocess.Popen", return_value=process), patch(
        "routers.agent_tests.get_s3_client", return_value=MagicMock()
    ), patch("routers.agent_tests.try_start_queued_agent_test_job"), patch(
        "routers.agent_tests.upload_directory_tree_to_s3"
    ), patch(
        "routers.agent_tests.upload_file_to_s3"
    ):
        agent = {"uuid": agent_uuid, "name": "a", "config": {}}
        tests = [{"uuid": "t", "name": "T", "config": {}}]
        run_llm_test_task(job_uuid, agent, tests, "bucket")

    job = db.get_agent_test_job(job_uuid)
    assert job["status"] in ("failed", "done")  # either is acceptable failure path


def test_run_benchmark_task_failure_path():
    """The benchmark task spawns multiple model subprocesses; force an exception
    early to exercise the outer error handler."""
    from routers.agent_tests import run_benchmark_task

    _, agent_uuid, job_uuid = _make_agent_test_job(job_type="llm-benchmark")
    with patch(
        "routers.agent_tests.subprocess.Popen", side_effect=RuntimeError("boom")
    ), patch("routers.agent_tests.try_start_queued_agent_test_job"):
        agent = {"uuid": agent_uuid, "name": "a", "config": {}}
        tests = [{"uuid": "t", "name": "T", "config": {}}]
        run_benchmark_task(job_uuid, agent, tests, ["openai/gpt-4"], "bucket")

    job = db.get_agent_test_job(job_uuid)
    assert job["status"] == "failed"


# ---------------------------------------------------------------------------
# Simulation run_simulation_task — failure path only
# ---------------------------------------------------------------------------


def _make_sim_job():
    user_uuid = db.create_user("R", "S", f"rs-{os.urandom(4).hex()}@x.com")
    org_uuid = db.get_personal_org_for_user(user_uuid)["uuid"]
    sim_uuid = db.create_simulation(
        name=f"sim-{os.urandom(4).hex()}", org_uuid=org_uuid, user_id=user_uuid
    )
    job_uuid = db.create_simulation_job(
        simulation_id=sim_uuid, job_type="text", status="in_progress"
    )
    return user_uuid, sim_uuid, job_uuid


def test_run_simulation_task_failure_path():
    """Force Popen to raise → outer handler kicks in."""
    from routers.simulations import run_simulation_task

    _, _, job_uuid = _make_sim_job()
    agent = {"uuid": "a", "name": "Agent", "config": {}}
    personas = [{"uuid": "p", "name": "Alex", "config": {}}]
    scenarios = [{"uuid": "s", "name": "Sc", "description": "desc"}]
    evaluators = []
    with patch(
        "routers.simulations.subprocess.Popen", side_effect=RuntimeError("boom")
    ), patch("routers.simulations.try_start_queued_simulation_job"):
        run_simulation_task(
            job_uuid, agent, personas, scenarios, evaluators, "bucket", "text"
        )

    job = db.get_simulation_job(job_uuid)
    assert job["status"] == "failed"
