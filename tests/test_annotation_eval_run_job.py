"""Tests for the long-running annotation_eval_runner._run_job worker.

Mocks subprocess + S3 + DB writes; walks the success and failure paths.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import db


def _make_user_and_task():
    user_id = db.create_user("X", "Y", f"x-{os.urandom(4).hex()}@x.com")
    task_uuid = db.create_annotation_task(
        name=f"task-{os.urandom(4).hex()}", type="stt", user_id=user_id
    )
    return user_id, task_uuid


def _resolved(uuid="ev-1", name="Safety"):
    return {
        "uuid": uuid,
        "name": name,
        "judge_model": "gpt",
        "system_prompt": "p",
        "output_type": "binary",
        "output_config": {},
        "variables": [],
        "variable_values": {},
        "kind": "single",
        "data_type": "text",
        "_evaluator_version_id": "ver-1",
    }


def test_run_job_task_missing():
    """_run_job: get_annotation_task returns None → fail path."""
    from annotation_eval_runner import _run_job

    with patch("annotation_eval_runner.get_annotation_task", return_value=None), patch(
        "annotation_eval_runner.update_job"
    ), patch("annotation_eval_runner.try_start_queued_job"):
        _run_job("j-1", "missing-task", "u-1", [_resolved()], item_ids=None)


def test_run_job_no_items():
    """No snapshot, no live items → fail path."""
    from annotation_eval_runner import _run_job

    with patch(
        "annotation_eval_runner.get_annotation_task", return_value={"type": "stt"}
    ), patch("annotation_eval_runner.get_eval_job_items", return_value=[]), patch(
        "annotation_eval_runner.get_annotation_items_for_task", return_value=[]
    ), patch(
        "annotation_eval_runner.update_job"
    ), patch(
        "annotation_eval_runner.try_start_queued_job"
    ):
        _run_job("j-1", "task", "u-1", [_resolved()], item_ids=None)


def test_run_job_subprocess_failure(tmp_path):
    """Subprocess exits with non-zero → CalledProcessError handler."""
    from annotation_eval_runner import _run_job

    task = {"type": "stt"}
    items = [
        {
            "uuid": "i1",
            "payload": {
                "predicted_transcript": "pred",
                "reference_transcript": "ref",
            },
        }
    ]

    class FakeProcess:
        def __init__(self):
            self.returncode = 1
            self.pid = 4242
            self._poll_results = [None, 1]

        def poll(self):
            if self._poll_results:
                return self._poll_results.pop(0)
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    with patch(
        "annotation_eval_runner.get_annotation_task", return_value=task
    ), patch(
        "annotation_eval_runner.get_eval_job_items", return_value=items
    ), patch(
        "annotation_eval_runner.subprocess.Popen", return_value=FakeProcess()
    ), patch(
        "annotation_eval_runner.update_job"
    ), patch(
        "annotation_eval_runner.try_start_queued_job"
    ), patch(
        "annotation_eval_runner.time.sleep"
    ), patch(
        "annotation_eval_runner.get_job",
        return_value={"updated_at": "2099-01-01 00:00:00"},
    ), patch(
        "annotation_eval_runner._persist_pgid"
    ), patch(
        "annotation_eval_runner.get_s3_client", return_value=MagicMock()
    ), patch(
        "annotation_eval_runner.get_s3_output_config", return_value="bucket"
    ), patch(
        "annotation_eval_runner.upload_file_to_s3"
    ):
        _run_job("j-1", "task", "u-1", [_resolved()], item_ids=None)


def test_run_job_unexpected_exception():
    """build_dataset_for_task_type raises → outer exception handler."""
    from annotation_eval_runner import _run_job

    task = {"type": "stt"}
    items = [{"uuid": "i1", "payload": {"x": "y"}}]  # missing required fields
    with patch(
        "annotation_eval_runner.get_annotation_task", return_value=task
    ), patch(
        "annotation_eval_runner.get_eval_job_items", return_value=items
    ), patch(
        "annotation_eval_runner.update_job"
    ), patch(
        "annotation_eval_runner.try_start_queued_job"
    ), patch(
        "annotation_eval_runner._try_upload_partial_outputs", return_value=None
    ):
        _run_job("j-1", "task", "u-1", [_resolved()], item_ids=None)


def test_run_job_backfill_snapshot(tmp_path):
    """Empty snapshot → falls back to live items + writes snapshot."""
    from annotation_eval_runner import _run_job

    items = [
        {
            "uuid": "i1",
            "payload": {
                "predicted_transcript": "p",
                "reference_transcript": "r",
            },
        }
    ]

    class FakeProcess:
        def __init__(self):
            self.returncode = 1
            self.pid = 1
            self._poll_results = [None, 1]

        def poll(self):
            if self._poll_results:
                return self._poll_results.pop(0)
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    with patch(
        "annotation_eval_runner.get_annotation_task",
        return_value={"type": "stt"},
    ), patch(
        "annotation_eval_runner.get_eval_job_items", return_value=[]
    ), patch(
        "annotation_eval_runner.get_annotation_items_for_task",
        return_value=items,
    ), patch(
        "annotation_eval_runner.snapshot_eval_job_items"
    ), patch(
        "annotation_eval_runner.subprocess.Popen", return_value=FakeProcess()
    ), patch(
        "annotation_eval_runner.update_job"
    ), patch(
        "annotation_eval_runner.try_start_queued_job"
    ), patch(
        "annotation_eval_runner.time.sleep"
    ), patch(
        "annotation_eval_runner.get_job",
        return_value={"updated_at": "2099-01-01 00:00:00"},
    ), patch(
        "annotation_eval_runner._persist_pgid"
    ), patch(
        "annotation_eval_runner.get_s3_client", return_value=MagicMock()
    ), patch(
        "annotation_eval_runner.get_s3_output_config", return_value="bucket"
    ), patch(
        "annotation_eval_runner.upload_file_to_s3"
    ):
        _run_job("j-1", "task", "u-1", [_resolved()], item_ids=None)


def test_run_calibrate_eval_only_success(tmp_path):
    """Drive _run_calibrate_eval_only success path."""
    from annotation_eval_runner import _run_calibrate_eval_only

    (tmp_path / "logs").mkdir()
    log_dir = tmp_path / "logs"

    class FakeProcess:
        def __init__(self):
            self.returncode = 0
            self.pid = 4242
            self._poll_results = [None, 0]

        def poll(self):
            if self._poll_results:
                return self._poll_results.pop(0)
            return self.returncode

    callback_called = []

    def on_started(pid):
        callback_called.append(pid)

    with patch(
        "annotation_eval_runner.subprocess.Popen", return_value=FakeProcess()
    ), patch("annotation_eval_runner.time.sleep"):
        rc, _, _ = _run_calibrate_eval_only(
            ["calibrate-agent"],
            cwd=tmp_path,
            log_dir=log_dir,
            on_started=on_started,
            heartbeat_seconds=0,
        )
    assert rc == 0
    assert callback_called == [4242]


def test_run_calibrate_eval_only_timeout(tmp_path):
    """The polling watchdog raises AnnotationEvalTimeoutError when updated_at
    is stale."""
    from annotation_eval_runner import (
        _run_calibrate_eval_only,
        AnnotationEvalTimeoutError,
    )

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    class FakeProcess:
        def __init__(self):
            self.returncode = 0
            self.pid = 4242

        # Always say "still running" so the polling loop continues until
        # the timeout watchdog fires.
        def poll(self):
            return None

        def wait(self, timeout=None):
            return None

    with patch(
        "annotation_eval_runner.subprocess.Popen", return_value=FakeProcess()
    ), patch("annotation_eval_runner.time.sleep"), patch(
        "annotation_eval_runner.get_job",
        return_value={"updated_at": "2000-01-01 00:00:00"},
    ), patch(
        "annotation_eval_runner.kill_process_group"
    ), patch(
        "annotation_eval_runner.is_job_timed_out", return_value=True
    ):
        with pytest.raises(AnnotationEvalTimeoutError):
            _run_calibrate_eval_only(
                ["calibrate-agent"],
                cwd=tmp_path,
                log_dir=log_dir,
                job_uuid="j-1",
                heartbeat_seconds=0,
            )


def test_run_calibrate_eval_only_on_started_raises(tmp_path):
    """A failing on_started callback should be logged but not crash."""
    from annotation_eval_runner import _run_calibrate_eval_only

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    class FakeProcess:
        def __init__(self):
            self.returncode = 0
            self.pid = 1

        def poll(self):
            return 0

    def on_started_fail(pid):
        raise RuntimeError("boom")

    with patch(
        "annotation_eval_runner.subprocess.Popen", return_value=FakeProcess()
    ), patch("annotation_eval_runner.time.sleep"):
        rc, _, _ = _run_calibrate_eval_only(
            ["calibrate-agent"],
            cwd=tmp_path,
            log_dir=log_dir,
            on_started=on_started_fail,
            heartbeat_seconds=0,
        )
    assert rc == 0
