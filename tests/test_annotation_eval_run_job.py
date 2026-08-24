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


def test_run_job_failure_uploads_before_temp_dir_is_removed():
    """CLI fails → partial artifacts upload while the temp dir still exists."""
    from annotation_eval_runner import _run_job

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
            self.pid = 7
            self._poll_results = [None, 1]

        def poll(self):
            if self._poll_results:
                return self._poll_results.pop(0)
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    seen = {}

    def fake_upload(output_dir, task_uuid, job_uuid):
        seen["exists"] = output_dir is not None and output_dir.exists()
        return "some/prefix"

    with patch(
        "annotation_eval_runner.get_annotation_task", return_value={"type": "stt"}
    ), patch(
        "annotation_eval_runner.get_eval_job_items", return_value=items
    ), patch(
        "annotation_eval_runner.subprocess.Popen", return_value=FakeProcess()
    ), patch(
        "annotation_eval_runner.update_job"
    ) as mock_update, patch(
        "annotation_eval_runner.try_start_queued_job"
    ), patch(
        "annotation_eval_runner.time.sleep"
    ), patch(
        "annotation_eval_runner.get_job",
        return_value={"updated_at": "2099-01-01 00:00:00"},
    ), patch(
        "annotation_eval_runner._persist_pgid"
    ), patch(
        "annotation_eval_runner._try_upload_partial_outputs", side_effect=fake_upload
    ):
        _run_job("j-1", "task", "u-1", [_resolved()], item_ids=None)

    assert seen.get("exists") is True
    final = mock_update.call_args
    assert final.kwargs["details"]["s3_prefix"] == "some/prefix"


class _FinishedProcess:
    """Popen stand-in that has already exited cleanly."""

    def __init__(self, returncode=0):
        self.returncode = returncode
        self.pid = 1

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def _run_job_with(returncode, parsed_rows):
    """Drive _run_job over a stubbed subprocess + parser. Returns the patched
    (clear, create) mocks plus the ordered call log."""
    from annotation_eval_runner import _run_job

    items = [{"uuid": "i1", "payload": {
        "predicted_transcript": "p", "reference_transcript": "r"}}]
    order = []
    clear = MagicMock(side_effect=lambda *a: order.append("clear") or 1)
    create = MagicMock(side_effect=lambda *a: order.append("create"))

    with patch(
        "annotation_eval_runner.get_annotation_task", return_value={"type": "stt"}
    ), patch(
        "annotation_eval_runner.get_eval_job_items", return_value=items
    ), patch(
        "annotation_eval_runner.subprocess.Popen",
        return_value=_FinishedProcess(returncode),
    ), patch(
        "annotation_eval_runner.parse_results_for_task_type",
        return_value=parsed_rows,
    ), patch(
        "annotation_eval_runner.clear_evaluator_runs_for_job", clear
    ), patch(
        "annotation_eval_runner.create_evaluator_runs", create
    ), patch(
        "annotation_eval_runner.update_job"
    ), patch(
        "annotation_eval_runner.try_start_queued_job"
    ), patch(
        "annotation_eval_runner._persist_pgid"
    ), patch(
        "annotation_eval_runner.capture_exception_to_sentry"
    ), patch(
        "annotation_eval_runner._try_upload_partial_outputs", return_value=None
    ), patch(
        "annotation_eval_runner.get_s3_client", return_value=MagicMock()
    ), patch(
        "annotation_eval_runner.get_s3_output_config", return_value="bucket"
    ), patch(
        "annotation_eval_runner.upload_file_to_s3"
    ):
        _run_job("j-1", "task", "u-1", [_resolved()], item_ids=None)
    return clear, create, order


def _row(item_id="i1"):
    return {
        "job_id": "j-1",
        "item_id": item_id,
        "evaluator_id": "ev-1",
        "evaluator_version_id": "ver-1",
        "value": {"value": True},
        "status": "completed",
    }


def test_run_job_rewrites_rows_from_the_finished_files():
    """Anything stored while calibrate was still writing is provisional — a
    successful run replaces it with what the complete files say."""
    clear, create, order = _run_job_with(0, [_row()])
    assert order == ["clear", "create"]
    assert create.call_args[0][0] == [_row()]


def test_run_job_discards_in_flight_rows_when_calibrate_fails():
    """A failed run contributes nothing to the task's scores."""
    clear, create, _ = _run_job_with(1, [_row()])
    clear.assert_called_once_with("j-1")
    create.assert_not_called()


def test_run_job_no_rows_skips_create():
    """A successful run that parses zero rows still clears the job's prior
    runs, but has nothing to insert."""
    clear, create, order = _run_job_with(0, [])
    assert order == ["clear"]
    create.assert_not_called()
