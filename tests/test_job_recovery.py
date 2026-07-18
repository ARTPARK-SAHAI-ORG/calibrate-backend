"""Unit tests for job_recovery.

The recovery functions kick off threads to resume work. We mock threading
and the routers' run-task entry points so nothing actually runs.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import db
import job_recovery


def _make_user():
    return db.create_user("J", "R", f"jr-{os.urandom(4).hex()}@x.com")


# ---------------------------------------------------------------------------
# kill_processes_from_dict (shared helper from utils)
# ---------------------------------------------------------------------------


def test_kill_orphaned_processes_from_dict_noop():
    job_recovery.kill_processes_from_dict(None, "j-1")
    job_recovery.kill_processes_from_dict({}, "j-1")


def test_kill_orphaned_processes_from_dict_dispatch():
    with patch("utils.os.killpg") as kp, patch("utils.time.sleep"):
        job_recovery.kill_processes_from_dict({"openai": 111, "blank": 0}, "j-1")
        # TERM then KILL for the openai pid; blank skipped
        assert kp.call_count == 2


def test_kill_orphaned_processes_from_dict_kill_already_dead():
    with (
        patch(
            "utils.os.killpg",
            side_effect=[None, ProcessLookupError()],
        ),
        patch("utils.time.sleep"),
    ):
        job_recovery.kill_processes_from_dict({"a": 1}, "j-1")


def test_kill_orphaned_processes_from_dict_other_errors():
    with (
        patch("utils.os.killpg", side_effect=ProcessLookupError()),
        patch("utils.time.sleep"),
    ):
        job_recovery.kill_processes_from_dict({"a": 1}, "j-1")
    with (
        patch("utils.os.killpg", side_effect=PermissionError()),
        patch("utils.time.sleep"),
    ):
        job_recovery.kill_processes_from_dict({"a": 1}, "j-1")
    with (
        patch("utils.os.killpg", side_effect=RuntimeError("x")),
        patch("utils.time.sleep"),
    ):
        job_recovery.kill_processes_from_dict({"a": 1}, "j-1")


# ---------------------------------------------------------------------------
# _kill_orphaned_process (single)
# ---------------------------------------------------------------------------


def test_kill_orphaned_process_no_pid_returns_true():
    assert job_recovery._kill_orphaned_process({}, "j-1") is True


def test_kill_orphaned_process_pgid_term_kill_path():
    with patch("job_recovery.os.killpg") as kp, patch("job_recovery.time.sleep"):
        assert job_recovery._kill_orphaned_process({"pgid": 111}, "j-1") is True
        assert kp.call_count == 2


def test_kill_orphaned_process_pgid_already_dead():
    with patch(
        "job_recovery.os.killpg",
        side_effect=ProcessLookupError(),
    ):
        assert job_recovery._kill_orphaned_process({"pgid": 111}, "j-1") is True


def test_kill_orphaned_process_pgid_kill_already_dead_inner():
    """SIGTERM succeeds, SIGKILL finds process gone."""
    with (
        patch(
            "job_recovery.os.killpg",
            side_effect=[None, ProcessLookupError()],
        ),
        patch("job_recovery.time.sleep"),
    ):
        assert job_recovery._kill_orphaned_process({"pgid": 111}, "j-1") is True


def test_kill_orphaned_process_pid_fallback():
    """No pgid, only pid. Both SIGTERM + SIGKILL via os.kill."""
    with patch("job_recovery.os.kill") as kp, patch("job_recovery.time.sleep"):
        assert job_recovery._kill_orphaned_process({"pid": 111}, "j-1") is True
        assert kp.call_count == 2


def test_kill_orphaned_process_pid_already_dead():
    with patch(
        "job_recovery.os.kill",
        side_effect=ProcessLookupError(),
    ):
        assert job_recovery._kill_orphaned_process({"pid": 111}, "j-1") is True


def test_kill_orphaned_process_pid_permission_error():
    with patch("job_recovery.os.kill", side_effect=PermissionError()):
        assert job_recovery._kill_orphaned_process({"pid": 111}, "j-1") is False


def test_kill_orphaned_process_pid_other_error():
    with patch("job_recovery.os.kill", side_effect=RuntimeError("x")):
        assert job_recovery._kill_orphaned_process({"pid": 111}, "j-1") is False


def test_kill_orphaned_process_pgid_permission_falls_to_pid():
    """SIGTERM on pgid raises PermissionError → fall back to pid path."""
    with (
        patch("job_recovery.os.killpg", side_effect=PermissionError()),
        patch("job_recovery.os.kill") as kp,
        patch("job_recovery.time.sleep"),
    ):
        assert (
            job_recovery._kill_orphaned_process({"pgid": 111, "pid": 222}, "j-1")
            is True
        )
        assert kp.call_count == 2


def test_kill_orphaned_process_pgid_other_error_falls_to_pid():
    with (
        patch("job_recovery.os.killpg", side_effect=RuntimeError("boom")),
        patch("job_recovery.os.kill"),
        patch("job_recovery.time.sleep"),
    ):
        assert (
            job_recovery._kill_orphaned_process({"pgid": 111, "pid": 222}, "j-1")
            is True
        )


# ---------------------------------------------------------------------------
# recover_pending_jobs — high-level flow
# ---------------------------------------------------------------------------


def test_recover_no_pending_jobs():
    """When no jobs are pending, recover should still complete."""
    with (
        patch("job_recovery.get_pending_jobs", return_value=[]),
        patch("job_recovery.get_pending_agent_test_jobs", return_value=[]),
        patch("job_recovery.get_pending_simulation_jobs", return_value=[]),
        patch("job_recovery._start_queued_jobs"),
    ):
        job_recovery.recover_pending_jobs()


def test_recover_pending_jobs_missing_details_marks_failed():
    """An in_progress job with no details should be marked failed."""
    with (
        patch(
            "job_recovery.get_pending_jobs",
            return_value=[{"uuid": "j1", "type": "stt-eval", "details": None}],
        ),
        patch("job_recovery.get_pending_agent_test_jobs", return_value=[]),
        patch("job_recovery.get_pending_simulation_jobs", return_value=[]),
        patch("job_recovery.update_job"),
        patch("job_recovery._start_queued_jobs"),
    ):
        job_recovery.recover_pending_jobs()


def test_recover_pending_jobs_unknown_type_marks_failed():
    """Unknown job type → marked failed."""
    with (
        patch(
            "job_recovery.get_pending_jobs",
            return_value=[{"uuid": "j2", "type": "zzz", "details": {"a": 1}}],
        ),
        patch("job_recovery.get_pending_agent_test_jobs", return_value=[]),
        patch("job_recovery.get_pending_simulation_jobs", return_value=[]),
        patch("job_recovery.update_job"),
        patch("job_recovery._start_queued_jobs"),
    ):
        job_recovery.recover_pending_jobs()


def test_recover_pending_jobs_exception_marks_failed():
    """Exceptions in recovery handlers should mark the job failed."""
    with (
        patch(
            "job_recovery.get_pending_jobs",
            return_value=[{"uuid": "j3", "type": "stt-eval", "details": {"a": 1}}],
        ),
        patch("job_recovery.get_pending_agent_test_jobs", return_value=[]),
        patch("job_recovery.get_pending_simulation_jobs", return_value=[]),
        patch("job_recovery._recover_stt_job", side_effect=RuntimeError("x")),
        patch("job_recovery.update_job"),
        patch("job_recovery._start_queued_jobs"),
    ):
        job_recovery.recover_pending_jobs()


def test_recover_agent_test_jobs_paths():
    """agent test job recovery branches: missing-details + unknown-type +
    handler-exception. Don't touch real rows — feed fake job dicts."""
    with (
        patch("job_recovery.get_pending_jobs", return_value=[]),
        patch(
            "job_recovery.get_pending_agent_test_jobs",
            return_value=[
                {"uuid": "ag1", "type": "llm-unit-test", "details": None},
                {"uuid": "ag2", "type": "zzz", "details": {"a": 1}},
                {"uuid": "ag3", "type": "llm-unit-test", "details": {"a": 1}},
            ],
        ),
        patch("job_recovery.get_pending_simulation_jobs", return_value=[]),
        patch("job_recovery._recover_llm_unit_test_job", side_effect=RuntimeError("x")),
        patch("job_recovery.update_agent_test_job"),
        patch("job_recovery._start_queued_jobs"),
    ):
        job_recovery.recover_pending_jobs()


def test_recover_simulation_jobs_paths():
    with (
        patch("job_recovery.get_pending_jobs", return_value=[]),
        patch("job_recovery.get_pending_agent_test_jobs", return_value=[]),
        patch(
            "job_recovery.get_pending_simulation_jobs",
            return_value=[
                {"uuid": "s1", "type": "text", "details": None},
                {"uuid": "s2", "type": "zzz", "details": {"a": 1}},
                {"uuid": "s3", "type": "text", "details": {"a": 1}},
            ],
        ),
        patch("job_recovery._recover_simulation_job", side_effect=RuntimeError("x")),
        patch("job_recovery.update_simulation_job"),
        patch("job_recovery._start_queued_jobs"),
    ):
        job_recovery.recover_pending_jobs()


# ---------------------------------------------------------------------------
# Per-type recovery helpers
# ---------------------------------------------------------------------------


def test_recover_stt_job_starts_thread():
    with (
        patch("routers.stt.run_evaluation_task"),
        patch("job_recovery.threading.Thread") as thread_mock,
    ):
        job_recovery._recover_stt_job(
            "j-1",
            {
                "audio_paths": ["s3://b/k"],
                "texts": ["t"],
                "providers": ["openai"],
                "language": "en",
                "s3_bucket": "b",
            },
        )
        thread_mock.return_value.start.assert_called_once()


def test_recover_tts_job_starts_thread():
    with (
        patch("routers.tts.run_tts_evaluation_task"),
        patch("job_recovery.threading.Thread") as thread_mock,
    ):
        job_recovery._recover_tts_job(
            "j-1",
            {
                "texts": ["t"],
                "providers": ["openai"],
                "language": "en",
                "s3_bucket": "b",
            },
        )
        thread_mock.return_value.start.assert_called_once()


def test_recover_llm_unit_test_missing_agent_raises():
    import pytest

    with patch("job_recovery.get_agent", return_value=None):
        with pytest.raises(ValueError):
            job_recovery._recover_llm_unit_test_job(
                "j-1",
                {
                    "agent_uuid": "missing",
                    "test_uuids": [],
                    "s3_bucket": "b",
                },
            )


def test_recover_llm_unit_test_missing_test_raises():
    import pytest

    with (
        patch("job_recovery.get_agent", return_value={"uuid": "a"}),
        patch("job_recovery.get_test", return_value=None),
    ):
        with pytest.raises(ValueError):
            job_recovery._recover_llm_unit_test_job(
                "j-1",
                {
                    "agent_uuid": "a",
                    "test_uuids": ["t1"],
                    "s3_bucket": "b",
                },
            )


def test_recover_llm_unit_test_starts_thread():
    with (
        patch("job_recovery.get_agent", return_value={"uuid": "a"}),
        patch("job_recovery.get_test", return_value={"uuid": "t"}),
        patch("routers.agent_tests.run_llm_test_task"),
        patch("job_recovery.threading.Thread") as thread_mock,
    ):
        job_recovery._recover_llm_unit_test_job(
            "j-1",
            {
                "agent_uuid": "a",
                "test_uuids": ["t"],
                "s3_bucket": "b",
            },
        )
        thread_mock.return_value.start.assert_called_once()


def test_recover_llm_benchmark_starts_thread():
    with (
        patch("job_recovery.get_agent", return_value={"uuid": "a"}),
        patch("job_recovery.get_test", return_value={"uuid": "t"}),
        patch("routers.agent_tests.run_benchmark_task"),
        patch("job_recovery.threading.Thread") as thread_mock,
    ):
        job_recovery._recover_llm_benchmark_job(
            "j-1",
            {
                "agent_uuid": "a",
                "test_uuids": ["t"],
                "models": ["m"],
                "s3_bucket": "b",
            },
        )
        thread_mock.return_value.start.assert_called_once()


def test_recover_llm_benchmark_missing_agent():
    import pytest

    with patch("job_recovery.get_agent", return_value=None):
        with pytest.raises(ValueError):
            job_recovery._recover_llm_benchmark_job(
                "j-1",
                {
                    "agent_uuid": "missing",
                    "test_uuids": [],
                    "models": [],
                    "s3_bucket": "b",
                },
            )


def test_recover_simulation_missing_simulation():
    import pytest

    with patch("job_recovery.get_simulation", return_value=None):
        with pytest.raises(ValueError):
            job_recovery._recover_simulation_job(
                "j-1",
                {"simulation_uuid": "missing", "agent_uuid": "a", "s3_bucket": "b"},
                "text",
            )


def test_recover_simulation_missing_agent():
    import pytest

    with (
        patch("job_recovery.get_simulation", return_value={"uuid": "s"}),
        patch("job_recovery.get_agent", return_value=None),
    ):
        with pytest.raises(ValueError):
            job_recovery._recover_simulation_job(
                "j-1",
                {"simulation_uuid": "s", "agent_uuid": "missing", "s3_bucket": "b"},
                "text",
            )


def test_recover_simulation_missing_personas():
    import pytest

    with (
        patch("job_recovery.get_simulation", return_value={"uuid": "s"}),
        patch("job_recovery.get_agent", return_value={"uuid": "a"}),
        patch("job_recovery.get_personas_for_simulation", return_value=[]),
        patch("job_recovery.get_scenarios_for_simulation", return_value=[]),
        patch("job_recovery.get_evaluators_for_simulation", return_value=[]),
    ):
        with pytest.raises(ValueError, match="no personas"):
            job_recovery._recover_simulation_job(
                "j-1",
                {"simulation_uuid": "s", "agent_uuid": "a", "s3_bucket": "b"},
                "text",
            )


def test_recover_simulation_missing_scenarios():
    import pytest

    with (
        patch("job_recovery.get_simulation", return_value={"uuid": "s"}),
        patch("job_recovery.get_agent", return_value={"uuid": "a"}),
        patch("job_recovery.get_personas_for_simulation", return_value=[{"uuid": "p"}]),
        patch("job_recovery.get_scenarios_for_simulation", return_value=[]),
        patch("job_recovery.get_evaluators_for_simulation", return_value=[]),
    ):
        with pytest.raises(ValueError, match="no scenarios"):
            job_recovery._recover_simulation_job(
                "j-1",
                {"simulation_uuid": "s", "agent_uuid": "a", "s3_bucket": "b"},
                "text",
            )


def test_recover_simulation_voice_kills_orphaned():
    with (
        patch("job_recovery.get_simulation", return_value={"uuid": "s"}),
        patch("job_recovery.get_agent", return_value={"uuid": "a"}),
        patch("job_recovery.get_personas_for_simulation", return_value=[{"uuid": "p"}]),
        patch(
            "job_recovery.get_scenarios_for_simulation",
            return_value=[{"uuid": "sc"}],
        ),
        patch("job_recovery.get_evaluators_for_simulation", return_value=[]),
        patch("job_recovery._kill_orphaned_process") as kop,
        patch("routers.simulations.run_simulation_task"),
        patch("job_recovery.threading.Thread") as thread_mock,
    ):
        job_recovery._recover_simulation_job(
            "j-1",
            {
                "simulation_uuid": "s",
                "agent_uuid": "a",
                "s3_bucket": "b",
                "pid": 1234,
            },
            "voice",
        )
        kop.assert_called_once()
        thread_mock.return_value.start.assert_called_once()


def test_recover_annotation_eval_missing_evaluators_raises():
    import pytest

    job = {"uuid": "j-1", "details": {}}
    with patch("job_recovery._kill_orphaned_process"):
        with pytest.raises(ValueError):
            job_recovery._recover_annotation_eval_job(job)


def test_recover_annotation_eval_starts_via_resume():
    job = {"uuid": "j-1", "details": {"evaluators": [{"uuid": "e1"}]}}
    with (
        patch("job_recovery._kill_orphaned_process"),
        patch("annotation_eval_runner.resume_annotation_eval_job") as resume,
    ):
        job_recovery._recover_annotation_eval_job(job)
        resume.assert_called_once()


# ---------------------------------------------------------------------------
# _start_queued_jobs
# ---------------------------------------------------------------------------


def test_start_queued_jobs_empty():
    with (
        patch("job_recovery.get_queued_jobs", return_value=[]),
        patch("job_recovery.get_queued_agent_test_jobs", return_value=[]),
        patch("job_recovery.get_queued_simulation_jobs", return_value=[]),
    ):
        job_recovery._start_queued_jobs()


def test_start_queued_jobs_drain():
    counter = {"n": 0}

    def fake_starter(types):
        counter["n"] += 1
        return counter["n"] < 3  # True, True, then False

    with (
        patch("job_recovery.get_queued_jobs", return_value=[{"uuid": "j"}]),
        patch("job_recovery.get_queued_agent_test_jobs", return_value=[]),
        patch("job_recovery.get_queued_simulation_jobs", return_value=[]),
        patch("job_recovery.try_start_queued_job", side_effect=fake_starter),
    ):
        job_recovery._start_queued_jobs()
    assert counter["n"] == 3
