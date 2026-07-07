"""Unit tests for the pure helper functions in routers/simulations.py."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


def test_should_regenerate_presigned_urls_no_timestamp():
    from routers.simulations import _should_regenerate_presigned_urls

    assert _should_regenerate_presigned_urls(None) is True


def test_should_regenerate_presigned_urls_recent():
    from routers.simulations import _should_regenerate_presigned_urls

    now = datetime.utcnow().isoformat()
    # Just generated → no need to regenerate
    assert _should_regenerate_presigned_urls(now) is False


def test_should_regenerate_presigned_urls_old():
    from routers.simulations import _should_regenerate_presigned_urls

    old = (datetime.utcnow() - timedelta(hours=2)).isoformat()
    assert _should_regenerate_presigned_urls(old) is True


def test_should_regenerate_presigned_urls_malformed():
    from routers.simulations import _should_regenerate_presigned_urls

    assert _should_regenerate_presigned_urls("not-a-timestamp") is True


def test_get_audio_urls_from_s3_key_no_files():
    """An S3 prefix with no matching audio files → empty list."""
    from routers.simulations import _get_audio_urls_from_s3_key

    s3_mock = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = iter(
        [{"Contents": [{"Key": "ignored/dir/"}]}]
    )
    s3_mock.get_paginator.return_value = paginator
    with patch("routers.simulations.get_s3_client", return_value=s3_mock):
        urls = _get_audio_urls_from_s3_key("prefix", "bucket")
    assert urls == []


def test_get_audio_urls_from_s3_key_sorts_by_transcript():
    """Audio files grouped into exchanges; the transcript's user-first signal
    should pull user audio before bot audio for that exchange."""
    from routers.simulations import _get_audio_urls_from_s3_key

    s3_mock = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = iter(
        [
            {
                "Contents": [
                    {"Key": "prefix/1_bot.wav"},
                    {"Key": "prefix/1_user.wav"},
                    {"Key": "prefix/2_bot.wav"},
                    {"Key": "prefix/2_user.wav"},
                ]
            }
        ]
    )
    s3_mock.get_paginator.return_value = paginator
    with patch("routers.simulations.get_s3_client", return_value=s3_mock), patch(
        "routers.simulations.generate_presigned_download_url",
        side_effect=lambda key, bucket: f"https://signed/{key}",
    ):
        urls = _get_audio_urls_from_s3_key(
            "prefix",
            "bucket",
            transcript=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hi back"},
                {"role": "user", "content": "yo"},
                {"role": "assistant", "content": "yo back"},
            ],
        )
    # First exchange begins with user → user file first
    assert urls[0].endswith("1_user.wav")


def test_get_audio_urls_from_s3_key_exception():
    """If get_s3_client raises, return empty list."""
    from routers.simulations import _get_audio_urls_from_s3_key

    with patch(
        "routers.simulations.get_s3_client", side_effect=RuntimeError("nope")
    ):
        assert _get_audio_urls_from_s3_key("prefix", "bucket") == []


def test_get_audio_urls_from_s3_key_presigned_failure_falls_back_to_s3():
    """If presigned URL generation fails, the s3:// path is used as a fallback."""
    from routers.simulations import _get_audio_urls_from_s3_key

    s3_mock = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = iter(
        [{"Contents": [{"Key": "prefix/1_bot.wav"}]}]
    )
    s3_mock.get_paginator.return_value = paginator
    with patch("routers.simulations.get_s3_client", return_value=s3_mock), patch(
        "routers.simulations.generate_presigned_download_url", return_value=None
    ):
        urls = _get_audio_urls_from_s3_key("prefix", "bucket")
    assert urls[0].startswith("s3://")


def test_is_job_aborted_helper():
    """_is_job_aborted reads from the simulation job's details.aborted."""
    from routers.simulations import _is_job_aborted

    with patch(
        "routers.simulations.get_simulation_job",
        return_value={"details": {"aborted": True}},
    ):
        assert _is_job_aborted("j-1") is True
    with patch(
        "routers.simulations.get_simulation_job", return_value={"details": {}}
    ):
        assert _is_job_aborted("j-1") is False
    with patch(
        "routers.simulations.get_simulation_job", return_value=None
    ):
        assert _is_job_aborted("j-1") is False


def test_snapshot_evaluators_for_job_details():
    from routers.simulations import _snapshot_evaluators_for_job_details

    out = _snapshot_evaluators_for_job_details(
        [
            {"uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479", "name": "Safety"},
            {"name": "no-uuid"},  # skipped
            {"uuid": "ev-2"},  # missing name → empty
        ]
    )
    assert out == [
        {"uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479", "name": "Safety"},
        {"uuid": "ev-2", "name": ""},
    ]


def test_apply_simulation_job_evaluator_enrichment():
    from routers.simulations import apply_simulation_job_evaluator_enrichment

    details = {"evaluators": [{"uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479", "name": "Safety"}]}
    sim_results = [
        {
            "evaluation_results": [
                {"evaluator_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479", "value": 0.9},
            ]
        }
    ]
    with patch(
        "routers.simulations.get_evaluator",
        return_value={"uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479", "name": "Safety", "description": "d"},
    ):
        evaluators_out, sim_out = apply_simulation_job_evaluator_enrichment(
            details, sim_results
        )
    assert evaluators_out and evaluators_out[0].evaluator_uuid == "f47ac10b-58cc-4372-a567-0e02b2c3d479"
    assert sim_out[0]["evaluation_results"][0]["evaluator_uuid"] == "f47ac10b-58cc-4372-a567-0e02b2c3d479"


def test_apply_simulation_job_evaluator_enrichment_no_snaps():
    from routers.simulations import apply_simulation_job_evaluator_enrichment

    evaluators_out, sim_out = apply_simulation_job_evaluator_enrichment(
        {"evaluators": []}, []
    )
    assert evaluators_out is None
