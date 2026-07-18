"""Unit tests for voice-simulation helpers in routers/simulations.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch



def test_is_simulation_complete(tmp_path):
    from routers.simulations import _is_simulation_complete

    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    assert _is_simulation_complete(sim_dir) is False
    (sim_dir / "evaluation_results.csv").write_text("name,value\n")
    assert _is_simulation_complete(sim_dir) is True


def test_is_simulation_started(tmp_path):
    from routers.simulations import _is_simulation_started

    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    assert _is_simulation_started(sim_dir) is False
    (sim_dir / "config.json").write_text("{}")
    assert _is_simulation_started(sim_dir) is True


def test_upload_audio_and_generate_urls_no_audios(tmp_path):
    from routers.simulations import _upload_audio_and_generate_urls

    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    s3_mock = MagicMock()
    with patch("routers.simulations.get_s3_client", return_value=s3_mock), patch(
        "routers.simulations.upload_file_to_s3"
    ), patch(
        "routers.simulations.generate_presigned_download_url",
        return_value="https://signed",
    ):
        out = _upload_audio_and_generate_urls(
            sim_dir, tmp_path, "bucket", "prefix", set()
        )
    audios_s3_path, _conversation_wav_s3_key, audio_urls, _conversation_wav_url = out
    assert audios_s3_path is None
    assert audio_urls == []


def test_upload_audio_and_generate_urls_with_audios(tmp_path):
    from routers.simulations import _upload_audio_and_generate_urls

    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    audios = sim_dir / "audios"
    audios.mkdir()
    (audios / "1_bot.wav").write_bytes(b"x")
    (audios / "1_user.wav").write_bytes(b"y")
    (audios / "non_audio.txt").write_text("ignored")
    # Also conversation.wav
    (sim_dir / "conversation.wav").write_bytes(b"z")

    s3_mock = MagicMock()
    with patch("routers.simulations.get_s3_client", return_value=s3_mock), patch(
        "routers.simulations.upload_file_to_s3"
    ), patch(
        "routers.simulations.generate_presigned_download_url",
        return_value="https://signed",
    ):
        out = _upload_audio_and_generate_urls(
            sim_dir, tmp_path, "bucket", "prefix", set()
        )
    audios_s3_path, conversation_wav_s3_key, audio_urls, conversation_wav_url = out
    assert audios_s3_path
    assert conversation_wav_s3_key
    assert len(audio_urls) == 2
    assert conversation_wav_url == "https://signed"


def test_parse_voice_simulation_in_progress(tmp_path):
    from routers.simulations import _parse_voice_simulation_in_progress

    # Non-existent directory
    assert (
        _parse_voice_simulation_in_progress(tmp_path / "missing") is None
    )

    sim_dir = tmp_path / "simulation_persona_1_scenario_1"
    sim_dir.mkdir()
    # Without any config or transcript, returns None
    assert _parse_voice_simulation_in_progress(sim_dir) is None

    # With config.json + transcript
    (sim_dir / "config.json").write_text(
        json.dumps(
            {
                "persona": {"label": "p", "name": "Alice"},
                "scenario": {"name": "s"},
            }
        )
    )
    (sim_dir / "transcript.json").write_text(
        json.dumps([{"role": "user", "content": "hi"}])
    )
    result = _parse_voice_simulation_in_progress(sim_dir)
    assert result is not None
    assert result["evaluation_results"] is None
    assert result["transcript"]


def test_parse_voice_simulation_in_progress_uses_personas_list_fallback(tmp_path):
    """When config.json is missing, fall back to personas_list/scenarios_list by index."""
    from routers.simulations import _parse_voice_simulation_in_progress

    sim_dir = tmp_path / "simulation_persona_1_scenario_2"
    sim_dir.mkdir()
    (sim_dir / "transcript.json").write_text(json.dumps([]))
    personas = [{"name": "Alice"}, {"name": "Bob"}]
    scenarios = [{"name": "Sc1"}, {"name": "Sc2"}]
    result = _parse_voice_simulation_in_progress(
        sim_dir, personas_list=personas, scenarios_list=scenarios
    )
    assert result["persona"]["name"] == "Alice"
    assert result["scenario"]["name"] == "Sc2"
