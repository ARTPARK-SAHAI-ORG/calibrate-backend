"""Extra tests for src/main.py — provider-status, OpenRouter, and sentry-debug."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app():
    import main as main_mod

    return main_mod.app


@pytest.fixture(scope="module")
def client(app):
    async def idle_provider_status_loop():
        await asyncio.sleep(3600)

    with patch("main.recover_pending_jobs"):
        with patch(
            "main.provider_status_monitor.refresh_loop", idle_provider_status_loop
        ):
            with TestClient(app) as c:
                yield c


@pytest.fixture(autouse=True)
def reset_provider_status_cache():
    import provider_status

    provider_status.provider_status_monitor.clear_cache()
    yield
    provider_status.provider_status_monitor.clear_cache()


# ---------------------------------------------------------------------------
# /provider-status — subprocess mocked
# ---------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def readline(self):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


def _make_fake_process(returncode: int, stdout: bytes, stderr: bytes):
    process = MagicMock()
    process.returncode = returncode
    process.stdout = _FakeStream([stdout] if stdout else [])
    process.stderr = _FakeStream([stderr] if stderr else [])
    process.communicate = AsyncMock(return_value=(stdout, stderr))
    process.wait = AsyncMock(return_value=None)
    process.kill = MagicMock()
    return process


def test_provider_status_all_pass(client):
    import provider_status

    process = _make_fake_process(
        0, json.dumps({"openai": {"status": "pass"}}).encode(), b""
    )
    with patch(
        "provider_status.asyncio.create_subprocess_exec",
        AsyncMock(return_value=process),
    ):
        asyncio.run(provider_status.provider_status_monitor.refresh_cache())
        resp = client.get("/provider-status")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert resp.json()["cached"] is True


def test_provider_status_some_failed(client):
    import provider_status

    process = _make_fake_process(
        0,
        json.dumps(
            {"openai": {"status": "pass"}, "deepgram": {"status": "fail", "error": "x"}}
        ).encode(),
        b"",
    )
    with patch(
        "provider_status.asyncio.create_subprocess_exec",
        AsyncMock(return_value=process),
    ):
        asyncio.run(provider_status.provider_status_monitor.refresh_cache())
        resp = client.get("/provider-status")
    assert resp.status_code == 503
    body = resp.json()
    assert body["success"] is False
    assert body["failed_providers"] == {"deepgram": "x"}


def test_provider_status_excludes_groq_by_default(client):
    import provider_status

    process = _make_fake_process(
        0,
        json.dumps(
            {
                "openai": {"status": "pass"},
                "groq": {"status": "fail", "error": "HTTP 429"},
            }
        ).encode(),
        b"",
    )
    with patch(
        "provider_status.asyncio.create_subprocess_exec",
        AsyncMock(return_value=process),
    ):
        asyncio.run(provider_status.provider_status_monitor.refresh_cache())
        resp = client.get("/provider-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert set(body["all_providers"]) == {"openai"}


def test_provider_status_subprocess_non_zero(client):
    import provider_status

    process = _make_fake_process(1, b"", b"boom")
    with patch(
        "provider_status.asyncio.create_subprocess_exec",
        AsyncMock(return_value=process),
    ):
        asyncio.run(provider_status.provider_status_monitor.refresh_cache())
        resp = client.get("/provider-status")
    assert resp.status_code == 500
    assert resp.json()["message"] == "calibrate status failed: boom"


def test_provider_status_invalid_json(client):
    import provider_status

    process = _make_fake_process(0, b"not json", b"")
    with patch(
        "provider_status.asyncio.create_subprocess_exec",
        AsyncMock(return_value=process),
    ):
        asyncio.run(provider_status.provider_status_monitor.refresh_cache())
        resp = client.get("/provider-status")
    assert resp.status_code == 500
    assert resp.json()["message"] == "Failed to parse calibrate status output"


def test_provider_status_calibrate_not_found(client):
    import provider_status

    with patch(
        "provider_status.asyncio.create_subprocess_exec",
        AsyncMock(side_effect=FileNotFoundError()),
    ):
        asyncio.run(provider_status.provider_status_monitor.refresh_cache())
        resp = client.get("/provider-status")
    assert resp.status_code == 500
    assert resp.json()["message"] == "calibrate-agent CLI not found"


def test_provider_status_timeout(client):
    import provider_status

    process = MagicMock()
    process.returncode = 0
    process.stdout = MagicMock()
    process.stdout.readline = AsyncMock(side_effect=asyncio.TimeoutError())
    process.stderr = _FakeStream([])
    process.wait = AsyncMock(return_value=None)
    process.kill = MagicMock()
    with patch(
        "provider_status.asyncio.create_subprocess_exec",
        AsyncMock(return_value=process),
    ):
        asyncio.run(provider_status.provider_status_monitor.refresh_cache())
        resp = client.get("/provider-status")
    assert resp.status_code == 504


@pytest.mark.parametrize(
    "group_kill_lands", [True, False], ids=["kill-lands", "kill-refused"]
)
def test_cancelled_check_reaps_the_probe_without_stalling(
    monkeypatch, group_kill_lands
):
    """A cancelled check must tear the probe down, promptly, either way.

    Shape matters. `uv run calibrate-agent status` forks the CLI as a grandchild
    that inherits stdout/stderr, and a `wait()` waiter registered while the
    child is still alive only resolves once every pipe disconnects, so killing
    just the direct child leaves the reap waiting on pipes the grandchild still
    holds. We simulate this behavior without needing uv or the network by running
    `sh -c 'sleep 30 & wait'`.

    killpg takes the grandchild down; we parameterize both kill scenarios. When
    killpg does not land, the transport close ahead of the wait keeps the reap
    fast, avoiding a timeout.
    """
    monkeypatch.delenv("FAKE_AI_PROVIDERS", raising=False)
    import os
    import signal

    import provider_status

    spawned: list[asyncio.subprocess.Process] = []
    # Bind before patching: we'll patch attributes of the asyncio and os
    # modules, so calling them by name below would recurse into the mock.
    real_create = asyncio.create_subprocess_exec
    real_killpg = os.killpg

    async def spawn_tree(*_args, **kwargs):
        # The reap kills the process GROUP, which is only safe because the
        # production spawn detaches (with start_new_session=True), otherwise
        # killpg would signal pytest's own group.
        assert kwargs.get("start_new_session") is True
        process = await real_create(
            "sh",
            "-c",
            "sleep 30 & wait",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        spawned.append(process)
        return process

    def maybe_kill(pgid, sig):
        if not group_kill_lands:
            raise PermissionError("killpg refused")
        return real_killpg(pgid, sig)

    async def body():
        monitor = provider_status.ProviderStatusMonitor(
            refresh_interval_seconds=60,
            cache_max_age_seconds=60,
            check_timeout_seconds=60,
        )
        with patch(
            "provider_status.asyncio.create_subprocess_exec", side_effect=spawn_tree
        ):
            task = asyncio.create_task(monitor.run_check())
            for _ in range(500):
                if spawned:
                    break
                await asyncio.sleep(0.01)
            assert spawned, "status subprocess never started"
            process = spawned[0]
            assert os.getpgid(process.pid) == process.pid, "probe was not detached"

            # Patched only for the reap, so the spawn above still detaches.
            with patch("provider_status.os.killpg", side_effect=maybe_kill):
                task.cancel()
                started = time.monotonic()
                # Generous outer bound so a regression fails this test instead
                # of wedging CI; the elapsed assertion below is the real one.
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=30)
                return process.pid, time.monotonic() - started

    pgid, elapsed = asyncio.run(body())

    if group_kill_lands:
        # Verify that the grandchild went too, not just `sh`.
        for _ in range(20):
            try:
                # Checks if any process in the group is still alive.
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            pytest.fail(f"process group {pgid} survived the reap")
    else:
        # The tree outlives a kill that never lands; this test cleans it up.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    assert elapsed < provider_status._REAP_TIMEOUT_SECONDS / 2, (
        f"reap took {elapsed:.2f}s; the transport close before wait() is missing"
    )


def test_provider_status_not_checked_yet(client):
    resp = client.get("/provider-status")
    assert resp.status_code == 503
    assert resp.json()["message"] == "Provider status has not been checked yet"


def test_provider_status_force_refresh(client):
    import provider_status

    process = _make_fake_process(
        0, json.dumps({"openai": {"status": "pass"}}).encode(), b""
    )
    with patch(
        "provider_status.asyncio.create_subprocess_exec",
        AsyncMock(return_value=process),
    ):
        resp = client.get("/provider-status", params={"refresh": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["refreshed"] is True
    assert body["cached"] is True


def test_provider_status_force_refresh_bypasses_stale_cache(client):
    import provider_status
    from datetime import datetime, timedelta

    stale_checked_at = (datetime.utcnow() - timedelta(hours=2)).isoformat() + "Z"
    provider_status.provider_status_monitor._cache = {
        "checked_at": stale_checked_at,
        "providers": {"openai": {"status": "pass"}},
        "error_status_code": None,
        "error_detail": None,
    }

    process = _make_fake_process(
        0,
        json.dumps({"openai": {"status": "pass"}, "deepgram": {"status": "pass"}}).encode(),
        b"",
    )
    with patch(
        "provider_status.asyncio.create_subprocess_exec",
        AsyncMock(return_value=process),
    ):
        resp = client.get("/provider-status", params={"refresh": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refreshed"] is True
    assert body["stale"] is False
    assert set(body["all_providers"]) == {"openai", "deepgram"}


def test_provider_status_refresh_ignored_on_head(client):
    import provider_status

    with patch.object(
        provider_status.provider_status_monitor,
        "refresh_cache",
        AsyncMock(),
    ) as refresh_mock:
        resp = client.head("/provider-status", params={"refresh": True})
    assert resp.status_code == 503
    refresh_mock.assert_not_called()


def test_provider_status_parses_progress_event_output(app):
    import provider_status

    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "progress",
                    "provider": "openai",
                    "stage": "input_sent",
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "provider": "openai",
                    "result": {"status": "pass", "error": None},
                }
            ),
        ]
    )

    assert provider_status.parse_provider_status_stdout(stdout) == {
        "openai": {"status": "pass", "error": None}
    }


def test_provider_status_logs_streamed_output(client, caplog):
    import logging
    import provider_status

    stdout_line = json.dumps(
        {
            "type": "progress",
            "provider": "openai",
            "stage": "input_sent",
        }
    ).encode()
    process = _make_fake_process(
        0,
        stdout_line + b"\n",
        b"stderr detail\n",
    )

    with caplog.at_level(logging.INFO, logger="provider_status"):
        with patch(
            "provider_status.asyncio.create_subprocess_exec",
            AsyncMock(return_value=process),
        ):
            asyncio.run(provider_status.provider_status_monitor.refresh_cache())

    assert "Provider status stdout:" in caplog.text
    assert "Provider status stderr: stderr detail" in caplog.text


# ---------------------------------------------------------------------------
# /openrouter/providers — filtered list path
# ---------------------------------------------------------------------------


def _openrouter_models_client(payload):
    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return payload

    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=FakeResp())
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)
    return fake_client


def test_openrouter_filtered_list(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_ALLOWED_PROVIDERS", "anthropic,openai")

    payload = {
        "data": [
            {"id": "anthropic/claude-sonnet-5", "name": "Anthropic: Claude Sonnet 5"},
            {"id": "openai/gpt-5", "name": "OpenAI: GPT-5"},
            {"id": "openai/gpt-4o", "name": "OpenAI: GPT-4o"},
            {"id": "google/gemini-2.5-pro", "name": "Google: Gemini 2.5 Pro"},
        ]
    }

    fake_client = _openrouter_models_client(payload)
    with patch("main.httpx.AsyncClient", return_value=fake_client):
        resp = client.get("/openrouter/providers")
    assert resp.status_code == 200
    body = resp.json()
    # Slug is the model-id author prefix, de-duped across models; name is the label.
    assert body["providers"] == [
        {"slug": "anthropic", "name": "Anthropic"},
        {"slug": "openai", "name": "OpenAI"},
    ]


def test_openrouter_filters_by_author_prefix_not_serving_provider(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    # The user allows "google" (the model-id author), NOT "google-ai-studio"
    # (OpenRouter's serving-provider slug) — Gemini must still surface.
    monkeypatch.setenv("OPENROUTER_ALLOWED_PROVIDERS", "openai,google")

    payload = {
        "data": [
            {"id": "openai/gpt-5", "name": "OpenAI: GPT-5"},
            {"id": "google/gemini-2.5-pro", "name": "Google: Gemini 2.5 Pro"},
        ]
    }

    fake_client = _openrouter_models_client(payload)
    with patch("main.httpx.AsyncClient", return_value=fake_client):
        resp = client.get("/openrouter/providers")
    assert resp.status_code == 200
    slugs = {p["slug"] for p in resp.json()["providers"]}
    assert slugs == {"openai", "google"}


def test_openrouter_filtered_list_http_error(client, monkeypatch):
    import httpx

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_ALLOWED_PROVIDERS", "anthropic")

    fake_client = MagicMock()
    fake_client.get = AsyncMock(side_effect=httpx.HTTPError("nope"))
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)

    with patch("main.httpx.AsyncClient", return_value=fake_client):
        resp = client.get("/openrouter/providers")
    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# /providers — availability by env var
# ---------------------------------------------------------------------------


def _clear_provider_env(monkeypatch):
    import provider_status

    monkeypatch.delenv("FAKE_AI_PROVIDERS", raising=False)
    for env_vars in provider_status.PROVIDER_ENV_VARS.values():
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)


def test_providers_none_configured(client, monkeypatch):
    _clear_provider_env(monkeypatch)
    resp = client.get("/providers")
    assert resp.status_code == 200
    assert resp.json() == {"providers": []}


def test_providers_reports_only_configured(client, monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "k")

    resp = client.get("/providers")
    assert resp.status_code == 200
    assert set(resp.json()["providers"]) == {"openai", "deepgram"}


def test_providers_empty_string_key_is_unset(client, monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "")

    resp = client.get("/providers")
    assert "sarvam" not in resp.json()["providers"]


def test_providers_multi_var_requires_all(client, monkeypatch):
    _clear_provider_env(monkeypatch)
    # google needs BOTH credentials + project id — one alone is not enough.
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/creds.json")

    resp = client.get("/providers")
    assert "google" not in resp.json()["providers"]

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT_ID", "proj")
    resp = client.get("/providers")
    assert "google" in resp.json()["providers"]


def test_providers_fake_flag_returns_all(client, monkeypatch):
    import provider_status

    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("FAKE_AI_PROVIDERS", "1")

    resp = client.get("/providers")
    assert resp.status_code == 200
    assert resp.json()["providers"] == list(provider_status.PROVIDER_ENV_VARS)


# ---------------------------------------------------------------------------
# /sentry-debug
# ---------------------------------------------------------------------------


def test_sentry_debug_raises(client):
    """The endpoint raises ZeroDivisionError; TestClient re-raises it.
    Either outcome (500 from FastAPI handler, or exception bubbling) proves
    the handler ran."""
    try:
        resp = client.get("/sentry-debug")
        assert resp.status_code == 500
    except ZeroDivisionError:
        pass
