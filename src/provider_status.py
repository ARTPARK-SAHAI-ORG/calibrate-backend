import asyncio
import json
import logging
import os
import signal
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from utils import env_bool, get_calibrate_agent_cli


logger = logging.getLogger(__name__)

# Grace period for the killed probe to be reaped. Only a safety net: with the
# group killed and the pipes closed, wait() resolves immediately.
_REAP_TIMEOUT_SECONDS = 5

# Providers reported healthy during integration testing: the union of the STT +
# TTS integrations (docs.calibrate.artpark.ai/docs/integrations), minus groq —
# which `_without_groq` strips from the real status anyway.
_FAKE_PROVIDER_NAMES = (
    "openai",
    "deepgram",
    "cartesia",
    "elevenlabs",
    "google",
    "gemini",
    "sarvam",
    "smallest",
)

# Every provider the calibrate CLI can probe, mapped to the env vars that must
# ALL be set for it to be usable. Mirrors calibrate_agent.status.PROVIDERS —
# kept in sync by hand because the backend only ever spawns that CLI as a
# subprocess, never imports it as a library.
PROVIDER_ENV_VARS: Dict[str, list[str]] = {
    "deepgram": ["DEEPGRAM_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "google": ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT_ID"],
    "gemini": ["GOOGLE_API_KEY"],
    "sarvam": ["SARVAM_API_KEY"],
    "elevenlabs": ["ELEVENLABS_API_KEY"],
    "cartesia": ["CARTESIA_API_KEY"],
    "smallest": ["SMALLEST_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}


def available_provider_names() -> list[str]:
    """Provider names whose required env vars are all set (registry order)."""
    # Integration testing: treat every provider as configured so the UI shows
    # the full set without real keys (see FAKE_AI_PROVIDERS in utils.py).
    if env_bool("FAKE_AI_PROVIDERS", False):
        return list(PROVIDER_ENV_VARS)
    return [
        name
        for name, env_vars in PROVIDER_ENV_VARS.items()
        if all(os.getenv(var) for var in env_vars)
    ]


def _fake_healthy_providers() -> Dict[str, Any]:
    return {name: {"status": "pass"} for name in _FAKE_PROVIDER_NAMES}


def _without_groq(providers: Dict[str, Any]) -> Dict[str, Any]:
    return {
        name: info for name, info in providers.items() if name.lower() != "groq"
    }


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def parse_provider_status_stdout(stdout: str) -> Dict[str, Any]:
    try:
        providers = json.loads(stdout)
        if isinstance(providers, dict) and providers.get("type") is None:
            return providers
    except json.JSONDecodeError:
        pass

    providers_from_events: Dict[str, Any] = {}
    final_providers: Optional[Dict[str, Any]] = None

    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(event, dict):
            continue
        if event.get("type") == "result" and event.get("provider"):
            providers_from_events[event["provider"]] = event.get("result", {})
        elif event.get("type") is None:
            final_providers = event

    if final_providers is not None:
        return final_providers
    if providers_from_events:
        return providers_from_events

    raise ValueError("Failed to parse calibrate status output")


def _failed_provider_details(providers: Dict[str, Any]) -> Dict[str, Any]:
    failed_providers = {
        name: info for name, info in providers.items() if info.get("status") != "pass"
    }
    failed_names = ", ".join(failed_providers.keys())
    errors = {
        name: info.get("error", "unknown error")
        for name, info in failed_providers.items()
    }
    return {
        "message": f"Providers failing: {failed_names}",
        "failed_providers": errors,
        "all_providers": providers,
    }


class ProviderStatusMonitor:
    def __init__(
        self,
        *,
        refresh_interval_seconds: int,
        cache_max_age_seconds: int,
        check_timeout_seconds: int,
    ):
        self.refresh_interval_seconds = refresh_interval_seconds
        self.cache_max_age_seconds = cache_max_age_seconds
        self.check_timeout_seconds = check_timeout_seconds
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> "ProviderStatusMonitor":
        return cls(
            refresh_interval_seconds=int(
                os.getenv("PROVIDER_STATUS_REFRESH_INTERVAL_SECONDS", "300")
            ),
            cache_max_age_seconds=int(
                os.getenv("PROVIDER_STATUS_CACHE_MAX_AGE_SECONDS", "900")
            ),
            check_timeout_seconds=int(
                os.getenv("PROVIDER_STATUS_CHECK_TIMEOUT_SECONDS", "120")
            ),
        )

    async def _read_process_stream(
        self,
        stream: asyncio.StreamReader,
        *,
        stream_name: str,
    ) -> bytes:
        chunks = []
        while True:
            line = await stream.readline()
            if not line:
                break

            chunks.append(line)
            decoded_line = line.decode(errors="replace").rstrip()
            if not decoded_line:
                continue

            if stream_name == "stderr":
                logger.warning("Provider status stderr: %s", decoded_line)
            else:
                logger.info("Provider status stdout: %s", decoded_line)

        return b"".join(chunks)

    async def _collect_process_output(
        self,
        process: asyncio.subprocess.Process,
    ) -> tuple[bytes, bytes]:
        if process.stdout is None or process.stderr is None:
            return await process.communicate()

        stdout_task = asyncio.create_task(
            self._read_process_stream(process.stdout, stream_name="stdout")
        )
        stderr_task = asyncio.create_task(
            self._read_process_stream(process.stderr, stream_name="stderr")
        )
        wait_task = asyncio.create_task(process.wait())

        try:
            stdout_bytes, stderr_bytes, _ = await asyncio.gather(
                stdout_task, stderr_task, wait_task
            )
        finally:
            # The lifespan cancels this check on shutdown with a CancelledError
            # (a BaseException), so `except Exception` would let the readers and
            # wait() outlive the loop. On the success path all three tasks are
            # done and cancel() is a no-op, so this drain covers every exit.
            for task in (stdout_task, stderr_task, wait_task):
                task.cancel()
            await asyncio.gather(
                stdout_task, stderr_task, wait_task, return_exceptions=True
            )

        return stdout_bytes, stderr_bytes

    async def _reap_subprocess(self, process: asyncio.subprocess.Process) -> None:
        """Kill the probe's whole process tree, close its pipes, then reap it.

        Motivation: `uv run` forks the CLI as a grandchild that inherits
        stdout/stderr, so killing only the direct child would leave the
        grandchild holding the pipes open and the CLI calling provider APIs with
        live keys. killpg takes both down at once.

        Close the transport before wait() to disconnect the pipes, not after.
        `wait()` returns at once for a child already reaped, but a waiter
        registered while one is still alive only resolves after every pipe
        disconnects. That is what the `wait()` below registers whenever the kill
        did not land (both errors are swallowed), so closing lets the reap
        resolve without timing out. This synchronous close() runs even if the
        task running this check is cancelled, preventing
        `BaseSubprocessTransport.__del__` from raising a
        RuntimeError('Event loop is closed').
        """
        if process.returncode is None:
            try:
                # pid == pgid because the spawn passes start_new_session=True.
                # SIGKILL directly: a status probe has no results to flush, so
                # no need for a SIGTERM grace period.
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

        # Private because asyncio.subprocess.Process exposes no close(). Already
        # a no-op on the happy path, where the protocol closes the transport
        # itself once every pipe has hit EOF and the process has exited.
        transport = getattr(process, "_transport", None)
        if transport is not None:
            transport.close()

        try:
            await asyncio.wait_for(process.wait(), timeout=_REAP_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("Provider status subprocess did not exit after SIGKILL")

    async def run_check(self) -> Dict[str, Any]:
        # Integration testing: skip the CLI probe and report every provider
        # healthy (see FAKE_AI_PROVIDERS in utils.py).
        if env_bool("FAKE_AI_PROVIDERS", False):
            return _fake_healthy_providers()
        try:
            process = await asyncio.create_subprocess_exec(
                "uv",
                "run",
                get_calibrate_agent_cli(),
                "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Give it its process group, so we can killpg it and not the
                # server (matching behavior of other subprocess spawns).
                start_new_session=True,
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="calibrate-agent CLI not found",
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                self._collect_process_output(process),
                timeout=self.check_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Provider status check timed out",
            )
        finally:
            await self._reap_subprocess(process)

        stdout = stdout_bytes.decode()
        stderr = stderr_bytes.decode()

        if process.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"calibrate status failed: {stderr.strip()}",
            )

        try:
            providers = _without_groq(parse_provider_status_stdout(stdout))
        except ValueError as exc:
            raise HTTPException(
                status_code=500,
                detail=str(exc),
            )

        return providers

    async def refresh_cache(self) -> None:
        logger.info("Provider status refresh started")
        try:
            providers = await self.run_check()
            failed_count = sum(
                1 for info in providers.values() if info.get("status") != "pass"
            )
            logger.info(
                "Provider status refresh completed: providers=%s failed=%s",
                len(providers),
                failed_count,
            )
            checked_at = _utc_now_iso()
            cache_entry = {
                "checked_at": checked_at,
                "providers": providers,
                "error_status_code": None,
                "error_detail": None,
            }
        except HTTPException as exc:
            logger.warning("Provider status refresh failed: %s", exc.detail)
            checked_at = _utc_now_iso()
            cache_entry = {
                "checked_at": checked_at,
                "providers": None,
                "error_status_code": exc.status_code,
                "error_detail": exc.detail,
            }

        async with self._cache_lock:
            self._cache = cache_entry

    async def refresh_loop(self) -> None:
        while True:
            await self.refresh_cache()
            await asyncio.sleep(self.refresh_interval_seconds)

    def clear_cache(self) -> None:
        self._cache = None

    async def response(self, *, force_refresh: bool = False) -> JSONResponse:
        if force_refresh:
            await self.refresh_cache()
        async with self._cache_lock:
            cache_entry = self._cache
        return self._response_from_cache(cache_entry, force_refresh=force_refresh)

    def _cache_age_seconds(self, cache_entry: Dict[str, Any]) -> float:
        checked_at = cache_entry.get("checked_at")
        if not checked_at:
            return float("inf")
        checked_at_datetime = datetime.fromisoformat(checked_at.rstrip("Z"))
        return (datetime.utcnow() - checked_at_datetime).total_seconds()

    def _response_from_cache(
        self,
        cache_entry: Optional[Dict[str, Any]],
        *,
        force_refresh: bool = False,
    ) -> JSONResponse:
        if cache_entry is None:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "cached": False,
                    "message": "Provider status has not been checked yet",
                    **({"refreshed": True} if force_refresh else {}),
                },
            )

        age_seconds = self._cache_age_seconds(cache_entry)
        is_stale = age_seconds > self.cache_max_age_seconds
        base_payload: Dict[str, Any] = {
            "cached": True,
            "checked_at": cache_entry["checked_at"],
            "age_seconds": round(age_seconds, 3),
            "stale": is_stale,
        }
        if force_refresh:
            base_payload["refreshed"] = True

        if cache_entry.get("error_detail") is not None:
            return JSONResponse(
                status_code=cache_entry.get("error_status_code") or 503,
                content={
                    "success": False,
                    **base_payload,
                    "message": cache_entry["error_detail"],
                },
            )

        providers = cache_entry["providers"]
        failed_providers = {
            name: info for name, info in providers.items() if info.get("status") != "pass"
        }
        if failed_providers:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    **base_payload,
                    **_failed_provider_details(providers),
                },
            )

        if is_stale:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    **base_payload,
                    "message": "Provider status cache is stale",
                    "all_providers": providers,
                },
            )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                **base_payload,
                "all_providers": providers,
            },
        )


provider_status_monitor = ProviderStatusMonitor.from_env()
