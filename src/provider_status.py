import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from utils import get_calibrate_agent_cli


logger = logging.getLogger(__name__)


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
        except Exception:
            for task in (stdout_task, stderr_task, wait_task):
                task.cancel()
            raise

        return stdout_bytes, stderr_bytes

    async def run_check(self) -> Dict[str, Any]:
        try:
            process = await asyncio.create_subprocess_exec(
                "uv",
                "run",
                get_calibrate_agent_cli(),
                "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    self._collect_process_output(process),
                    timeout=self.check_timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise HTTPException(
                    status_code=504,
                    detail="Provider status check timed out",
                )
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="calibrate-agent CLI not found",
            )

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
