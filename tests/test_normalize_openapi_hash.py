"""Tests for stable public OpenAPI spec hashing used in auto-publish-sdk.yml."""

import json
import subprocess
import sys
from pathlib import Path

from main import _build_public_openapi

_SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "normalize_openapi_hash.py"


def _hash_spec(spec: dict) -> str:
    path = Path(__file__).resolve().parent / "_tmp_public_openapi.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    try:
        return subprocess.check_output(
            [sys.executable, str(_SCRIPT), str(path)],
            text=True,
        ).strip()
    finally:
        path.unlink(missing_ok=True)


def test_hash_ignores_servers_block():
    base = _build_public_openapi()
    with_servers = dict(base)
    with_servers["servers"] = [{"url": "https://example-a.test"}]
    without_servers = dict(base)
    without_servers.pop("servers", None)
    assert _hash_spec(with_servers) == _hash_spec(without_servers)


def test_hash_changes_when_paths_change():
    base = _build_public_openapi()
    mutated = json.loads(json.dumps(base))
    mutated["paths"]["/synthetic"] = {"get": {"responses": {"200": {"description": "ok"}}}}
    assert _hash_spec(base) != _hash_spec(mutated)
