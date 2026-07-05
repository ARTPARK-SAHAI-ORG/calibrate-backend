#!/usr/bin/env python3
"""Print a stable SHA-256 of a public OpenAPI spec (servers stripped, keys sorted)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def normalize_spec(spec: dict) -> dict:
    normalized = dict(spec)
    normalized.pop("servers", None)
    return normalized


def hash_spec_file(path: Path) -> str:
    spec = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(
        normalize_spec(spec),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "openapi/openapi.json")
    if not path.is_file():
        print(f"::error::OpenAPI spec not found: {path}", file=sys.stderr)
        sys.exit(1)
    print(hash_spec_file(path))


if __name__ == "__main__":
    main()
