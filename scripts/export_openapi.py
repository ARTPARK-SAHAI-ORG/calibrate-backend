"""Export the **public** OpenAPI spec to ``openapi.json`` at the repo root.

This is the single source of truth Fern generates the Python SDK from. It is
deliberately the *public* surface (``_build_public_openapi()`` in
:mod:`main`) — the subset of endpoints tagged ``Public API`` that accept an
``sk_`` org API key — NOT the full internal ``/openapi.json``. Internal
JWT-only endpoints therefore never enter the SDK: the auth boundary is
structural (which tag a route carries), so the SDK can never drift out of sync
with what we intend to publicly support.

We also inject Fern's ``x-fern-sdk-group-name`` / ``x-fern-sdk-method-name``
extensions so the generated client has stable, hand-picked method names
(``client.agent_tests.run(...)`` etc.) instead of the long, mechanical names
Fern would otherwise derive from FastAPI's auto-generated operationIds. The
name map is the contract downstream repos import against — see ``SDK_NAMES``.

Run it (from the repo root)::

    uv run python scripts/export_openapi.py

It writes ``openapi.json`` and prints a summary. It needs no secrets: env vars
read at import time by ``src/`` modules are seeded with throwaway defaults
below (we only ever call ``app.openapi()`` — nothing touches the DB or S3).
"""

import json
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
_OUTPUT = _REPO_ROOT / "openapi.json"

# Seed throwaway env BEFORE importing any `src/` module — several resolve env
# vars at import time (db.py's DB_PATH, JWT/S3 config). Mirrors tests/conftest.py.
# `setdefault` means a real environment (e.g. a dev's shell) still wins.
os.environ.setdefault("DB_ROOT_DIR", str(Path(tempfile.gettempdir()) / "pense-openapi-export"))
os.environ.setdefault("JWT_SECRET_KEY", "openapi-export-dummy-secret-key-32-chars-min")
os.environ.setdefault("JWT_EXPIRATION_HOURS", "1")
os.environ.setdefault("S3_OUTPUT_BUCKET", "openapi-export-dummy-bucket")
os.environ.setdefault("MAX_CONCURRENT_JOBS", "1")
os.environ.setdefault("MAX_CONCURRENT_JOBS_PER_ORG", "1")
os.environ.setdefault("DEFAULT_MAX_ROWS_PER_EVAL", "20")
os.environ.setdefault("SUPERADMIN_EMAIL", "admin@example.com")
os.environ.setdefault("DOCS_USERNAME", "admin")
os.environ.setdefault("DOCS_PASSWORD", "changeme")

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# Stable, hand-picked SDK names keyed by (path, http_method). Adding a new
# `Public API`-tagged endpoint REQUIRES adding it here — build_public_spec()
# raises if a public operation has no entry, so a new endpoint can never slip
# into the SDK with an auto-generated ugly name. (group, method) →
# client.<group>.<method>(...).
SDK_NAMES = {
    ("/agent-tests/agent/{agent_uuid}/run", "post"): ("agent_tests", "run"),
    ("/agent-tests/run", "post"): ("agent_tests", "run_batch"),
    ("/agent-tests/run/{task_id}", "get"): ("agent_tests", "get_run"),
    ("/agents", "get"): ("agents", "list"),
    ("/agents/resolve", "post"): ("agents", "resolve"),
}


def inject_fern_sdk_names(spec: dict) -> dict:
    """Annotate each public operation with Fern's SDK group/method extensions.

    Mutates ``spec`` in place and returns it. Raises ``ValueError`` if any
    public operation is missing from :data:`SDK_NAMES` — the forcing function
    that keeps the SDK surface deliberate.
    """
    unmapped = []
    for path, ops in spec.get("paths", {}).items():
        for method, op in ops.items():
            if not isinstance(op, dict):
                continue
            key = (path, method.lower())
            if key not in SDK_NAMES:
                unmapped.append(f"{method.upper()} {path}")
                continue
            group, name = SDK_NAMES[key]
            op["x-fern-sdk-group-name"] = group
            op["x-fern-sdk-method-name"] = name
    if unmapped:
        raise ValueError(
            "Public API endpoint(s) have no SDK name in scripts/export_openapi.py "
            "SDK_NAMES — add an entry so the generated SDK method name is "
            f"deliberate, not auto-derived:\n  " + "\n  ".join(unmapped)
        )
    return spec


def build_public_spec() -> dict:
    """Return the public OpenAPI spec with Fern SDK names injected."""
    # Import lazily so env seeding above is guaranteed to have run first.
    from main import _build_public_openapi

    return inject_fern_sdk_names(_build_public_openapi())


def main() -> None:
    spec = build_public_spec()
    _OUTPUT.write_text(json.dumps(spec, indent=2) + "\n")

    paths = spec.get("paths", {})
    op_count = sum(len(ops) for ops in paths.values())
    print(f"Wrote {_OUTPUT.relative_to(_REPO_ROOT)} — {len(paths)} paths, {op_count} operations:")
    for path, ops in sorted(paths.items()):
        for method, op in ops.items():
            group = op.get("x-fern-sdk-group-name")
            name = op.get("x-fern-sdk-method-name")
            print(f"  {method.upper():6} {path:42}  → client.{group}.{name}(...)")


if __name__ == "__main__":
    main()
