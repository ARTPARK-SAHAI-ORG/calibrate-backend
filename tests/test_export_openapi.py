"""Tests for scripts/export_openapi.py — the Fern SDK spec exporter.

Locks in the two load-bearing properties: (1) the exported spec is the PUBLIC
surface only, and (2) every public operation carries a deliberate, hand-picked
Fern SDK method name — a new public endpoint without one must fail loudly.
"""

import importlib.util
from pathlib import Path

import pytest

# Load scripts/export_openapi.py by path — it's not on the test pythonpath.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_openapi.py"
_spec = importlib.util.spec_from_file_location("export_openapi", _SCRIPT)
export_openapi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_openapi)


EXPECTED_OPS = {
    ("/agent-tests/agent/{agent_uuid}/run", "post"): ("agent_tests", "run"),
    ("/agent-tests/run", "post"): ("agent_tests", "run_batch"),
    ("/agent-tests/run/{task_id}", "get"): ("agent_tests", "get_run"),
    ("/agents", "get"): ("agents", "list"),
    ("/agents/resolve", "post"): ("agents", "resolve"),
}


def test_sdk_names_match_expected_contract():
    """SDK_NAMES is the contract downstream repos import against; pin it."""
    assert export_openapi.SDK_NAMES == EXPECTED_OPS


def test_public_spec_is_public_surface_only():
    spec = export_openapi.build_public_spec()
    got = {
        (path, method.lower())
        for path, ops in spec["paths"].items()
        for method in ops
    }
    assert got == set(EXPECTED_OPS), "public spec drifted from the expected endpoint set"
    # No internal/JWT-only path should leak in.
    for path in spec["paths"]:
        assert path.startswith(("/agent-tests", "/agents"))


def test_every_public_op_has_clean_fern_names():
    spec = export_openapi.build_public_spec()
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            group, name = EXPECTED_OPS[(path, method.lower())]
            assert op["x-fern-sdk-group-name"] == group
            assert op["x-fern-sdk-method-name"] == name


def test_unmapped_public_op_fails_loudly():
    """A new Public API endpoint without an SDK_NAMES entry must raise."""
    fake = {"paths": {"/new-public-thing": {"get": {"tags": ["agents"]}}}}
    with pytest.raises(ValueError, match="no SDK name"):
        export_openapi.inject_fern_sdk_names(fake)
