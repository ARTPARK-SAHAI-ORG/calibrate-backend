"""Guard: fern/openapi-overrides.yml must name every Public API route (and only
those). The SDK is generated from the public spec, and Fern takes each method
name from this overrides file — a public endpoint missing here would ship in the
SDK with an ugly auto-derived name. So we fail the build instead, restoring the
enforcement the old export script provided. See CLAUDE.md, "Public API docs are
tag-gated" (the SYNC RULE).
"""

from pathlib import Path

import yaml

from main import _build_public_openapi

_OVERRIDES = Path(__file__).resolve().parents[1] / "fern" / "openapi-overrides.yml"


def _public_ops() -> set:
    spec = _build_public_openapi()
    return {(path, m.lower()) for path, ops in spec["paths"].items() for m in ops}


def _override_ops() -> dict:
    data = yaml.safe_load(_OVERRIDES.read_text())
    return {
        (path, m.lower()): op
        for path, methods in data["paths"].items()
        for m, op in methods.items()
    }


def test_overrides_cover_exactly_the_public_routes():
    public = _public_ops()
    overrides = set(_override_ops())
    assert not (public - overrides), (
        "Public API routes missing an SDK name in fern/openapi-overrides.yml: "
        f"{sorted(public - overrides)}"
    )
    assert not (overrides - public), (
        "fern/openapi-overrides.yml names routes that aren't Public API "
        f"(stale after a rename/removal?): {sorted(overrides - public)}"
    )


def test_every_override_has_group_and_method_names():
    for (path, method), op in _override_ops().items():
        assert op.get("x-fern-sdk-group-name"), f"{method.upper()} {path}: no x-fern-sdk-group-name"
        assert op.get("x-fern-sdk-method-name"), f"{method.upper()} {path}: no x-fern-sdk-method-name"
