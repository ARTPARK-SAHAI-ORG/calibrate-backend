"""Guard: public-route naming overlays must cover exactly the Public API routes.

Python SDK (Fern) reads fern/openapi-overrides.yml; Speakeasy CLI reads
openapi/overlay.yaml. A missing entry ships ugly auto-derived names.
See CLAUDE.md, "Public API docs are tag-gated" (the SYNC RULE).
"""

import re
from pathlib import Path

import yaml

from main import _build_public_openapi

_FERN_OVERRIDES = Path(__file__).resolve().parents[1] / "fern" / "openapi-overrides.yml"
_SPEAKEASY_OVERLAY = Path(__file__).resolve().parents[1] / "openapi" / "overlay.yaml"
_OVERLAY_TARGET_RE = re.compile(r"""^\$\.paths\['([^']+)'\]\.(\w+)$""")


def _public_ops() -> set:
    spec = _build_public_openapi()
    return {(path, m.lower()) for path, ops in spec["paths"].items() for m in ops}


def _fern_override_ops() -> dict:
    data = yaml.safe_load(_FERN_OVERRIDES.read_text())
    return {
        (path, m.lower()): op
        for path, methods in data["paths"].items()
        for m, op in methods.items()
    }


def _speakeasy_overlay_ops() -> dict:
    data = yaml.safe_load(_SPEAKEASY_OVERLAY.read_text())
    result = {}
    for action in data.get("actions", []):
        match = _OVERLAY_TARGET_RE.match(action["target"])
        assert match, f"Unparseable overlay target: {action['target']!r}"
        path, method = match.group(1), match.group(2).lower()
        result[(path, method)] = action["update"]
    return result


def test_fern_overrides_cover_exactly_the_public_routes():
    public = _public_ops()
    overrides = set(_fern_override_ops())
    assert not (public - overrides), (
        "Public API routes missing an SDK name in fern/openapi-overrides.yml: "
        f"{sorted(public - overrides)}"
    )
    assert not (overrides - public), (
        "fern/openapi-overrides.yml names routes that aren't Public API "
        f"(stale after a rename/removal?): {sorted(overrides - public)}"
    )


def test_every_fern_override_has_group_and_method_names():
    for (path, method), op in _fern_override_ops().items():
        assert op.get("x-fern-sdk-group-name"), f"{method.upper()} {path}: no x-fern-sdk-group-name"
        assert op.get("x-fern-sdk-method-name"), f"{method.upper()} {path}: no x-fern-sdk-method-name"


def test_speakeasy_overlay_covers_exactly_the_public_routes():
    public = _public_ops()
    overrides = set(_speakeasy_overlay_ops())
    assert not (public - overrides), (
        "Public API routes missing an SDK name in openapi/overlay.yaml: "
        f"{sorted(public - overrides)}"
    )
    assert not (overrides - public), (
        "openapi/overlay.yaml names routes that aren't Public API "
        f"(stale after a rename/removal?): {sorted(overrides - public)}"
    )


def test_every_speakeasy_overlay_has_group_and_method_names():
    for (path, method), op in _speakeasy_overlay_ops().items():
        assert op.get("x-speakeasy-group"), f"{method.upper()} {path}: no x-speakeasy-group"
        assert op.get("x-speakeasy-name-override"), (
            f"{method.upper()} {path}: no x-speakeasy-name-override"
        )


def test_every_speakeasy_overlay_has_mcp_metadata():
    for (path, method), op in _speakeasy_overlay_ops().items():
        mcp = op.get("x-speakeasy-mcp")
        assert mcp, f"{method.upper()} {path}: no x-speakeasy-mcp"
        assert mcp.get("name") or mcp.get("scopes"), (
            f"{method.upper()} {path}: x-speakeasy-mcp needs name or scopes"
        )


def test_no_destructive_or_visibility_route_is_public():
    """Guard the public-surface design decision (see CLAUDE.md): the API-key
    surface exposes create/read/update/launch only. No DELETE reaches the
    anonymous SDK page, and no `/visibility` share-toggle (UI-only) leaks into
    it. Publishing one is almost always an accident — fail loudly here so it
    can't ship silently just because someone also added an overlay entry.
    """
    offenders = [
        (path, method)
        for (path, method) in _public_ops()
        if method == "delete" or path.endswith("/visibility")
    ]
    assert not offenders, (
        "Destructive/visibility routes must not be tagged Public API: "
        f"{sorted(offenders)}"
    )


def test_public_surface_has_only_api_key_security():
    """Every public op must advertise the single apiKey scheme as its sole auth
    (see `_build_public_openapi`), so Fern/Speakeasy emit `api_key` as the one
    required constructor arg. A public route wired to a JWT-only dependency
    (e.g. `get_current_user_id`) would still render but 401 for real API-key
    clients — this catches that class of mistake.
    """
    spec = _build_public_openapi()
    schemes = set(spec.get("components", {}).get("securitySchemes", {}))
    assert schemes == {"ApiKeyAuth"}, f"unexpected public security schemes: {schemes}"
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            sec = op.get("security")
            assert sec == [{"ApiKeyAuth": []}], (
                f"{method.upper()} {path}: public op must require only ApiKeyAuth, got {sec}"
            )
