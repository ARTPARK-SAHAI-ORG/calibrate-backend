"""Guard: openapi/overlay.yaml must name every Public API route (and only
those). The SDK/CLI are generated from the public spec, and Speakeasy takes each
method name from this overlay — a public endpoint missing here would ship with an
ugly auto-derived name. So we fail the build instead. See CLAUDE.md, "Public
API docs are tag-gated" (the SYNC RULE).
"""

import re
from pathlib import Path

import yaml

from main import _build_public_openapi

_OVERLAY = Path(__file__).resolve().parents[1] / "openapi" / "overlay.yaml"
_TARGET_RE = re.compile(r"""^\$\.paths\['([^']+)'\]\.(\w+)$""")


def _public_ops() -> set:
    spec = _build_public_openapi()
    return {(path, m.lower()) for path, ops in spec["paths"].items() for m in ops}


def _overlay_ops() -> dict:
    data = yaml.safe_load(_OVERLAY.read_text())
    result = {}
    for action in data.get("actions", []):
        match = _TARGET_RE.match(action["target"])
        assert match, f"Unparseable overlay target: {action['target']!r}"
        path, method = match.group(1), match.group(2).lower()
        result[(path, method)] = action["update"]
    return result


def test_overrides_cover_exactly_the_public_routes():
    public = _public_ops()
    overrides = set(_overlay_ops())
    assert not (public - overrides), (
        "Public API routes missing an SDK name in openapi/overlay.yaml: "
        f"{sorted(public - overrides)}"
    )
    assert not (overrides - public), (
        "openapi/overlay.yaml names routes that aren't Public API "
        f"(stale after a rename/removal?): {sorted(overrides - public)}"
    )


def test_every_override_has_group_and_method_names():
    for (path, method), op in _overlay_ops().items():
        assert op.get("x-speakeasy-group"), f"{method.upper()} {path}: no x-speakeasy-group"
        assert op.get("x-speakeasy-name-override"), (
            f"{method.upper()} {path}: no x-speakeasy-name-override"
        )
