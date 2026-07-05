"""Tests for client-repo tag version bumping used in auto-publish-sdk.yml."""

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "next_sdk_version.py"
_SPEC = importlib.util.spec_from_file_location("next_sdk_version", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)


def test_latest_version_from_tags_uses_max_across_repos():
    tags = ["v0.0.4", "v0.0.5", "v0.0.2", "not-a-version", "v1.0.0"]
    assert _mod.latest_version_from_tags(tags) == (1, 0, 0)


def test_latest_version_from_tags_returns_none_when_empty():
    assert _mod.latest_version_from_tags([]) is None


def test_bump_patch_increments_only_patch_component():
    assert _mod.bump_patch((0, 0, 5)) == "0.0.6"
