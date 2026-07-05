#!/usr/bin/env python3
"""Print the next patch version from the latest v* tags on client repos."""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Iterable

CLIENT_REPOS = (
    "dalmia/calibrate-python-sdk",
    "dalmia/calibrate-cli",
)
TAG_RE = re.compile(r"^v(\d+\.\d+\.\d+)$")


def parse_version(tag: str) -> tuple[int, int, int] | None:
    match = TAG_RE.match(tag.strip())
    if not match:
        return None
    major, minor, patch = (int(part) for part in match.group(1).split("."))
    return major, minor, patch


def latest_version_from_tags(tags: Iterable[str]) -> tuple[int, int, int] | None:
    versions = [parsed for tag in tags if (parsed := parse_version(tag)) is not None]
    return max(versions) if versions else None


def bump_patch(version: tuple[int, int, int]) -> str:
    major, minor, patch = version
    return f"{major}.{minor}.{patch + 1}"


def fetch_repo_tags(repo: str) -> list[str]:
    output = subprocess.check_output(
        ["gh", "api", f"repos/{repo}/tags", "--paginate", "--jq", ".[].name"],
        text=True,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def next_version_from_client_repos() -> str:
    versions: list[tuple[int, int, int]] = []
    for repo in CLIENT_REPOS:
        for tag in fetch_repo_tags(repo):
            parsed = parse_version(tag)
            if parsed is not None:
                versions.append(parsed)
    latest = max(versions) if versions else None
    if latest is None:
        return "0.0.1"
    return bump_patch(latest)


def main() -> None:
    print(next_version_from_client_repos())


if __name__ == "__main__":
    main()
