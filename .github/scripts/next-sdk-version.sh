#!/usr/bin/env bash
# Print the next patch SDK/CLI version from existing sdk-v* tags (e.g. 0.0.6).
set -euo pipefail

git fetch --tags --force >/dev/null 2>&1 || true

LAST_TAG="$(git tag -l 'sdk-v*' --sort=-v:refname | head -1 || true)"
if [[ -z "$LAST_TAG" ]]; then
  echo "0.0.1"
  exit 0
fi

CURRENT="${LAST_TAG#sdk-v}"
python3 - "${CURRENT}" <<'PY'
import sys

parts = sys.argv[1].split(".")
if len(parts) != 3 or not all(p.isdigit() for p in parts):
    raise SystemExit(f"::error::Invalid sdk-v* tag version: {sys.argv[1]!r}")
parts[-1] = str(int(parts[-1]) + 1)
print(".".join(parts))
PY
