#!/usr/bin/env bash
# Speakeasy emits brews[].token at the wrong level for GoReleaser >=2.17.
# Token must live under brews[].repository.token (same indent as branch).
set -euo pipefail

FILE="${1:-.speakeasy/out/cli/.goreleaser.yaml}"

if [[ ! -f "$FILE" ]]; then
  echo "No .goreleaser.yaml at $FILE — skipping patch"
  exit 0
fi

python3 - "$FILE" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

# Patched: token is a sibling of branch under repository (same indent).
if re.search(r"^(\s+)branch: main\n\1token:", text, re.MULTILINE):
    print(f"{path}: already patched")
    sys.exit(0)

patched, n = re.subn(
    r"(?m)^(\s+)branch: main\n\s+token: (.+)$",
    r"\1branch: main\n\1token: \2",
    text,
    count=1,
)
if n != 1:
    print(
        f"::warning::{path}: expected brews repository.token layout; no change made",
        file=sys.stderr,
    )
    sys.exit(0)

path.write_text(patched)
print(f"Patched {path} for GoReleaser >=2.17 (token under repository)")
PY
