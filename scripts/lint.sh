#!/usr/bin/env bash
# Unused / dead / duplicate code gate. Shared by .githooks/pre-commit and CI so
# the two can't drift. Config for all three tools lives in pyproject.toml.
set -euo pipefail

cd "$(dirname "$0")/.."

status=0

echo "→ ruff (unused imports, variables, arguments)"
uv run ruff check src tests scripts || status=1

echo "→ vulture (dead code)"
uv run vulture || status=1

echo "→ pylint (duplicate code)"
uv run pylint -j 4 src || status=1

if [ "$status" -ne 0 ]; then
    echo ""
    echo "✗ Lint failed. Fix the findings above."
    echo "  Most ruff findings auto-fix with: uv run ruff check --fix src tests scripts"
    exit 1
fi

echo "✓ Lint passed."
