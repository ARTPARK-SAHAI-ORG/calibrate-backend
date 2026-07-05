#!/usr/bin/env bash
# Print the next patch SDK/CLI version from the latest v* tags on client repos.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "${ROOT}/.github/scripts/next_sdk_version.py"
