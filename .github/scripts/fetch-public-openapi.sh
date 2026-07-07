#!/usr/bin/env bash
# Boot the backend (deps must already be installed) and write /public-api/openapi.json.
set -euo pipefail

OUT="${1:-openapi/openapi.json}"
ROOT="${GITHUB_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 1

export DB_ROOT_DIR="${DB_ROOT_DIR:-/tmp/sdk-openapi}"
export JWT_SECRET_KEY="${JWT_SECRET_KEY:-sdk-export-dummy-secret-key-32-chars-min}"
export S3_OUTPUT_BUCKET="${S3_OUTPUT_BUCKET:-sdk-export-dummy-bucket}"
export MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-1}"
export MAX_CONCURRENT_JOBS_PER_ORG="${MAX_CONCURRENT_JOBS_PER_ORG:-1}"
export DEFAULT_MAX_ROWS_PER_EVAL="${DEFAULT_MAX_ROWS_PER_EVAL:-20}"
export SUPERADMIN_EMAIL="${SUPERADMIN_EMAIL:-admin@example.com}"
# Empty string counts as set for os.getenv — default so servers.url is never blank.
if [ -z "${PUBLIC_API_BASE_URL:-}" ]; then
  export PUBLIC_API_BASE_URL="http://localhost:8000"
fi

(cd src && uv run uvicorn main:app --port 8000 --log-level warning &)
for _ in $(seq 1 30); do
  curl -sf http://localhost:8000/public-api/openapi.json -o /dev/null && break
  sleep 2
done
mkdir -p "$(dirname "${OUT}")"
curl -sf http://localhost:8000/public-api/openapi.json -o "${OUT}"
test -s "${OUT}" || { echo "::error::Failed to fetch public OpenAPI spec to ${OUT}" >&2; exit 1; }
