#!/usr/bin/env bash
# Create a v<version> tag on a client repo's main branch to trigger its own
# release/publish CI. Shared by publish-sdk.yml across the SDK, CLI, and MCP jobs.
# Usage: tag-client-repo.sh <owner/repo> <version>
# Requires GH_TOKEN in the environment (a PAT with contents:write on the repo).
set -euo pipefail

REPO="${1:?owner/repo required}"
VERSION="${2:?version required}"
: "${GH_TOKEN:?GH_TOKEN is required}"

TAG="v${VERSION}"
SHA=$(gh api "repos/${REPO}/git/ref/heads/main" --jq .object.sha)
gh api "repos/${REPO}/git/refs" -f ref="refs/tags/${TAG}" -f sha="${SHA}"
echo "Tagged ${REPO} @ ${SHA} as ${TAG} — its ci.yml will publish the release."
