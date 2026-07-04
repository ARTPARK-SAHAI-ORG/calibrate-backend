# Fern — auto-generated Python SDK

This directory configures [Fern](https://buildwithfern.com) to generate and
publish a typed Python client for the **public** Calibrate API. No client code
is hand-written or committed to this repo — the SDK lives entirely on PyPI as
[`calibrate-python-sdk`](https://pypi.org/project/calibrate-python-sdk/), with
the main client class **`Calibrate`**.

```python
from calibrate import Calibrate  # import name confirmed after the first publish

client = Calibrate(api_key="sk_...")
client.agents.list()
client.agent_tests.run(agent_uuid="...", request=...)
```

## How it works

- **Source of truth is the public spec, not the full API.** `../openapi.json`
  is produced by [`scripts/export_openapi.py`](../scripts/export_openapi.py),
  which dumps only the `Public API`-tagged endpoints (`_build_public_openapi()`
  in `src/main.py`). Internal, JWT-only endpoints never enter the SDK — the auth
  boundary is structural (the tag), so the SDK can't drift out of sync with what
  we intend to support. To publish a new endpoint in the SDK, tag it
  `Public API` and give it a name in `SDK_NAMES` (see below).
- **Method names are hand-picked and stable.** The export script injects Fern's
  `x-fern-sdk-group-name` / `x-fern-sdk-method-name` so the client reads nicely:

  | Endpoint | SDK call |
  |---|---|
  | `POST /agent-tests/agent/{uuid}/run` | `client.agent_tests.run(...)` |
  | `POST /agent-tests/run` | `client.agent_tests.run_batch(...)` |
  | `GET /agent-tests/run/{task_id}` | `client.agent_tests.get_run(...)` |
  | `GET /agents` | `client.agents.list(...)` |
  | `POST /agents/resolve` | `client.agents.resolve(...)` |

  Adding a `Public API` endpoint without an `SDK_NAMES` entry makes the export
  **fail loudly** — a new public endpoint can never ship with an auto-derived
  ugly name by accident.
- **Publishing is SDK-tag driven, on a dedicated tag line.** [`.github/workflows/publish-sdk.yml`](../.github/workflows/publish-sdk.yml)
  runs on an **`sdk-v*`** tag, re-exports a fresh spec, and runs `fern generate`.
  The SDK version is the tag minus its `sdk-v` prefix (`sdk-v0.1.0` → `0.1.0`),
  passed via `fern generate --version`. The `sdk-v*` namespace is deliberately
  separate from the backend's own `v*` release tags (which drive `deploy.yml` on
  `release: published`) — so an SDK publish is always a deliberate act and never
  piggybacks on a backend deploy. PyPI versions are immutable, so **releasing the
  SDK = pushing a new `sdk-v*` tag.**

## One-time manual setup (needs human accounts)

1. **Fern account + org.** Create a Fern account and an organization named
   **`artpark`** (must match `fern.config.json`). Run `npx fern-api login` and
   grab a `FERN_TOKEN`.
2. **PyPI.** Claim the `calibrate-python-sdk` name (verified free at time of
   writing) and create a PyPI API token scoped to it.
3. **GitHub secrets.** Add both to the repo secrets: `FERN_TOKEN` and
   `PYPI_TOKEN`.
4. **Cut the first release:** `git tag sdk-v0.1.0 && git push origin sdk-v0.1.0`.

## Cutting a release

SDK releases use their own `sdk-v*` tag line — NOT the backend's `v*` release
tags. Only the leading `sdk-v` is stripped to form the PyPI version:

```bash
git tag sdk-v0.1.0
git push origin sdk-v0.1.0
```

The workflow publishes `calibrate-python-sdk==0.1.0`. To re-run/backfill a
specific version, use the workflow's manual `workflow_dispatch` (it requires a
`version` input).

## Regenerating the committed spec locally

```bash
uv run python scripts/export_openapi.py   # rewrites ../openapi.json
```

The committed `openapi.json` is a reviewable snapshot; CI always regenerates it
fresh before publishing, so a stale commit never affects a release.

## Follow-ups (not blocking)

- **Pin the generator version.** `generators.yml` uses `fernapi/fern-python-sdk`
  at `latest`. After the first successful publish, pin the exact version for
  reproducible builds.
- **Confirm the import name.** Fern derives the Python import name from
  `package-name`; confirm whether it emits `calibrate` or `calibrate_python_sdk`
  from the first generation and update the snippet above.
