# Self-Hosting Guide

This guide walks you through deploying Calibrate Backend for a new tenant, end to end. Pick your target cloud and follow that section — each is fully self-contained.

## Contents

1. [Architecture decisions](#architecture-decisions)
2. [Per-tenant isolation checklist](#per-tenant-isolation-checklist)
3. [Deploy on GCP](#deploy-on-gcp)
4. [Deploy on AWS](#deploy-on-aws)

---

## Architecture decisions

### Why a VM, not Cloud Run / ECS Fargate / Lambda

Three properties of this app rule out serverless containers:

1. **SQLite on a local volume.** Single-writer, file-system-bound. Multi-instance containers would corrupt it. (See `${APP_FOLDER_PATH}:/appdata` in `docker-compose.yml`.)
2. **Long-running `calibrate` subprocesses with process groups** — `os.killpg`, `start_new_session=True`, 5-minute timeout checks, abort signals. Needs a real OS, not a request-scoped sandbox.
3. **Temp-file-based intermediate results** that must persist for the lifetime of a job (CLAUDE.md: "STT/TTS intermediate results are disk-only").

So the recommended deployment is a **single VM running Docker Compose** — EC2 on AWS, GCE on GCP. Identical container, identical compose file, different host.

### What goes on the persistent disk

Only the SQLite file (`pense.db` under `DB_ROOT_DIR`, mounted at `/appdata` inside the container).

Everything else is ephemeral:

- **Audio uploads** never touch the server's disk. Clients `PUT` directly to object storage via presigned URLs from `POST /presigned-url`. The DB stores only the resulting `s3://bucket/key` reference.
- **Intermediate job artifacts** (calibrate CLI outputs, configs, generated audio, conversation `.wav`s) are written into `tempfile.TemporaryDirectory()` blocks. The backend uploads them to object storage from there; the temp dir is GC'd on context exit.
- **Subprocess stdout/stderr logs** also use `NamedTemporaryFile`.

So: **persistent disk = SQLite. Object storage = everything else.** A 20–50 GB persistent disk is plenty unless you set `TMPDIR=/appdata/tmp` to put job temp work on it as well (recommended — covered in each cloud section).

---

## Per-tenant isolation checklist

A new tenant gets its **own** copy of *all* of these. Never share with an existing tenant:

- VM + persistent disk (own SQLite DB at `APP_FOLDER_PATH`)
- Object storage bucket (`S3_OUTPUT_BUCKET`)
- `JWT_SECRET_KEY` (fresh `openssl rand -base64 32`)
- Google OAuth client ID (`GOOGLE_CLIENT_ID`) — its allowed origins point at the new domain
- `SUPERADMIN_EMAIL`, `DOCS_USERNAME` / `DOCS_PASSWORD`
- DNS name + TLS cert
- Sentry / Langfuse projects (or environment tag)
- `DEFAULT_USER_*` (the seeded admin)

API keys for upstream providers (OpenAI, Deepgram, OpenRouter, etc.) **may** be shared, but most teams want per-tenant keys for billing and quota separation.

---

# Deploy on GCP

End-to-end walkthrough on Google Cloud (Compute Engine + GCS). Substitute `<project-id>` with your GCP project ID.

## GCP / 0. Set gcloud defaults

```bash
gcloud config set project <project-id>
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
```

> **Gotcha:** `gcloud` does **not** read `$REGION` / `$ZONE` from your shell. It reads from `gcloud config`. If `gcloud config list` shows the wrong default zone (e.g. `asia-south1-a`), every command will silently target the wrong region. Verify with `gcloud config list` first.

## GCP / 1. Reserve a static IP

```bash
gcloud compute addresses create calibrate-backend-ip --region=us-central1
```

> A reserved-but-unattached static IP costs ~$0.01/hour (~$7/month). Once attached to a running VM, it's free. So either attach promptly (step 4) or release it (`gcloud compute addresses delete calibrate-backend-ip --region=us-central1`) until you're ready.

## GCP / 2. Create the persistent disk

```bash
gcloud compute disks create calibrate-appdata \
  --size=100GB --type=pd-balanced --zone=us-central1-a
```

You'll see a warning that the disk is unformatted — **that's fine**. Formatting happens after the VM is up; don't try to format from your laptop.

> **Disk sizing:** the SQLite file alone won't grow into the GB range unless you accumulate millions of dataset rows. 20 GB is enough for the DB. We use 100 GB so you can also point `TMPDIR` at the persistent disk for job temp work without filling the boot disk.

## GCP / 3. Verify firewall rules

GCP firewalls live on the **network**, not on instances. They attach via target tags. The default network usually has these pre-created:

```bash
gcloud compute firewall-rules list
```

You're looking for two rows:

| Need | What to check |
|---|---|
| Port 80 open | A row with `tcp:80` allowed, source `0.0.0.0/0`, target tag `http-server` (or empty target) |
| Port 443 open | A row with `tcp:443` allowed, source `0.0.0.0/0`, target tag `https-server` (or empty target) |

The default rules are named `default-allow-http` and `default-allow-https`. If both are present, **skip to step 4**. Otherwise create the missing one(s):

```bash
gcloud compute firewall-rules create allow-http  --allow tcp:80  --target-tags=http-server
gcloud compute firewall-rules create allow-https --allow tcp:443 --target-tags=https-server
```

> **Gotcha:** `--filter="targetTags:http-server"` and `--filter="targetTags=http-server"` **both error** on `firewall-rules list` due to a long-standing gcloud quirk. To filter, dump everything and grep client-side: `gcloud compute firewall-rules list --format="value(name,targetTags.list())" | grep http-server`.

## GCP / 4. Create the VM

```bash
gcloud compute instances create calibrate-backend \
  --machine-type=e2-standard-4 \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=30GB \
  --disk=name=calibrate-appdata,device-name=appdata,mode=rw,boot=no \
  --tags=http-server,https-server \
  --metadata=enable-oslogin=TRUE \
  --address=$(gcloud compute addresses describe calibrate-backend-ip --region=us-central1 --format='value(address)')
```

What each flag does:

- `--machine-type=e2-standard-4` — 4 vCPU, 16 GB RAM. Reasonable starting size; bump to `e2-standard-8` if benchmarks saturate it.
- `--image-family=debian-12 --image-project=debian-cloud` — Debian 12. **Ubuntu** is fine if your team prefers it: swap to `--image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud`. Both work identically with Docker.
- `--boot-disk-size=30GB` — the 10 GB default is too small once the OS, Docker, the image, and any temp files land on it.
- `--disk=name=calibrate-appdata,device-name=appdata,...` — attaches the persistent disk from step 2. **`device-name=appdata` is critical:** GCE creates a stable symlink at `/dev/disk/by-id/google-appdata` based on this name. `/etc/fstab` and the mount commands below depend on it.
- `--tags=http-server,https-server` — these tags are what tie the VM to the firewall rules from step 3.
- `--metadata=enable-oslogin=TRUE` — uses Google identity for SSH instead of static keys. Recommended for tenant deploys.
- `--address=$(...)` — attaches the static IP from step 1 in one shot. If you forget this, the VM gets an ephemeral IP that you'd swap later (see Troubleshooting at the bottom of this section).

## GCP / 5. SSH in and prepare the disk

```bash
gcloud compute ssh calibrate-backend
```

> **Gotcha:** if SSH errors with "resource not found" pointing at a different zone, it's the gcloud default-zone bug. Either pass `--zone=us-central1-a` explicitly or fix the default with `gcloud config set compute/zone us-central1-a`. Shell vars like `$ZONE` don't help — gcloud doesn't read them.

Inside the VM:

```bash
# Confirm the disk symlink exists
ls -l /dev/disk/by-id/ | grep appdata
# Expect: lrwxrwxrwx ... google-appdata -> ../../sdb

# Check whether already formatted (idempotency for re-runs of this guide)
sudo file -sL /dev/disk/by-id/google-appdata
# If output says "data" → blank, run mkfs below.
# If output mentions "ext4 filesystem" → already formatted, skip mkfs.
```

> **Gotcha:** `sudo file -s` on a symlink reports the symlink itself, not its target. Use `-L` to follow, or pass `/dev/sdb` directly. The `-sL` form is what works.

If blank, format it (**destructive — only the first time**):

```bash
sudo mkfs.ext4 -F /dev/disk/by-id/google-appdata
```

Mount and persist across reboots:

```bash
sudo mkdir -p /appdata
echo '/dev/disk/by-id/google-appdata /appdata ext4 discard,defaults 0 2' | sudo tee -a /etc/fstab
sudo mount /appdata
sudo chown -R $USER /appdata
```

What the `/etc/fstab` line does, field by field:

| Field | Value | Meaning |
|---|---|---|
| 1 | `/dev/disk/by-id/google-appdata` | What to mount (the GCE-stable symlink) |
| 2 | `/appdata` | Where to mount it |
| 3 | `ext4` | Filesystem type |
| 4 | `discard,defaults` | `discard` = SSD TRIM, `defaults` = standard rw/auto |
| 5 | `0` | Skip legacy `dump` backups |
| 6 | `2` | Run `fsck` on boot, after the root disk |

Verify:

```bash
df -h /appdata
# Expect: /dev/sdb (or similar)  ~98G  ...  /appdata
```

If it shows `/dev/root` instead, the disk didn't mount — you forgot the `sudo mount /appdata` step or the `/etc/fstab` line is malformed. **Writes to `/appdata` before you fix this go to the boot disk and disappear when you fix it.**

## GCP / 6. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
docker run --rm hello-world
```

Should print "Hello from Docker!" with no `permission denied` error. If you see permission denied, the `newgrp docker` didn't take effect — log out and back in.

## GCP / 7. Set up GCS (from your laptop)

```bash
PROJECT=$(gcloud config get-value project)

# 7a. Create the bucket
gcloud storage buckets create gs://calibrate-backend-artifacts \
  --location=us-central1 --uniform-bucket-level-access

# 7b. Enable versioning (recoverable from accidental overwrites/deletes)
gcloud storage buckets update gs://calibrate-backend-artifacts --versioning

# 7c. Service account for storage access
gcloud iam service-accounts create calibrate-backend-storage \
  --display-name="Calibrate backend storage"

# 7d. Grant object-level access on just this bucket (least privilege)
gcloud storage buckets add-iam-policy-binding gs://calibrate-backend-artifacts \
  --member="serviceAccount:calibrate-backend-storage@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# 7e. Generate HMAC keys — SAVE THE SECRET, it's shown ONCE
gcloud storage hmac create \
  calibrate-backend-storage@${PROJECT}.iam.gserviceaccount.com
```

Copy the `accessId` (looks like `GOOG1E...`) and `secret` from the output. You'll need them in the `.env` in step 9.

## GCP / 8. Clone the repo and build the image

On the VM:

```bash
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/<your-org>/calibrate-backend.git
cd calibrate-backend
docker build -t calibrate-backend:local .
```

The build takes 5–15 minutes the first time. The image lands in the VM's local Docker cache; no registry needed for the initial deploy. (You can graduate to GitHub Actions + Artifact Registry later — see CI/CD subsection.)

If the repo is private, set up a deploy key (preferred) or use a personal access token over HTTPS.

## GCP / 9. Create the `.env` file

On the VM, in the repo root:

```bash
cd ~/calibrate-backend
cat > .env <<'EOF'
# Image
IMAGE_NAME=calibrate-backend
IMAGE_TAG=local
CONTAINER_NAME=calibrate-backend
PORT=80

# Persistence
APP_FOLDER_PATH=/appdata
DB_ROOT_DIR=/appdata
TMPDIR=/appdata/tmp

# Auth — generate fresh, do NOT reuse from any other tenant
JWT_SECRET_KEY=PASTE_OUTPUT_OF_openssl_rand_-base64_32
JWT_EXPIRATION_HOURS=168

# Object storage (GCS via S3 interop)
S3_ENDPOINT_URL=https://storage.googleapis.com
S3_OUTPUT_BUCKET=calibrate-backend-artifacts
AWS_ACCESS_KEY_ID=<HMAC accessId from step 7e>
AWS_SECRET_ACCESS_KEY=<HMAC secret from step 7e>
AWS_REGION=auto

# Admin / default seeded user
SUPERADMIN_EMAIL=you@example.com
DEFAULT_USER_EMAIL=you@example.com
DEFAULT_USER_FIRST_NAME=You
DEFAULT_USER_LAST_NAME=Admin

# Docs HTTP basic auth
DOCS_USERNAME=admin
DOCS_PASSWORD=CHANGE_ME

# CORS — restrict to your frontend origin in production
CORS_ALLOWED_ORIGINS=*

# Concurrency
MAX_CONCURRENT_JOBS=1
MAX_CONCURRENT_JOBS_PER_USER=1
DEFAULT_MAX_ROWS_PER_EVAL=20

# Provider keys
OPENROUTER_API_KEY=
OPENAI_API_KEY=
DEEPGRAM_API_KEY=
CARTESIA_API_KEY=
SMALLEST_API_KEY=
GROQ_API_KEY=
SARVAM_API_KEY=
ELEVENLABS_API_KEY=
GOOGLE_API_KEY=
GOOGLE_CLIENT_ID=
GOOGLE_APPLICATION_CREDENTIALS=
GOOGLE_CLOUD_PROJECT_ID=

# Tracing
SENTRY_DSN=
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=1.0
SENTRY_PROFILES_SAMPLE_RATE=1.0
ENVIRONMENT=production
ENABLE_TRACING=false
OTEL_EXPORTER_OTLP_ENDPOINT=
OTEL_EXPORTER_OTLP_HEADERS=
LANGFUSE_TRACING_ENVIRONMENT=
LANGFUSE_HOST=
LANGFUSE_BASE_URL=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
EOF
chmod 600 .env

# Generate JWT secret and paste into .env
openssl rand -base64 32

# Create the TMPDIR
mkdir -p /appdata/tmp
```

> `PORT=80` works because the existing `default-allow-http` firewall rule already exposes port 80. Compose maps `${PORT}:8000` so the container listens on 8000 internally and the VM exposes it externally on 80. Switch to `PORT=8000` once Caddy is in front (HTTPS subsection below).

## GCP / 10. Start the app

```bash
docker compose up -d
docker compose logs -f
```

Watch for `Uvicorn running on http://0.0.0.0:8000`. Ctrl-C the log tail (the container keeps running).

## GCP / 11. Verify from the internet

From your laptop:

```bash
IP=$(gcloud compute addresses describe calibrate-backend-ip --region=us-central1 --format='value(address)')
curl http://$IP/openapi.json | head -c 200
```

If you get JSON back, the API is live.

## GCP / 12. Verify GCS uploads work

Inside the container:

```bash
docker exec -it calibrate-backend uv run python -c "
from utils import get_s3_client
c = get_s3_client()
print('endpoint:', c.meta.endpoint_url)
"
# Expect: endpoint: https://storage.googleapis.com
```

After running any job (or hitting `POST /presigned-url`):

```bash
gcloud storage ls --recursive gs://calibrate-backend-artifacts/ | head
```

You should see object keys appearing.

## GCP / Object storage (GCS via S3 interop)

The codebase only speaks the AWS S3 protocol via boto3, but `get_s3_client()` ([src/utils.py](src/utils.py)) honors `S3_ENDPOINT_URL`:

```python
endpoint_url = os.getenv("S3_ENDPOINT_URL")
if endpoint_url:
    kwargs["endpoint_url"] = endpoint_url
```

Pointing this at `https://storage.googleapis.com` + HMAC keys = boto3 talking to GCS. What works:

- `upload_file` / `put_object`
- `get_object`
- Presigned URLs for `get_object` and `put_object` (SigV4)
- The `s3://bucket/key` URI scheme stored in the DB — `presign_audio_path()` parses it as bucket+key, the client routes to whichever endpoint is configured. Nothing branches on the literal `s3://` string.

What's caveated:

- Multipart uploads — not exercised in this codebase (file sizes are small enough for single-part PUTs).
- HMAC keys are tied to the service account, not user-managed. If the SA is deleted, the keys die. **Don't delete `calibrate-backend-storage`** without first rotating to new keys on a different SA.

## GCP / Authentication and first login

### The seeded default user has no password

`init_db()` creates a row in the `users` table from `DEFAULT_USER_EMAIL`, but **does not set `password_hash`** ([src/db.py:803](src/db.py:803)). The seeded user can only log in via Google OAuth (or have a password set later via the API).

### Pick one

**Path A — Google OAuth (recommended for human users)**

1. GCP Console → APIs & Services → Credentials → Create OAuth client ID.
2. Application type: **Web application**.
3. Authorized JavaScript origins: your frontend's URL (e.g. `https://app.tenant.example.com`). For local testing, also add `http://localhost:3000`.
4. Copy the client ID into `GOOGLE_CLIENT_ID` in `.env`. Restart the container.
5. The Google email logging in must match `DEFAULT_USER_EMAIL` (or you'll create a second user).

**Path B — email/password signup**

1. Hit the password signup endpoint (check [src/routers/auth.py](src/routers/auth.py) for the exact route — typically `POST /auth/signup`).
2. Creates a new user row distinct from the seeded one. Not a superadmin unless their email matches `SUPERADMIN_EMAIL`.

**Path C — API key (for programmatic access)**

API keys (`/api-keys` endpoints) authenticate via `X-API-Key` or `Authorization: Bearer calib_...`. Useful for CI integrations and `POST /evaluators/{uuid}/invoke`. Created by an authenticated user — so chicken-and-egg, you need Path A or B first.

## GCP / Moving from HTTP to HTTPS (Caddy)

**Don't put real users on plain HTTP.** JWTs and basic-auth credentials cross the wire in cleartext. Once you've verified the deploy on `http://<ip>` and DNS is pointing at the VM, immediately put it behind TLS.

This section assumes you already have the app running on `PORT=80` and the domain resolves to the VM's static IP.

Caddy is the simplest option on Linux: one binary, one-line config, automatic Let's Encrypt cert provisioning + renewal, automatic HTTP→HTTPS redirect.

### Order matters

The container currently holds port 80. Caddy will need to take it over to handle the cert challenge and serve TLS. Sequence:

1. Install Caddy (it'll fail to bind port 80 — expected at this stage).
2. Move the container off port 80 (`PORT=80` → `PORT=8000`).
3. Configure Caddy with your domain + reverse proxy to localhost:8000.
4. Restart Caddy → it binds 80/443, fetches a cert, starts serving HTTPS.

### Step 1 — Install Caddy

On the VM:

```bash
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update && sudo apt-get install -y caddy
```

The install starts Caddy via systemd. It'll fail to bind port 80 (the container has it). That's expected — `systemctl status caddy` will be unhappy until step 3.

### Step 2 — Move the container off port 80

```bash
cd ~/calibrate-backend
sed -i 's/^PORT=80$/PORT=8000/' .env
docker compose up -d
```

Verify the swap:

```bash
docker compose ps                                # should show 0.0.0.0:8000->8000/tcp
curl http://localhost:8000/openapi.json | head -c 100   # should still return JSON
```

### Step 3 — Configure Caddy

```bash
sudo tee /etc/caddy/Caddyfile <<'EOF'
api.tenant.example.com {
    reverse_proxy localhost:8000
}
EOF

sudo systemctl restart caddy
sudo systemctl status caddy
```

Status should show `active (running)`. If it's still erroring:

```bash
sudo journalctl -u caddy -n 50 --no-pager
```

### Step 4 — Verify

From your laptop:

```bash
curl -I https://api.tenant.example.com/openapi.json
```

Expect `HTTP/2 200`. The first request triggers Caddy to provision a Let's Encrypt cert (HTTP-01 challenge); takes a few seconds, then it caches.

Browser: `https://api.tenant.example.com/docs`.

### Step 5 — Update CORS and OAuth

Set `CORS_ALLOWED_ORIGINS` to your **frontend's** origin (NOT the backend URL — see note below):

```bash
sed -i 's|^CORS_ALLOWED_ORIGINS=.*|CORS_ALLOWED_ORIGINS=https://app.tenant.example.com|' .env
docker compose up -d
```

Multiple origins are comma-separated:

```
CORS_ALLOWED_ORIGINS=https://app.tenant.example.com,https://staging.tenant.example.com,http://localhost:3000
```

If using Google OAuth, add `https://api.tenant.example.com` to your OAuth client's **Authorized JavaScript origins** in GCP Console → APIs & Services → Credentials.

> **What CORS does:** controls which *browser-tab origins* can call your backend. The backend's own URL never appears as an `Origin` header on requests to itself, so listing it here is a no-op. Same-origin tooling like Swagger UI on `/docs` doesn't need a CORS entry either. `curl` and Postman never trigger CORS at all.

### Gotchas

- **Port 80 must stay open**, even after HTTPS works. Caddy uses HTTP-01 for cert renewal every 60 days. If you close port 80 in the firewall, renewal silently fails and the cert eventually expires.
- **HTTP→HTTPS redirect is automatic.** Caddy adds a 308 from `http://...` to `https://...` for free. `curl -I http://api.tenant.example.com/` should return `308`.
- **DNS propagation** — if Caddy fails the cert challenge with "no such host" or "context deadline exceeded," DNS hasn't propagated to the cert authority yet. Wait 5–10 minutes, then `sudo systemctl restart caddy`.
- **Port 443 firewall rule** — `default-allow-https` should already cover this. If `https://` times out: `gcloud compute firewall-rules list | grep 443`.
- **Cert renewal** is automatic. No cron, no manual action — Caddy renews ~30 days before expiry.

### Alternative: GCP HTTPS Load Balancer

Heavier setup but offloads TLS, gives you GCP-managed certs, and lets you front multiple backends. Worth it if you want WAF (Cloud Armor), multi-region, or a single ingress for backend + frontend. Not required for a single-VM tenant.

## GCP / Operational concerns

### Docker log rotation

Already configured in `docker-compose.yml`:

```yaml
logging:
  driver: json-file
  options:
    max-size: "10m"
    max-file: "5"
```

Caps each container at ~50 MB total. **Recreate the container after pulling the change**: `docker compose up -d`. A `restart` is not enough — the logging driver is set on creation.

### Persistent disk snapshots

```bash
gcloud compute resource-policies create snapshot-schedule daily-appdata \
  --region=us-central1 \
  --max-retention-days=14 \
  --start-time=03:00 \
  --daily-schedule \
  --on-source-disk-delete=keep-auto-snapshots

gcloud compute disks add-resource-policies calibrate-appdata \
  --zone=us-central1-a \
  --resource-policies=daily-appdata
```

What this gives you:

- Daily snapshot at 03:00 UTC, retained for 14 days.
- Snapshots are incremental & compressed — typically tens of MB per day, total cost ~pennies/month.
- `--on-source-disk-delete=keep-auto-snapshots` means snapshots survive even if the disk is deleted.
- RPO = 24 hours. For tighter, use `--hourly-schedule --hours-in-cycle=6`.

### Where job temp files go

`tempfile.TemporaryDirectory()` follows `TMPDIR`, defaulting to `/tmp` (boot disk). Big benchmarks can stage hundreds of MB before upload. The `.env` template above sets `TMPDIR=/appdata/tmp`. Make sure the dir exists on the host:

```bash
mkdir -p /appdata/tmp
```

And the `TMPDIR` env var is plumbed through to the container — add to `docker-compose.yml`:

```yaml
- TMPDIR=${TMPDIR:-/tmp}
```

### Error monitoring (Sentry)

The architecture explicitly relies on `capture_exception_to_sentry()` for background-thread failures (CLAUDE.md: "All job failures route through `capture_exception_to_sentry()`"). Without `SENTRY_DSN`, those failures go to container stdout and nowhere else.

1. Create a Sentry project for this tenant.
2. Set `SENTRY_DSN`, `SENTRY_ENVIRONMENT=production` in `.env`.
3. Restart the container.

### Restart on VM reboot

The container has `restart: unless-stopped`. Docker starts at boot via systemd. **Test it once** before relying on it:

```bash
sudo reboot
# Wait 60 seconds, then from your laptop:
curl http://<ip>/openapi.json
```

If the API doesn't come back, check `systemctl status docker` and `docker compose ps` on the VM.

### Lock down SSH (Identity-Aware Proxy)

The default `default-allow-ssh` rule allows `tcp:22` from `0.0.0.0/0`. Switch to IAP-only ingress:

```bash
gcloud compute firewall-rules delete default-allow-ssh
gcloud compute firewall-rules create allow-iap-ssh \
  --allow=tcp:22 --source-ranges=35.235.240.0/20

# SSH from your laptop now uses --tunnel-through-iap
gcloud compute ssh calibrate-backend --tunnel-through-iap
```

Eliminates the entire bot-bruteforce SSH attack surface.

### Secret Manager (when ready)

`.env` on disk in cleartext is fine for a single-admin deploy. For tenant-grade, store each value as a Secret Manager secret and fetch on deploy:

```bash
echo -n "<value>" | gcloud secrets create calibrate-jwt-key --data-file=-

# Grant the VM's service account
gcloud secrets add-iam-policy-binding calibrate-jwt-key \
  --member="serviceAccount:<vm-sa-email>" \
  --role="roles/secretmanager.secretAccessor"

# At deploy time
gcloud secrets versions access latest --secret=calibrate-jwt-key
```

## GCP / CI/CD: replacing build-on-VM

Building on the VM works for a one-shot but doesn't scale. Switch to: build in GitHub Actions → push to Artifact Registry → SSH onto VM, `pull && up -d`.

The existing AWS workflows ([.github/workflows/deploy.yml](.github/workflows/deploy.yml), [.github/workflows/deploy-staging.yml](.github/workflows/deploy-staging.yml)) are the template. To adapt for GCP:

1. Create a new GitHub Actions environment with all the tenant's secrets.
2. Replace the EC2-targeted SSH step with one of:
   - `appleboy/ssh-action` against the GCE static IP (set up an OS Login key in GitHub secrets), **or**
   - [`google-github-actions/ssh-compute`](https://github.com/google-github-actions/ssh-compute) with Workload Identity Federation (no SSH key in secrets).
3. Swap Docker Hub for GCP Artifact Registry to keep the image close to the VM.
4. Use a distinct Compose project name (`docker compose -p calibrate-tenant-x`) so multiple tenants on one host don't collide.

## GCP / Troubleshooting

### "resource not found" on `gcloud compute ssh` pointing at a wrong zone

```
ERROR: ... 'projects/.../zones/asia-south1-a/instances/calibrate-backend' was not found
```

gcloud is using its global default zone, **not** your shell's `$ZONE`. Fix:

```bash
gcloud config list
gcloud config set compute/zone us-central1-a
gcloud config set compute/region us-central1
```

Or pass `--zone=us-central1-a` explicitly on every command.

### `df -h /appdata` shows `/dev/root` instead of `/dev/sdb`

The persistent disk isn't mounted. Most likely you ran `mkdir` and the `tee >> /etc/fstab` but not `sudo mount /appdata`. Also check `sudo file -sL /dev/disk/by-id/google-appdata` — if it says `data`, the disk needs `mkfs.ext4` first.

**Important:** anything written to `/appdata` while it was unmounted is on the boot disk. Once you mount the persistent disk over the same path, those files are shadowed (not deleted). To recover: `sudo umount /appdata && ls /appdata`.

### `file -s` reports "symbolic link to ../../sdb"

`-s` doesn't follow symlinks. Use `-sL`:

```bash
sudo file -sL /dev/disk/by-id/google-appdata
```

Or pass the resolved device: `sudo file -s /dev/sdb`.

### `gcloud compute firewall-rules list --filter="targetTags:http-server"` errors

Known gcloud quirk on this resource. Filter client-side:

```bash
gcloud compute firewall-rules list --format="value(name,targetTags.list())" | grep http-server
```

### Container exits immediately after `docker compose up -d`

Almost always a missing required env var. Check:

```bash
docker compose logs --tail=50
```

Look for "ValueError: S3_OUTPUT_BUCKET environment variable is required" or `KeyError: 'JWT_SECRET_KEY'`. Fill in, `docker compose up -d` again.

### `curl http://<ip>/...` times out

1. `docker compose ps` — STATUS should say `Up`.
2. Confirm port mapping in `docker compose ps`.
3. Confirm firewall rule covers your port (default rules cover 80/443; if you set `PORT=8000`, you need a separate rule).
4. Confirm static IP attached: `gcloud compute instances describe calibrate-backend --format="value(networkInterfaces[0].accessConfigs[0].natIP)"` should match `gcloud compute addresses describe calibrate-backend-ip --region=us-central1 --format="value(address)"`.

### Static IP shows `RESERVED` instead of `IN_USE`

Not attached to a VM. Either you forgot `--address=...` on `instances create`, or the VM was created with an ephemeral IP. To swap:

```bash
gcloud compute instances delete-access-config calibrate-backend \
  --access-config-name="external-nat"
gcloud compute instances add-access-config calibrate-backend \
  --access-config-name="external-nat" \
  --address=$(gcloud compute addresses describe calibrate-backend-ip --region=us-central1 --format='value(address)')
```

Brief network blip during the swap — SSH sessions drop. Plan around it for live deploys.

### GCS uploads hit the wrong endpoint

Inside the container:

```bash
docker exec -it calibrate-backend uv run python -c "
from utils import get_s3_client
c = get_s3_client()
print('endpoint:', c.meta.endpoint_url)
"
```

If it prints `https://s3.amazonaws.com` (or similar) instead of `https://storage.googleapis.com`, `S3_ENDPOINT_URL` didn't make it into the container. Check `.env`, then `docker compose up -d` to recreate.

### `docker run hello-world` says "permission denied" after install

`newgrp docker` didn't apply to your current shell. Log out and back in.

## GCP / Restore from snapshot

```bash
# 1. List recent snapshots
gcloud compute snapshots list --filter="sourceDisk:calibrate-appdata"

# 2. Create a new disk from the snapshot
gcloud compute disks create calibrate-appdata-restored \
  --source-snapshot=<snapshot-name> --zone=us-central1-a

# 3. Stop, swap, restart
gcloud compute instances stop calibrate-backend
gcloud compute instances detach-disk calibrate-backend --disk=calibrate-appdata
gcloud compute instances attach-disk calibrate-backend \
  --disk=calibrate-appdata-restored --device-name=appdata
gcloud compute instances start calibrate-backend
```

The `--device-name=appdata` is critical — it's what makes `/dev/disk/by-id/google-appdata` resolve, which `/etc/fstab` references. Without it the VM boots but `/appdata` stays unmounted.

After confirming the restore is good, delete the old disk: `gcloud compute disks delete calibrate-appdata --zone=us-central1-a`.

## GCP / Environment variable reference

See [src/.env.example](src/.env.example) for the canonical list. GCP-specific guidance:

| Var | Required? | GCP value |
|---|---|---|
| `IMAGE_NAME`, `IMAGE_TAG`, `CONTAINER_NAME`, `PORT` | **yes** | Compose interpolation |
| `APP_FOLDER_PATH` | **yes** | `/appdata` (your mount point) |
| `DB_ROOT_DIR` | **yes** | `/appdata` (in-container path) |
| `TMPDIR` | recommended | `/appdata/tmp` |
| `JWT_SECRET_KEY` | **yes** | `openssl rand -base64 32` per tenant |
| `S3_OUTPUT_BUCKET` | **yes** | Your GCS bucket name |
| `S3_ENDPOINT_URL` | **yes** | `https://storage.googleapis.com` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | **yes** | HMAC keys from step 7e |
| `AWS_REGION` | **yes** | `auto` (GCS ignores it; boto3 needs *something*) |
| `SUPERADMIN_EMAIL` | **yes** | For mutating user-limit endpoints |
| `DEFAULT_USER_EMAIL` / `..._FIRST_NAME` / `..._LAST_NAME` | **yes** | Seeded user |
| `GOOGLE_CLIENT_ID` | yes for OAuth | OAuth client ID for sign-in |
| `OPENROUTER_API_KEY` | yes for evaluators | Used by `POST /evaluators/{uuid}/invoke` |
| `OPENAI_API_KEY` and other provider keys | depends | Only what the tenant uses |
| `CORS_ALLOWED_ORIGINS` | recommended | **Frontend origin(s)**, comma-separated (e.g. `https://app.tenant.example.com`). NOT the backend URL — CORS gates browser-tab origins, not the API itself |
| `DOCS_USERNAME` / `DOCS_PASSWORD` | recommended | HTTP Basic auth on `/docs` |
| `MAX_CONCURRENT_JOBS` etc. | optional | Tune for VM size |
| `SENTRY_DSN` | recommended | Background-thread failures route through this |

> **When adding/changing/removing env vars:** update [src/.env.example](src/.env.example), [docker-compose.yml](docker-compose.yml), and the deploy workflows together. See `.cursor/rules/env-var.md`.

---

# Deploy on AWS

End-to-end walkthrough on AWS (EC2 + S3). Substitute `<region>` with your AWS region (e.g. `ap-south-1`, `us-east-1`).

> The existing AWS production deploy is fully automated by [.github/workflows/deploy.yml](.github/workflows/deploy.yml). For a brand-new tenant on AWS, you provision the infra once (steps 1–7 below), then add a GitHub Actions environment with the tenant's secrets and trigger that workflow for subsequent deploys. The first-time provisioning is the part this section walks through.

## AWS / 0. Set CLI defaults

```bash
aws configure              # if not already set up
export AWS_REGION=<region>
export AWS_DEFAULT_REGION=<region>
```

## AWS / 1. Create the S3 bucket

```bash
aws s3api create-bucket \
  --bucket calibrate-backend-artifacts \
  --region $AWS_REGION \
  --create-bucket-configuration LocationConstraint=$AWS_REGION

# Block all public access — the app uses presigned URLs for client access
aws s3api put-public-access-block \
  --bucket calibrate-backend-artifacts \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

# Enable versioning (recoverable from accidental overwrites/deletes)
aws s3api put-bucket-versioning \
  --bucket calibrate-backend-artifacts \
  --versioning-configuration Status=Enabled
```

> If your region is `us-east-1`, **omit** `--create-bucket-configuration` — that region rejects the flag.

Optionally add a lifecycle rule to expire non-current versions after 30 days so versioning doesn't balloon costs:

```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket calibrate-backend-artifacts \
  --lifecycle-configuration '{
    "Rules": [{
      "ID": "expire-old-versions",
      "Status": "Enabled",
      "Filter": {"Prefix": ""},
      "NoncurrentVersionExpiration": {"NoncurrentDays": 30}
    }]
  }'
```

## AWS / 2. Create an IAM role for the EC2 instance

This avoids putting AWS keys in `.env` — boto3 picks up creds from the EC2 metadata service automatically.

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

# Trust policy: allow EC2 to assume this role
cat > /tmp/trust-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

aws iam create-role \
  --role-name calibrate-backend-ec2 \
  --assume-role-policy-document file:///tmp/trust-policy.json

# Permissions: scoped to just this tenant's bucket
cat > /tmp/bucket-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket"
    ],
    "Resource": [
      "arn:aws:s3:::calibrate-backend-artifacts",
      "arn:aws:s3:::calibrate-backend-artifacts/*"
    ]
  }]
}
EOF

aws iam put-role-policy \
  --role-name calibrate-backend-ec2 \
  --policy-name calibrate-bucket-access \
  --policy-document file:///tmp/bucket-policy.json

# Also attach SSM-managed-instance policy so SSM Session Manager works
aws iam attach-role-policy \
  --role-name calibrate-backend-ec2 \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# Wrap in an instance profile (EC2 attaches via this, not the role directly)
aws iam create-instance-profile --instance-profile-name calibrate-backend-ec2
aws iam add-role-to-instance-profile \
  --instance-profile-name calibrate-backend-ec2 \
  --role-name calibrate-backend-ec2
```

## AWS / 3. Create the security group

```bash
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text)

SG_ID=$(aws ec2 create-security-group \
  --group-name calibrate-backend-sg \
  --description "Calibrate backend ingress" \
  --vpc-id $VPC_ID \
  --query GroupId --output text)

# HTTP and HTTPS open to the world
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 80  --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 443 --cidr 0.0.0.0/0

# Note: NOT opening port 22. We'll use SSM Session Manager instead.
echo "Security group: $SG_ID"
```

> **Why no SSH (port 22)?** SSM Session Manager (set up via the IAM policy in step 2) gives you shell access without exposing port 22 to the internet. Eliminates the bot-bruteforce attack surface entirely. If you'd rather have SSH, add `--protocol tcp --port 22 --cidr <your-ip>/32` (your IP only — not `0.0.0.0/0`).

## AWS / 4. Create the EBS volume for `/appdata`

```bash
# Pick the same AZ where the EC2 instance will live
AZ=${AWS_REGION}a

VOLUME_ID=$(aws ec2 create-volume \
  --availability-zone $AZ \
  --size 100 \
  --volume-type gp3 \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=calibrate-appdata}]' \
  --query VolumeId --output text)

# Wait for it to be available
aws ec2 wait volume-available --volume-ids $VOLUME_ID
echo "Volume: $VOLUME_ID"
```

> 100 GB is generous; the SQLite file alone is small. The headroom is so you can also point `TMPDIR` at the volume for job temp work.

## AWS / 5. Launch the EC2 instance

```bash
# Find the latest Ubuntu 22.04 AMI (or use Amazon Linux 2023 — both work)
AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' \
  --output text)

# Launch
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t3.large \
  --security-group-ids $SG_ID \
  --iam-instance-profile Name=calibrate-backend-ec2 \
  --placement AvailabilityZone=$AZ \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=calibrate-backend}]' \
  --query 'Instances[0].InstanceId' --output text)

aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Attach the EBS volume — pick a device name; /dev/sdf is conventional for first extra volume
aws ec2 attach-volume \
  --volume-id $VOLUME_ID \
  --instance-id $INSTANCE_ID \
  --device /dev/sdf

echo "Instance: $INSTANCE_ID"
```

> **Instance type:** `t3.large` (2 vCPU, 8 GB) is a reasonable starting size. `t3.xlarge` or `c6i.xlarge` if benchmarks saturate it. Use Graviton (`t4g.large`) for cost savings — Docker images need to be built for ARM64 in that case.

## AWS / 6. Allocate and associate an Elastic IP

```bash
EIP_ALLOC=$(aws ec2 allocate-address --domain vpc --query AllocationId --output text)
aws ec2 associate-address --instance-id $INSTANCE_ID --allocation-id $EIP_ALLOC

EIP=$(aws ec2 describe-addresses --allocation-ids $EIP_ALLOC --query 'Addresses[0].PublicIp' --output text)
echo "Elastic IP: $EIP"
```

> Elastic IPs are free while attached to a running instance and chargeable when unattached.

## AWS / 7. Connect via SSM Session Manager

```bash
aws ssm start-session --target $INSTANCE_ID
```

> If this errors with "instance not registered with SSM," wait 1–2 minutes after launch — the SSM agent takes time to come up. Confirm the IAM instance profile is attached and includes `AmazonSSMManagedInstanceCore`.

Once in the shell, become the default user:

```bash
sudo su - ubuntu       # for Ubuntu AMI
# or
sudo su - ec2-user     # for Amazon Linux
```

## AWS / 8. Prepare the EBS volume

Inside the instance:

```bash
# Find the device. AWS may name it /dev/xvdf, /dev/nvme1n1, or similar regardless of what you requested.
lsblk
# Look for an unmounted ~100GB device

# Substitute the actual device name in the commands below — for the example, assume /dev/nvme1n1
DEV=/dev/nvme1n1

# Check whether already formatted
sudo file -sL $DEV
# "data" → blank, run mkfs. "ext4 filesystem" → skip mkfs.

# Format if blank (DESTRUCTIVE — only the first time)
sudo mkfs.ext4 -F $DEV

# Mount and persist
sudo mkdir -p /appdata
echo "$DEV /appdata ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab
sudo mount /appdata
sudo chown -R $USER /appdata
df -h /appdata
```

> **Why `nofail` on AWS but not GCP?** EBS device naming on Nitro instances is non-deterministic across reboots. `nofail` prevents the system from refusing to boot if the device temporarily isn't visible. For maximum stability, use the EBS volume's UUID instead of the device path:
> ```bash
> UUID=$(sudo blkid -s UUID -o value $DEV)
> echo "UUID=$UUID /appdata ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab
> ```

## AWS / 9. Install Docker

```bash
# Ubuntu
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
docker run --rm hello-world
```

For Amazon Linux 2023, replace the `get.docker.com` line with:

```bash
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER && newgrp docker
# Compose plugin
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
```

## AWS / 10. Clone the repo and build the image

```bash
sudo apt-get update && sudo apt-get install -y git    # or: sudo dnf install -y git
git clone https://github.com/<your-org>/calibrate-backend.git
cd calibrate-backend
docker build -t calibrate-backend:local .
```

Takes 5–15 minutes the first time.

## AWS / 11. Create the `.env` file

```bash
cd ~/calibrate-backend
cat > .env <<'EOF'
# Image
IMAGE_NAME=calibrate-backend
IMAGE_TAG=local
CONTAINER_NAME=calibrate-backend
PORT=80

# Persistence
APP_FOLDER_PATH=/appdata
DB_ROOT_DIR=/appdata
TMPDIR=/appdata/tmp

# Auth — generate fresh, do NOT reuse from any other tenant
JWT_SECRET_KEY=PASTE_OUTPUT_OF_openssl_rand_-base64_32
JWT_EXPIRATION_HOURS=168

# Object storage (AWS S3 — IAM instance role provides creds)
S3_ENDPOINT_URL=
S3_OUTPUT_BUCKET=calibrate-backend-artifacts
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=<region>

# Admin / default seeded user
SUPERADMIN_EMAIL=you@example.com
DEFAULT_USER_EMAIL=you@example.com
DEFAULT_USER_FIRST_NAME=You
DEFAULT_USER_LAST_NAME=Admin

# Docs HTTP basic auth
DOCS_USERNAME=admin
DOCS_PASSWORD=CHANGE_ME

# CORS — restrict to your frontend origin in production
CORS_ALLOWED_ORIGINS=*

# Concurrency
MAX_CONCURRENT_JOBS=1
MAX_CONCURRENT_JOBS_PER_USER=1
DEFAULT_MAX_ROWS_PER_EVAL=20

# Provider keys
OPENROUTER_API_KEY=
OPENAI_API_KEY=
DEEPGRAM_API_KEY=
CARTESIA_API_KEY=
SMALLEST_API_KEY=
GROQ_API_KEY=
SARVAM_API_KEY=
ELEVENLABS_API_KEY=
GOOGLE_API_KEY=
GOOGLE_CLIENT_ID=
GOOGLE_APPLICATION_CREDENTIALS=
GOOGLE_CLOUD_PROJECT_ID=

# Tracing
SENTRY_DSN=
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=1.0
SENTRY_PROFILES_SAMPLE_RATE=1.0
ENVIRONMENT=production
ENABLE_TRACING=false
OTEL_EXPORTER_OTLP_ENDPOINT=
OTEL_EXPORTER_OTLP_HEADERS=
LANGFUSE_TRACING_ENVIRONMENT=
LANGFUSE_HOST=
LANGFUSE_BASE_URL=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
EOF
chmod 600 .env

# Generate JWT secret and paste into .env
openssl rand -base64 32

# Create the TMPDIR
mkdir -p /appdata/tmp
```

> **Critical for AWS:** leave `S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY` **empty**. With those empty and the IAM instance profile attached, boto3 picks up temporary credentials from the EC2 metadata service automatically. This is more secure than long-lived static keys.
>
> Set `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` only if you'd rather use a dedicated IAM user instead of the instance role.

## AWS / 12. Start the app

```bash
docker compose up -d
docker compose logs -f
```

Watch for `Uvicorn running on http://0.0.0.0:8000`. Ctrl-C the log tail.

## AWS / 13. Verify from the internet

From your laptop:

```bash
curl http://$EIP/openapi.json | head -c 200
```

If you get JSON back, the API is live.

## AWS / 14. Verify S3 uploads work

Inside the container:

```bash
docker exec -it calibrate-backend uv run python -c "
from utils import get_s3_client
c = get_s3_client()
print('endpoint:', c.meta.endpoint_url)
print('region:', c.meta.region_name)
"
# Expect endpoint: https://s3.<region>.amazonaws.com
```

After running any job:

```bash
aws s3 ls s3://calibrate-backend-artifacts/ --recursive | head
```

You should see object keys appearing.

## AWS / Object storage (AWS S3)

The codebase uses boto3 with default behavior:

- `S3_ENDPOINT_URL` empty → defaults to AWS S3
- Credentials picked up from EC2 instance role automatically (or static keys if set in `.env`)
- `AWS_REGION` selects the regional S3 endpoint
- `S3_OUTPUT_BUCKET` is the bucket name

The DB stores blob URIs as `s3://bucket/key` — same format the app uses everywhere.

## AWS / Authentication and first login

### The seeded default user has no password

`init_db()` creates a row in the `users` table from `DEFAULT_USER_EMAIL`, but **does not set `password_hash`** ([src/db.py:803](src/db.py:803)). Pick one:

**Path A — Google OAuth (recommended for human users)**

1. Google Cloud Console → APIs & Services → Credentials → Create OAuth client ID. (Yes — you can use a Google OAuth client even when deploying to AWS; it's just an identity provider.)
2. Application type: **Web application**.
3. Authorized JavaScript origins: your frontend's URL.
4. Copy the client ID into `GOOGLE_CLIENT_ID` in `.env`. Restart the container.
5. The Google email logging in must match `DEFAULT_USER_EMAIL`.

**Path B — email/password signup**

Hit the password signup endpoint (typically `POST /auth/signup` — confirm in [src/routers/auth.py](src/routers/auth.py)).

**Path C — API key**

API keys (`/api-keys`) authenticate via `X-API-Key` or `Authorization: Bearer calib_...`. Created by an authenticated user.

## AWS / Moving from HTTP to HTTPS (nginx + certbot)

**Don't put real users on plain HTTP.** Once verified on `http://<eip>` and DNS is pointing at the Elastic IP, immediately put it behind TLS.

This section assumes you already have the app running on `PORT=80` and the domain resolves to the Elastic IP.

The standard AWS-shop-friendly approach: nginx as a reverse proxy + certbot to issue and renew Let's Encrypt certs. nginx is what most ops engineers know, and certbot's `--nginx` plugin auto-edits the nginx config to add TLS — minimal hand-rolling.

### Order matters

The container currently holds port 80. nginx will need to take it over. Sequence:

1. Move the container off port 80 (`PORT=80` → `PORT=8000`).
2. Install nginx, configure as a plain HTTP reverse proxy on port 80.
3. Install certbot, run it against your domain — it issues a cert and rewrites the nginx config to terminate TLS on 443 and redirect 80→443.
4. Verify HTTPS works.

### Step 1 — Move the container off port 80

```bash
cd ~/calibrate-backend
sed -i 's/^PORT=80$/PORT=8000/' .env
docker compose up -d

# Verify
docker compose ps                                # should show 0.0.0.0:8000->8000/tcp
curl http://localhost:8000/openapi.json | head -c 100
```

Port 80 is now free for nginx.

### Step 2 — Install nginx

```bash
# Ubuntu
sudo apt-get update && sudo apt-get install -y nginx

# Amazon Linux 2023
# sudo dnf install -y nginx && sudo systemctl enable --now nginx
```

Confirm it's running and serving the default page:

```bash
curl http://localhost/   # should return nginx's default welcome page
```

### Step 3 — Configure nginx as a reverse proxy

Replace the default site with one that proxies to your container:

```bash
sudo tee /etc/nginx/sites-available/calibrate-backend <<'EOF'
server {
    listen 80;
    server_name api.tenant.example.com;

    # Increase upload size — TTS/STT audio uploads can be larger than the 1 MB default
    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Long-running requests (job status polls, CLI handoffs)
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
EOF

# Enable the site, disable the default
sudo ln -sf /etc/nginx/sites-available/calibrate-backend /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Validate config and reload
sudo nginx -t
sudo systemctl reload nginx
```

(On Amazon Linux there's no `sites-available` / `sites-enabled` convention — drop the file in `/etc/nginx/conf.d/calibrate-backend.conf` instead and skip the `ln`/`rm` steps.)

Verify HTTP through nginx works:

```bash
curl http://api.tenant.example.com/openapi.json | head -c 100
```

### Step 4 — Install certbot

The Let's Encrypt-recommended path on Ubuntu/Debian is via snap; on Amazon Linux it's via EPEL. Both work the same once installed.

**Ubuntu:**

```bash
sudo snap install --classic certbot
sudo ln -sf /snap/bin/certbot /usr/bin/certbot
```

**Amazon Linux 2023:**

```bash
sudo dnf install -y python3-certbot-nginx
```

### Step 5 — Issue the cert

```bash
sudo certbot --nginx -d api.tenant.example.com
```

certbot will:

1. Ask for an email (used for expiry warnings — give it one you actually read).
2. Ask whether to redirect HTTP to HTTPS — **answer yes**. It rewrites your nginx config to add a 301 from port 80 to 443.
3. Use the HTTP-01 challenge against `http://api.tenant.example.com/.well-known/acme-challenge/...` to prove domain control.
4. Install the cert at `/etc/letsencrypt/live/api.tenant.example.com/`.
5. Reload nginx.

If it errors with "Failed authorization procedure," DNS hasn't propagated yet or port 80 isn't reachable. Wait 5 minutes and retry. Confirm port 80 is open in the security group from `0.0.0.0/0`.

### Step 6 — Verify

From your laptop:

```bash
curl -I https://api.tenant.example.com/openapi.json
```

Expect `HTTP/2 200`. Browser: `https://api.tenant.example.com/docs`.

### Step 7 — Confirm auto-renewal

certbot installs a systemd timer (or cron job on older systems) that renews ~30 days before expiry. Test it:

```bash
sudo certbot renew --dry-run
```

This should report success without actually renewing. Then:

```bash
systemctl list-timers | grep certbot
```

You should see `snap.certbot.renew.timer` (snap install) or `certbot.timer` (apt/dnf install) scheduled.

### Step 8 — Update CORS and OAuth

Set `CORS_ALLOWED_ORIGINS` to your **frontend's** origin (NOT the backend URL — see note below):

```bash
sed -i 's|^CORS_ALLOWED_ORIGINS=.*|CORS_ALLOWED_ORIGINS=https://app.tenant.example.com|' .env
docker compose up -d
```

Multiple origins are comma-separated:

```
CORS_ALLOWED_ORIGINS=https://app.tenant.example.com,https://staging.tenant.example.com,http://localhost:3000
```

If using Google OAuth, add `https://api.tenant.example.com` to your OAuth client's **Authorized JavaScript origins** in Google Cloud Console → APIs & Services → Credentials.

> **What CORS does:** controls which *browser-tab origins* can call your backend. The backend's own URL never appears as an `Origin` header on requests to itself, so listing it here is a no-op. Same-origin tooling like Swagger UI on `/docs` doesn't need a CORS entry either. `curl` and Postman never trigger CORS at all.

### Gotchas

- **Port 80 must stay open** in the security group, even after HTTPS works. certbot uses HTTP-01 for renewal. Closing port 80 silently breaks renewal and the cert eventually expires.
- **Port 443 must also be open** — confirm with: `aws ec2 describe-security-groups --group-ids $SG_ID --query 'SecurityGroups[0].IpPermissions'`. If 443 isn't listed: `aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 443 --cidr 0.0.0.0/0`.
- **`client_max_body_size`** — default is 1 MB. Audio uploads will fail with `413 Request Entity Too Large` if not increased. The config above sets 100 MB.
- **Long-running endpoints** — nginx's default proxy timeout is 60s. Job-status polls and a few internal handoffs can exceed that. The config sets 300s.
- **`X-Forwarded-For` / `X-Forwarded-Proto`** are set so the FastAPI app behind nginx sees the real client IP and knows the original request was HTTPS (matters for any redirect logic and for accurate logging).
- **Cert files location**: `/etc/letsencrypt/live/<domain>/fullchain.pem` and `privkey.pem`. Don't move them — certbot's renewal hook expects them there.

### Alternative: AWS Application Load Balancer + ACM

Heavier setup but offloads TLS to a managed service, integrates with ACM for free public certs (auto-renewed by AWS), gives you health checks, and lets you front multiple backends. Worth it if you want WAF (AWS WAF), multi-AZ, or a single ingress for backend + frontend. The tradeoff: ALB costs ~$22/month minimum (LCUs) vs nginx-on-EC2 being free. Not required for a single-VM tenant.

## AWS / Operational concerns

### Docker log rotation

Already configured in `docker-compose.yml`:

```yaml
logging:
  driver: json-file
  options:
    max-size: "10m"
    max-file: "5"
```

**Recreate the container after pulling the change**: `docker compose up -d`. A `restart` is not enough.

### EBS snapshots (AWS Backup)

```bash
# Create a backup vault if you don't have one
aws backup create-backup-vault --backup-vault-name calibrate-backend-vault

# Create a backup plan: daily at 03:00 UTC, retain 14 days
cat > /tmp/backup-plan.json <<'EOF'
{
  "BackupPlanName": "calibrate-appdata-daily",
  "Rules": [{
    "RuleName": "DailyBackup",
    "TargetBackupVaultName": "calibrate-backend-vault",
    "ScheduleExpression": "cron(0 3 ? * * *)",
    "Lifecycle": {"DeleteAfterDays": 14}
  }]
}
EOF

PLAN_ID=$(aws backup create-backup-plan \
  --backup-plan file:///tmp/backup-plan.json \
  --query BackupPlanId --output text)

# Tag the volume so the plan picks it up
aws ec2 create-tags --resources $VOLUME_ID --tags Key=Backup,Value=Daily

# Selection: target volumes with that tag
cat > /tmp/selection.json <<EOF
{
  "SelectionName": "DailyAppdataSelection",
  "IamRoleArn": "arn:aws:iam::${ACCOUNT}:role/service-role/AWSBackupDefaultServiceRole",
  "ListOfTags": [{
    "ConditionType": "STRINGEQUALS",
    "ConditionKey": "Backup",
    "ConditionValue": "Daily"
  }]
}
EOF

aws backup create-backup-selection \
  --backup-plan-id $PLAN_ID \
  --backup-selection file:///tmp/selection.json
```

> If `AWSBackupDefaultServiceRole` doesn't exist in your account, AWS Backup auto-creates it the first time you use the console. Run the console flow once, then the CLI commands above will work.

Cheaper alternative (and equally fine for SQLite): **Data Lifecycle Manager** with an EBS snapshot policy. Same daily-retention semantics, simpler IAM.

### Where job temp files go

`tempfile.TemporaryDirectory()` follows `TMPDIR`. The `.env` template sets `TMPDIR=/appdata/tmp`. Make sure the dir exists:

```bash
mkdir -p /appdata/tmp
```

And add to `docker-compose.yml`:

```yaml
- TMPDIR=${TMPDIR:-/tmp}
```

### Error monitoring (Sentry)

The architecture explicitly relies on `capture_exception_to_sentry()` for background-thread failures. Without `SENTRY_DSN`, those failures go to container stdout and nowhere else. Create a Sentry project and set `SENTRY_DSN`, `SENTRY_ENVIRONMENT=production`.

### Restart on instance reboot

`restart: unless-stopped` in compose + Docker enabled at boot via systemd handles this. Test it:

```bash
sudo reboot
# Wait 60s, then from your laptop:
curl http://$EIP/openapi.json
```

### Lock down access (already done if you followed steps 2 + 3)

- No SSH port open → no bot bruteforce surface.
- SSM Session Manager via the IAM role for shell access.
- Bucket has public-access-block on; client access is via presigned URLs only.

### AWS Secrets Manager (when ready)

`.env` on disk is fine for a single-admin deploy. For tenant-grade:

```bash
# Create a secret
aws secretsmanager create-secret \
  --name calibrate/jwt-secret \
  --secret-string "<value>"

# Grant the EC2 instance role read access
aws iam put-role-policy \
  --role-name calibrate-backend-ec2 \
  --policy-name secrets-access \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:'"$AWS_REGION"':'"$ACCOUNT"':secret:calibrate/*"
    }]
  }'

# At deploy time
aws secretsmanager get-secret-value \
  --secret-id calibrate/jwt-secret \
  --query SecretString --output text
```

Bootstrap `.env` from secrets, run, delete the temp `.env` if you're being strict.

## AWS / CI/CD: the existing GitHub Actions workflow

Unlike the GCP path (which builds on the VM), AWS production already has a fully-automated workflow: [.github/workflows/deploy.yml](.github/workflows/deploy.yml). It builds the image in Actions, pushes to Docker Hub, then SSHes onto the EC2 instance and runs `docker compose pull && up -d`.

To onboard a new AWS tenant:

1. Create a new GitHub Actions environment (e.g. `Tenant-X-Production`).
2. Populate every `secrets.*` and `vars.*` referenced in [.github/workflows/deploy.yml](.github/workflows/deploy.yml) — `EC2_HOST` (the Elastic IP), `EC2_USER` (`ubuntu`/`ec2-user`), `EC2_SSH_KEY` (or use SSM and rewrite the SSH step), all the env vars from the `.env`, `IMAGE_NAME`, `CONTAINER_NAME`, `DOCKERHUB_*`, etc.
3. Copy `deploy.yml` to `deploy-<tenant>.yml`, change the `environment:` line to point at your new environment.
4. Trigger via "Run workflow" → workflow_dispatch.

Subsequent deploys are automatic — Actions handles build, push, SSH, pull, up. Steps 10–12 of this guide (build + start manually) are only the **first-time** manual bootstrap.

## AWS / Troubleshooting

### `lsblk` shows the EBS volume but `mkfs` errors

You may be looking at the wrong device. Nitro instances enumerate volumes as `/dev/nvme*n1` regardless of the `/dev/sdf` you specified at attach time. Run `lsblk` and pick the one with the right size and no mountpoint.

### `df -h /appdata` shows `/dev/root` instead of the EBS volume

Same as the GCP version: you ran `mkdir` and the `tee >> /etc/fstab` but skipped `sudo mount /appdata`. Also check `sudo file -sL <device>` — if it says `data`, the volume needs `mkfs.ext4` first.

**Important:** anything written to `/appdata` while it was unmounted is on the boot volume. Once the EBS volume mounts over the path, those files are shadowed (not deleted). Recover with `sudo umount /appdata && ls /appdata`.

### EBS volume not visible after instance reboot

If `/etc/fstab` references a device path like `/dev/nvme1n1` and AWS re-enumerates it as `/dev/nvme2n1` on reboot, the mount fails. Use the volume's UUID instead:

```bash
UUID=$(sudo blkid -s UUID -o value /dev/nvme1n1)
sudo sed -i "s|^/dev/nvme1n1.*|UUID=$UUID /appdata ext4 discard,defaults,nofail 0 2|" /etc/fstab
```

The `nofail` option keeps the system bootable even if the volume isn't visible at boot time.

### `aws ssm start-session` says "instance not registered"

1. Wait 1–2 minutes after launch — the SSM agent takes time.
2. Confirm IAM instance profile is attached: `aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].IamInstanceProfile'`
3. Confirm the role has `AmazonSSMManagedInstanceCore` attached.
4. Confirm the instance has internet access — without NAT/IGW, the SSM agent can't reach the SSM endpoint. Either give it a public IP (Elastic IP, step 6) or set up VPC endpoints for SSM.

### Container exits immediately after `docker compose up -d`

Almost always a missing required env var. Check:

```bash
docker compose logs --tail=50
```

Look for "ValueError: S3_OUTPUT_BUCKET environment variable is required" or `KeyError: 'JWT_SECRET_KEY'`.

### `curl http://<eip>/...` times out

1. `docker compose ps` — STATUS should say `Up`.
2. Confirm port mapping.
3. Confirm security group has tcp:80 (and 443 once nginx is in front) open from `0.0.0.0/0`.
4. Confirm the Elastic IP is associated: `aws ec2 describe-addresses --allocation-ids $EIP_ALLOC --query 'Addresses[0].InstanceId'` should return the instance ID.

### S3 operations fail with `NoCredentialsError`

The IAM instance role isn't attached or doesn't have the right permissions. Check:

```bash
# From inside the EC2 instance:
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
# Should print the role name (e.g. "calibrate-backend-ec2")

aws sts get-caller-identity
# Should show an assumed-role ARN containing "calibrate-backend-ec2"
```

If those work but bucket operations fail, check the role's bucket policy (step 2).

### `docker run hello-world` says "permission denied" after install

`newgrp docker` didn't apply. Log out and back in.

## AWS / Restore from EBS snapshot

```bash
# 1. List recent snapshots
aws ec2 describe-snapshots --owner-ids self \
  --filters "Name=volume-id,Values=$VOLUME_ID" \
  --query 'sort_by(Snapshots,&StartTime)[-5:].[SnapshotId,StartTime,Description]' \
  --output table

# 2. Create a new volume from the snapshot
NEW_VOLUME_ID=$(aws ec2 create-volume \
  --snapshot-id <snap-id> \
  --availability-zone $AZ \
  --volume-type gp3 \
  --query VolumeId --output text)
aws ec2 wait volume-available --volume-ids $NEW_VOLUME_ID

# 3. Stop the instance, swap, restart
aws ec2 stop-instances --instance-ids $INSTANCE_ID
aws ec2 wait instance-stopped --instance-ids $INSTANCE_ID
aws ec2 detach-volume --volume-id $VOLUME_ID
aws ec2 wait volume-available --volume-ids $VOLUME_ID
aws ec2 attach-volume --volume-id $NEW_VOLUME_ID --instance-id $INSTANCE_ID --device /dev/sdf
aws ec2 start-instances --instance-ids $INSTANCE_ID
```

If you used a UUID-based `/etc/fstab` entry, the new volume will have a different UUID and won't auto-mount. Either:

- Update `/etc/fstab` to the new UUID (`sudo blkid` to find it), or
- Use device-path-based fstab entries (less stable but matches across volume swaps).

After confirming the restore is good, delete the old volume:

```bash
aws ec2 delete-volume --volume-id $VOLUME_ID
```

## AWS / Environment variable reference

See [src/.env.example](src/.env.example) for the canonical list. AWS-specific guidance:

| Var | Required? | AWS value |
|---|---|---|
| `IMAGE_NAME`, `IMAGE_TAG`, `CONTAINER_NAME`, `PORT` | **yes** | Compose interpolation |
| `APP_FOLDER_PATH` | **yes** | `/appdata` |
| `DB_ROOT_DIR` | **yes** | `/appdata` |
| `TMPDIR` | recommended | `/appdata/tmp` |
| `JWT_SECRET_KEY` | **yes** | `openssl rand -base64 32` per tenant |
| `S3_OUTPUT_BUCKET` | **yes** | Your S3 bucket name |
| `S3_ENDPOINT_URL` | **leave empty** | AWS S3 default |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | **leave empty** | Use the EC2 instance role |
| `AWS_REGION` | **yes** | Real region (e.g. `ap-south-1`, `us-east-1`) |
| `SUPERADMIN_EMAIL` | **yes** | For mutating user-limit endpoints |
| `DEFAULT_USER_EMAIL` / `..._FIRST_NAME` / `..._LAST_NAME` | **yes** | Seeded user |
| `GOOGLE_CLIENT_ID` | yes for OAuth | OAuth client ID for sign-in |
| `OPENROUTER_API_KEY` | yes for evaluators | Used by `POST /evaluators/{uuid}/invoke` |
| `OPENAI_API_KEY` and other provider keys | depends | Only what the tenant uses |
| `CORS_ALLOWED_ORIGINS` | recommended | **Frontend origin(s)**, comma-separated (e.g. `https://app.tenant.example.com`). NOT the backend URL — CORS gates browser-tab origins, not the API itself |
| `DOCS_USERNAME` / `DOCS_PASSWORD` | recommended | HTTP Basic auth on `/docs` |
| `MAX_CONCURRENT_JOBS` etc. | optional | Tune for instance size |
| `SENTRY_DSN` | recommended | Background-thread failures route through this |

> **When adding/changing/removing env vars:** update [src/.env.example](src/.env.example), [docker-compose.yml](docker-compose.yml), and the deploy workflows together. See `.cursor/rules/env-var.md`.
