# Self-Hosting Guide

This guide walks you through self-hosting Calibrate's Backend on your infra. The self-hosting guide for the frontend can be found [here](https://github.com/ARTPARK-SAHAI-ORG/calibrate-frontend/blob/main/SELF_HOSTING.md).

Pick your target cloud (AWS/GCP) and follow that section. Before you start, make sure to `fork` this repo to your team's Github account.

## Contents

1. [What you need](#what-you-need)
2. [How the backend and frontend connect](#how-the-backend-and-frontend-connect)
3. [Release an update](#release-an-update)
4. [Deploy on AWS](#deploy-on-aws)
5. [Deploy on GCP](#deploy-on-gcp)

## What you need

A typical setup runs the backend on one GCP virtual machine and the frontend on Vercel. Have these ready before you start:

- **Your own copies of both repos** on GitHub: `calibrate-backend` and `calibrate-frontend`, forked into your organisation.
- **A GCP project** with billing on. The backend uses one Compute Engine machine, one persistent disk for the database, and one Cloud Storage bucket for audio and results.
- **A Vercel account** connected to your copy of the frontend repo.
- **Two web addresses**, for example `calibrate.yourdomain.com` for the frontend and `calibrate-api.yourdomain.com` for the backend, and access to their DNS settings. The backend address points at the machine's static IP. The frontend address is added in Vercel.
- **A Google OAuth client** for sign-in with Google. One client is shared by both: the frontend uses its ID and secret, the backend uses its ID.
- **A Resend account with a verified sending domain**, if you want email. Without it, signup works but no email is sent, so people cannot verify an address or reset a password. See [Set up the email templates](#set-up-the-email-templates).
- **API keys for the AI providers** you will evaluate with, such as OpenRouter or OpenAI. Each workspace can also add its own keys in the app.
- Optional: a Sentry project for error reports.

## How the backend and frontend connect

These values must agree, or sign-in, uploads, or email links break:

| Where                         | Setting                            | Value                                             |
| ----------------------------- | ---------------------------------- | ------------------------------------------------- |
| Vercel (frontend)             | `NEXT_PUBLIC_BACKEND_URL`          | The backend address, e.g. `https://calibrate-api.yourdomain.com` |
| Vercel (frontend)             | `AUTH_URL`, `NEXT_PUBLIC_APP_URL`  | The frontend address                              |
| Backend `.env`                | `CORS_ALLOWED_ORIGINS`             | The frontend address                              |
| Backend `.env`                | `FRONTEND_URL`                     | The frontend address, used for links in emails    |
| Backend `.env` and Vercel     | `GOOGLE_CLIENT_ID`                 | The same OAuth client ID in both                  |
| Google OAuth client           | Authorised origins and redirect URI | The frontend address, and `<frontend address>/api/auth/callback/google` |
| Storage bucket CORS           | Allowed origin                     | The frontend address, since browsers upload audio straight to the bucket |

## Release an update

**Backend.** Pull the latest changes from the upstream repo into your copy on GitHub. Then publish a GitHub release (**Releases → Draft a new release**, new tag such as `v1.2.0`). Publishing the release runs **Deploy to Production**, which builds the image and restarts the backend on your machine. To restart without new code, for example after changing a secret, run **Actions → Redeploy to Production**. It can also roll back to an earlier release.

**Frontend.** Pull the latest changes from upstream into your copy on GitHub. Vercel deploys it on its own.

Update both together. A new frontend can call backend routes an older backend does not have.

# Deploy on AWS

End-to-end walkthrough on AWS (EC2 + S3). Substitute `<region>` with your AWS region (e.g. `ap-south-1`, `us-east-1`).

> The existing AWS production deploy is fully automated by [.github/workflows/deploy.yml](.github/workflows/deploy.yml). For a brand-new tenant on AWS, you provision the infra once (steps 1–7 below), then add a GitHub Actions environment with your secrets and trigger that workflow for subsequent deploys. The first-time provisioning is the part this section walks through.

## Create the S3 bucket

An S3 bucket is used to store the results of all your evals and media files. Create one and name it `calibrate-backend-artifacts`. Ensure to block all public access for the S3 bucket as the app uses presigned URLs for client access.

Update the CORS permissions for your bucket:

```
[
    {
        "AllowedHeaders": [
            "*"
        ],
        "AllowedMethods": [
            "GET",
            "PUT",
            "POST"
        ],
        "AllowedOrigins": [
            "*"
        ],
        "ExposeHeaders": []
    }
]
```

Keep it stricter if you need to.

## Create an IAM role for the EC2 instance

Go to **IAM → Roles → Create role**.

1. **Trusted entity**: AWS service → **EC2**.
2. **Permissions**: skip attaching managed policies for now, click Next, name it `calibrate-backend-ec2`, create.
3. Open the role → **Add permissions → Create inline policy** → JSON tab. Paste an S3 policy scoped to `calibrate-backend-artifacts` (Get/Put/Delete/ListBucket on the bucket and `/*`). Name it `calibrate-bucket-access`.

## Launch an EC2 instance

Attach the `calibrate-backend-ec2` role to the instance.

Update the **Inbound rules** for the security group attached to the instance.

- HTTP (80), Source `0.0.0.0/0`
- HTTPS (443), Source `0.0.0.0/0`

Increase the size of the volume attached to the EC2 instance. 50 GB is good enough to begin. Increase it later as usage increases.

Allocate and associate an Elastic IP with the instance too.

## SSH and create the root directory for the database

Calibrate uses SQLite as the DB. The db file is stored on the root volume attached to the EC2.

```
ssh -i <your-key>.pem ubuntu@<ELASTIC_IP>
sudo mkdir /appdata
sudo chown -R $USER /appdata
```

The rest of the steps need to be done within the instance (not your laptop).

## 7. Install Docker

```
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
docker run --rm hello-world
```

## Clone the repo and build the image

```bash
sudo apt-get update && sudo apt-get install -y git    # or: sudo dnf install -y git
git clone https://github.com/<your-org>/calibrate-backend.git
cd calibrate-backend
docker build -t calibrate-backend:local .
```

Takes 5–15 minutes the first time.

## Create the `.env` file

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

# Auth — generate fresh, do NOT reuse from any other tenant
JWT_SECRET_KEY=PASTE_OUTPUT_OF_openssl_rand_-base64_32
JWT_EXPIRATION_HOURS=168

# Object storage (AWS S3 — IAM instance role provides creds)
OBJECT_STORAGE_MODE=s3
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
CORS_ALLOWED_ORIGINS=https://<your frontend address>

# Concurrency
MAX_CONCURRENT_JOBS=1
MAX_CONCURRENT_JOBS_PER_USER=1
DEFAULT_MAX_ROWS_PER_EVAL=20
DEFAULT_MAX_TRACES=50000

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
GOOGLE_APPLICATION_CREDENTIALS=
GOOGLE_CLOUD_PROJECT_ID=

# Auth
GOOGLE_CLIENT_ID=

# Encrypts the AI provider keys a workspace saves in the app. Generate once, never change.
PROVIDER_KEYS_ENCRYPTION_KEY=

# Email. Leave RESEND_API_KEY empty to send nothing.
# FRONTEND_URL is the address of your frontend, used for links inside emails.
RESEND_API_KEY=
EMAIL_FROM=Calibrate <onboarding@resend.dev>
FRONTEND_URL=http://localhost:3000

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

# Generate PROVIDER_KEYS_ENCRYPTION_KEY and paste into .env
docker run --rm calibrate-backend:local uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Update the default values given above as needed. For example, you might want to set the `SUPERADMIN_EMAIL` to an email address you own. Set the API keys for different providers (e.g. OpenRouter, OpenAI, etc.). Set the `GOOGLE_CLIENT_ID` to the same value as the one used for self-hosting the frontend. Set `FRONTEND_URL` to the address your frontend runs on, or the links in invite emails will point at localhost. If you want email at all, you also have to create the Resend templates once: see [Set up the email templates](#set-up-the-email-templates).

If you used a different name other than `/appdata` in the `SSH and create the root directory for the database` step, update `APP_FOLDER_PATH` and `DB_ROOT_DIR` in the `.env` file accordingly.

Refer to [ENV.md](./ENV.md) for the full list of environment variables and their description.


## Set up the email templates

The wording of every email lives in Resend, not in this repo, so you can edit it in Resend's dashboard without a deploy. A fresh Resend account has none of them, and an email whose template is missing is never sent.

Create them once, with a **full access** API key (the key you deploy with only needs to send):

```bash
RESEND_API_KEY=<full access key> uv run python scripts/create_email_templates.py
```

That creates and publishes six templates: the welcome email, the workspace invite, the two emails telling a workspace owner someone joined, email verification, and password reset. Running it again changes nothing, so it will not overwrite wording you have since edited in the dashboard.

To change any of the wording afterwards, edit the template in Resend and press Publish. A draft does not go out until you publish it.

## Start the app

```bash
docker compose up -d
docker compose logs -f
```

Watch for `Uvicorn running on http://0.0.0.0:8000`. Ctrl-C the log tail.

## Verify it works

Open `http://<INSTANCE_ELASTIC_IP>:8000/docs` on your browser. It should load the FastAPI docs for the server.

## Move from HTTP to HTTPS

Use nginx to route your custom domain (e.g. calibrate-backend.<yourdomain.com>) to the server. Use certbot to make the connection secure using HTTPS. Set `client_max_body_size 50m;` in the nginx server block. The nginx default of 1 MB refuses audio uploads and run imports sent through the API.

## Verify everything works

Open `https://<YOUR_DOMAIN>/docs` on your browser. It should load the FastAPI docs for the server.

If you frontend is set up, [create a new speech-to-text dataset](https://docs.calibrate.artpark.ai/core-concepts/speech-to-text#create-a-dataset) and upload one audio. If it uploads successfully, your S3 connection works.

## Set up EBS snapshots

The SQLite DB lives on the EBS volume. If the volume gets corrupted, accidentally deleted, or the AZ goes down, you lose all tenant state. Schedule daily snapshots using the Data Lifecycle Manager or AWS Backup.

## Automate deployments with GitHub Actions

The repo ships with [.github/workflows/deploy.yml](.github/workflows/deploy.yml) and [.github/workflows/deploy-staging.yml](.github/workflows/deploy-staging.yml). Both build the Docker image, push to GitHub Container Registry (ghcr.io), SSH onto the VM, and run `docker compose pull && up -d`.

### 1. SSH key the workflow can use

You already have one from launching the EC2 instance (`<your-key>.pem`). Use that, or generate a dedicated key:

```bash
# On your laptop — generate a dedicated key (don't reuse your personal one)
ssh-keygen -t ed25519 -f ~/.ssh/calibrate-deploy -C "calibrate-deploy" -N ""

# Append the public key to the EC2 instance's authorized_keys
ssh -i <your-key>.pem ubuntu@<ELASTIC_IP> \
  "echo '$(cat ~/.ssh/calibrate-deploy.pub)' >> ~/.ssh/authorized_keys"
```

Test from your laptop: `ssh -i ~/.ssh/calibrate-deploy ubuntu@<ELASTIC_IP>` should log you in.

### 2. Set GitHub Actions environment secrets

Go to **GitHub → Repo settings → Environments → New environment** named `Production` (or `Staging`). Create new secrets for the environment variables you set in the `.env` file along with the following. On every deploy the workflow starts the backend with these secrets, not with the `.env` file on the machine, so every value in your `.env` needs a matching secret.

| Secret                                        | Value                                                                       |
| --------------------------------------------- | --------------------------------------------------------------------------- |
| `VM_HOST`                                     | The EC2 instance's Elastic IP                                               |
| `VM_USER`                                     | `ubuntu` (or `ec2-user` for Amazon Linux)                                   |
| `VM_SSH_KEY`                                  | Contents of the private key (`<your-key>.pem` or `~/.ssh/calibrate-deploy`) |
| Provider API keys (`OPENROUTER_API_KEY` etc.) | As needed                                                                   |
| `SENTRY_*`, `LANGFUSE_*`, etc.                | Optional                                                                    |

Leave `S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` empty (or unset) — the IAM instance role on the EC2 provides S3 credentials automatically.

### 3. Trigger a deploy

Publish a GitHub release (**Releases → Draft a new release**, new tag, Publish). That runs **Deploy to Production**. **Deploy to Staging** is started by hand from **Actions → Deploy to Staging → Run workflow**.

Watch the run. If SSH connects and `docker compose up -d` succeeds, you're set. The workflow logs the machine in to ghcr.io with its own short-lived token, so the machine needs no GitHub token of its own.

### Common gotchas

- **Security group**: port 22 must be open to the GitHub Actions runner. Either keep it open to `0.0.0.0/0` or restrict to GitHub's runner IP ranges (changes periodically — keep an eye on [GitHub's meta endpoint](https://api.github.com/meta)).
- **Compose project name**: the workflow uses `docker compose -p pense-production` for project isolation. If you started the container manually with a different project name, the workflow will spin up a parallel container instead of replacing yours — stop the old one with `docker compose down` once the workflow is green.

# Deploy on GCP

End-to-end walkthrough on Google Cloud (Compute Engine + GCS). Substitute `<project-id>` with your GCP project ID and `<region>`/`<zone>` with your target (e.g. `us-central1` / `us-central1-a`).

## Create the GCS bucket

A GCS bucket is used to store the results of all your evals and media files. The codebase talks to it via the S3 protocol (boto3 with a custom endpoint), so HMAC keys are required.

Go to **Cloud Storage → Buckets → Create** and step through the wizard:

1. **Name**: `calibrate-backend-artifacts`.
2. **Location type**: **Region** (not the default Multi-region). **Location**: same region you'll use for the VM.
3. **Storage class**: Standard. Leave Hierarchical namespace and Rapid Cache unchecked.
4. **Access control**: Uniform. **Public access prevention**: enforced.
5. **Protection**: keep Soft delete on (default 7 days). Enable Object versioning. Leave Retention off. Encryption: default.

Click **Create**.

Browsers upload audio straight to the bucket, so the bucket needs CORS. Save this as `cors.json`, with your frontend address:

```json
[
  {
    "origin": ["https://<your frontend address>"],
    "method": ["GET", "PUT", "POST"],
    "responseHeader": ["Content-Type"],
    "maxAgeSeconds": 3600
  }
]
```

Then apply it:

```bash
gcloud storage buckets update gs://calibrate-backend-artifacts --cors-file=cors.json
```

## Create a service account + HMAC keys

Go to **IAM & Admin → Service Accounts → Create service account**.

1. **Name**: `calibrate-backend-storage`. Create.
2. Open the bucket create above → **Permissions → Grant access** → add the service account with role **Storage Object Admin** (scoped to just this bucket).
3. Go to **Cloud Storage → Settings → Interoperability** tab → under "Access keys for service accounts" → **Create a key for a service account** → pick `calibrate-backend-storage`.
4. Copy the **Access key** and **Secret** — the secret is shown only once. You'll paste these into `.env` later.

## Verify firewall rules

GCP firewalls live on the network and attach to instances via **network tags**. The default network usually has `default-allow-http` and `default-allow-https` pre-created.

Go to **VPC network → Firewall** and confirm both rules exist (target tags `http-server` and `https-server`, source `0.0.0.0/0`). If missing, create them with the same target tags.

## Create a persistent disk

This will be attached to the VM and the SQLite DB file will live on it.

Go to **Compute Engine → Disks → Create disk**.

1. **Name**: `calibrate-appdata`. **Type**: Balanced persistent disk. **Size**: 100 GB.
2. **Region/Zone**: pick the zone where the VM will live — must match, since disks can't cross zones.
3. **Source type**: Blank disk. Create.

Leave it unformatted; formatting happens on the VM after attach.

## 5. Create the VM

Create a new Compute Engine instance. Keep the following in mind:

1. **Machine configuration**: **E2** series, `e2-standard-4` (4 vCPU, 16 GB). Cheapest option that handles the default `MAX_CONCURRENT_JOBS=1` with headroom. Bump to `e2-standard-8` if you raise concurrency or run heavy benchmarks.
2. **Networking → Network tags**: add `http-server` and `https-server`.
3. **Advanced → Disks → Attach existing disk**: pick `calibrate-appdata`, set **Device name** to `appdata`.
4. **Advanced → Backups → Snapshot schedules**: Set up daily backups.

## Reserve and attach a static IP

Go to **VPC network → IP addresses → Reserve external static IP address**.

1. **Name**: `calibrate-backend-ip`. **Region**: match the VM. **Attached to**: `calibrate-backend`. Reserve.
2. Note the IP shown — that's your stable public address.

## SSH in and prepare the disk

From your laptop:

```bash
gcloud compute ssh calibrate-backend --zone=<zone>
```

Inside the VM, format (first time only) and mount the persistent disk:

```bash
# Check if already formatted (re-run safety)
sudo file -sL /dev/disk/by-id/google-appdata
# If output says "data" → blank, run mkfs. If "ext4 filesystem" → skip mkfs.

sudo mkfs.ext4 -F /dev/disk/by-id/google-appdata

sudo mkdir -p /appdata
echo '/dev/disk/by-id/google-appdata /appdata ext4 discard,defaults 0 2' | sudo tee -a /etc/fstab
sudo mount /appdata
sudo chown -R $USER /appdata

df -h /appdata    # Expect ~98G on /appdata, not /dev/root
```

If `df` shows `/dev/root`, the mount failed and writes to `/appdata` are silently going to the boot disk — fix before continuing.

## Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
docker run --rm hello-world
```

If the hello-world test prints `permission denied`, log out and back in.

## Clone the repo and build the image

```bash
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/<your-org>/calibrate-backend.git
cd calibrate-backend
docker build -t calibrate-backend:local .
```

Takes 5–15 minutes the first time.

## Create the `.env` file

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

# Auth — generate fresh, do NOT reuse from any other tenant
JWT_SECRET_KEY=PASTE_OUTPUT_OF_openssl_rand_-base64_32
JWT_EXPIRATION_HOURS=168

# Object storage (GCS via S3 interop)
OBJECT_STORAGE_MODE=s3
S3_ENDPOINT_URL=https://storage.googleapis.com
S3_OUTPUT_BUCKET=calibrate-backend-artifacts
AWS_ACCESS_KEY_ID=<HMAC access key from step 2>
AWS_SECRET_ACCESS_KEY=<HMAC secret from step 2>
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
CORS_ALLOWED_ORIGINS=https://<your frontend address>

# Concurrency
MAX_CONCURRENT_JOBS=1
MAX_CONCURRENT_JOBS_PER_USER=1
DEFAULT_MAX_ROWS_PER_EVAL=20
DEFAULT_MAX_TRACES=50000

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
GOOGLE_APPLICATION_CREDENTIALS=
GOOGLE_CLOUD_PROJECT_ID=

# Auth
GOOGLE_CLIENT_ID=

# Encrypts the AI provider keys a workspace saves in the app. Generate once, never change.
PROVIDER_KEYS_ENCRYPTION_KEY=

# Email. Leave RESEND_API_KEY empty to send nothing.
# FRONTEND_URL is the address of your frontend, used for links inside emails.
RESEND_API_KEY=
EMAIL_FROM=Calibrate <onboarding@resend.dev>
FRONTEND_URL=http://localhost:3000

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

# Generate PROVIDER_KEYS_ENCRYPTION_KEY and paste into .env
docker run --rm calibrate-backend:local uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Update the default values given above as needed. For example, you might want to set the `SUPERADMIN_EMAIL` to an email address you own. Set the API keys for different providers (e.g. OpenRouter, OpenAI, etc.). Set the `GOOGLE_CLIENT_ID` to the same value as the one used for self-hosting the frontend. Set `FRONTEND_URL` to the address your frontend runs on, or the links in invite emails will point at localhost. If you want email at all, you also have to create the Resend templates once: see [Set up the email templates](#set-up-the-email-templates).

`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` need to be set as the HMAC secret and access keys from step 2. Let `S3_ENDPOINT_URL` be as it is.

If you used a different name other than `/appdata` in the `SSH and create the root directory for the database` step, update `APP_FOLDER_PATH` and `DB_ROOT_DIR` in the `.env` file accordingly.

Refer to [ENV.md](./ENV.md) for the full list of environment variables and their description.

## Start the app

```bash
docker compose up -d
docker compose logs -f
```

Watch for `Uvicorn running on http://0.0.0.0:8000`. Ctrl-C the log tail.

## Verify it works

Open `http://<STATIC_IP>/docs` on your browser. It should load the FastAPI docs.

## Moving from HTTP to HTTPS

Use nginx to route your custom domain (e.g. calibrate-backend.<yourdomain.com>) to the server. Use certbot to make the connection secure using HTTPS. Set `client_max_body_size 50m;` in the nginx server block. The nginx default of 1 MB refuses audio uploads and run imports sent through the API.

## Verify everything works

Open `https://<YOUR_DOMAIN>/docs` on your browser. It should load the FastAPI docs.

If your frontend is set up, [create a new speech-to-text dataset](https://docs.calibrate.artpark.ai/core-concepts/speech-to-text#create-a-dataset) and upload one audio. If it uploads successfully, your GCS connection works.

## Automate deployments with GitHub Actions

The repo ships with [.github/workflows/deploy.yml](.github/workflows/deploy.yml) and [.github/workflows/deploy-staging.yml](.github/workflows/deploy-staging.yml). Both build the Docker image, push to GitHub Container Registry (ghcr.io), SSH onto the VM, and run `docker compose pull && up -d`. They're cloud-neutral — works for GCE the same as EC2.

### 1. Add an SSH key the workflow can use

GCE defaults to OS Login (Google-identity-based SSH). GitHub Actions can't use that directly, so add a project-wide SSH key the workflow can authenticate with:

```bash
# On your laptop — generate a dedicated key (don't reuse your personal one)
ssh-keygen -t ed25519 -f ~/.ssh/calibrate_key_name.pub -C "calibrate_key_name" -N ""

# Disable OS Login on the VM (uses metadata-based SSH keys instead)
gcloud compute instances add-metadata calibrate-backend-prod \
  --zone=<zone> --metadata=enable-oslogin=FALSE

# Add the public key to the instance
gcloud compute instances add-metadata calibrate-backend-prod \
  --zone=<zone> \
  --metadata="ssh-keys=<username_on_instance>:$(cat ~/.ssh/calibrate_key_name.pub)"
```

Test from your laptop: `ssh -i ~/.ssh/calibrate-deploy ubuntu@<STATIC_IP>` should log you in.

### 2. Set GitHub Actions environment secrets

Go to **GitHub → Repo settings → Environments → New environment** named `Production` (or `Staging`). Create new secrets for the environment variables you set in the `.env` file along with the following. On every deploy the workflow starts the backend with these secrets, not with the `.env` file on the machine, so every value in your `.env` needs a matching secret.

| Secret                                        | Value                                                       |
| --------------------------------------------- | ----------------------------------------------------------- |
| `VM_HOST`                                     | The VM's static IP                                          |
| `VM_USER`                                     | `ubuntu` (or whatever username matches the SSH key)         |
| `VM_SSH_KEY`                                  | Contents of `~/.ssh/calibrate-deploy` (the **private** key) |
| Provider API keys (`OPENROUTER_API_KEY` etc.) | As needed                                                   |
| `SENTRY_*`, `LANGFUSE_*`, etc.                | Optional                                                    |

### 3. Trigger a deploy

Publish a GitHub release (**Releases → Draft a new release**, new tag, Publish). That runs **Deploy to Production**. **Deploy to Staging** is started by hand from **Actions → Deploy to Staging → Run workflow**.

Watch the run. If SSH connects and `docker compose up -d` succeeds, you're set. The workflow logs the machine in to ghcr.io with its own short-lived token, so the machine needs no GitHub token of its own.

### Common gotchas

- **Firewall**: port 22 must be open to the GitHub Actions runner. Either keep `default-allow-ssh` as-is (0.0.0.0/0) or restrict to GitHub's runner IP ranges (changes periodically — keep an eye on [GitHub's meta endpoint](https://api.github.com/meta)).
- **Compose project name**: the workflow uses `docker compose -p pense-production` for project isolation. If you started the container manually with a different project name, the workflow will spin up a parallel container instead of replacing yours — stop the old one with `docker compose down` once the workflow is green.

## Point the Calibrate CLI at your instance

The `calibrate` CLI defaults to the hosted backend. To use it against your self-hosted instance, set the server URL **once** — it persists to `~/.config/calibrate/config.yaml` next to your API key:

```
calibrate configure --no-interactive --server-url https://calibrate-backend.<yourdomain.com>
```

Interactive `calibrate configure` also prompts for it. Verify with `calibrate whoami` (shows the resolved URL and its source). Resolution is flag > env > config, so you can still override per-call with `--server-url …` or `export CALIBRATE_SERVER_URL=https://…`. Create the API key under Workspace settings → API keys on your instance.
