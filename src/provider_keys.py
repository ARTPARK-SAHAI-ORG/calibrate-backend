import asyncio
import json
import os
import tempfile
from typing import Dict, Optional

from cryptography.fernet import Fernet, InvalidToken

import db
from provider_status import PROVIDER_ENV_VARS, parse_provider_status_stdout
from utils import env_bool, get_calibrate_agent_cli

CHECK_TIMEOUT_SECONDS = 45
PROVIDER_NAMES = {
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "deepgram": "Deepgram",
    "elevenlabs": "ElevenLabs",
    "sarvam": "Sarvam",
    "cartesia": "Cartesia",
    "smallest": "Smallest AI",
    "groq": "Groq",
    "gemini": "Gemini",
    "google": "Google Cloud",
}
UNREADABLE_KEYS_ERROR = (
    "Your saved provider keys cannot be read. Add them again in Workspace settings."
)


def _fernet() -> Fernet:
    return Fernet(os.environ["PROVIDER_KEYS_ENCRYPTION_KEY"].encode())


def encryption_configured() -> bool:
    return bool(os.getenv("PROVIDER_KEYS_ENCRYPTION_KEY"))


def encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def decrypt(token: str) -> str:
    return _fernet().decrypt(token.encode()).decode()


def parse_google_credentials(value: str) -> Optional[dict]:
    """The service-account JSON, or None when it is not one."""
    try:
        creds = json.loads(value)
    except ValueError:
        return None
    if not isinstance(creds, dict) or not creds.get("project_id") or not creds.get("client_email"):
        return None
    return creds


def masked_value(provider: str, value: str) -> str:
    if provider == "google":
        return parse_google_credentials(value)["client_email"]
    return "••••" + value[-4:]


def org_uses_own_keys(org_uuid: str) -> bool:
    return db.org_has_provider_keys(org_uuid)


def google_creds_path(org_uuid: str) -> str:
    return os.path.join(tempfile.gettempdir(), "calibrate-provider-creds", f"{org_uuid}.json")


def _write_private_file(path: str, text: str) -> None:
    folder = os.path.dirname(path)
    os.makedirs(folder, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _env_without_provider_keys() -> Dict[str, str]:
    env = dict(os.environ)
    for names in PROVIDER_ENV_VARS.values():
        for name in names:
            env.pop(name, None)
    return env


def _add_provider(env: Dict[str, str], provider: str, value: str, google_path: str) -> None:
    if provider == "google":
        _write_private_file(google_path, value)
        env["GOOGLE_APPLICATION_CREDENTIALS"] = google_path
        env["GOOGLE_CLOUD_PROJECT_ID"] = json.loads(value)["project_id"]
    else:
        env[PROVIDER_ENV_VARS[provider][0]] = value


def provider_env(org_uuid: str) -> Optional[Dict[str, str]]:
    """Environment for a calibrate-agent run of this org, or None to inherit the server's."""
    rows = db.list_org_provider_keys(org_uuid)
    if not rows:
        return None
    env = _env_without_provider_keys()
    for row in rows:
        try:
            value = decrypt(row["encrypted_value"])
        except (InvalidToken, KeyError, ValueError) as exc:
            raise RuntimeError(UNREADABLE_KEYS_ERROR) from exc
        _add_provider(env, row["provider"], value, google_creds_path(org_uuid))
    return env


async def check_provider_key(provider: str, value: str) -> Optional[str]:
    """Run calibrate-agent's own provider check with only this key. None when it works."""
    if env_bool("FAKE_AI_PROVIDERS", False):
        return None
    # calibrate-agent loads a .env from its working folder upward with override,
    # so it runs from an empty folder or the server's own keys would leak in.
    with tempfile.TemporaryDirectory() as cwd:
        env = _env_without_provider_keys()
        _add_provider(env, provider, value, os.path.join(cwd, "google.json"))
        process = await asyncio.create_subprocess_exec(
            get_calibrate_agent_cli(),
            "status",
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), CHECK_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return f"No answer from {provider} within {CHECK_TIMEOUT_SECONDS} seconds."
    try:
        result = parse_provider_status_stdout(stdout.decode()).get(provider) or {}
    except ValueError:
        result = {}
    if result.get("status") == "pass":
        return None
    error = result.get("error") or ""
    if error.startswith(("HTTP 401", "HTTP 403")):
        return f"{PROVIDER_NAMES.get(provider, provider)} did not accept this key. Check that you copied the full API key."
    return error or f"Could not check the {provider} key."
