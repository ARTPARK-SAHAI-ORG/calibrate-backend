"""Provider API keys a workspace sets for itself.

A run hands the raw key to the `calibrate-agent` CLI through the environment, so
these are encrypted rather than hashed the way `api_keys` are — they have to be
readable again. Missing or unreadable values fall back to the server's own keys
rather than failing the run, so one bad `PROVIDER_KEY_ENCRYPTION_KEY` cannot take
every workspace down.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from cryptography.fernet import Fernet, InvalidToken

from db import get_org_provider_keys
from provider_status import PROVIDER_ENV_VARS, available_provider_names
from utils import capture_exception_to_sentry

logger = logging.getLogger(__name__)

ENCRYPTION_KEY_ENV = "PROVIDER_KEY_ENCRYPTION_KEY"

# Google Cloud is the only provider whose variables are not all secrets: the
# project is an identifier worth showing back in full, and the credentials hold
# the service-account JSON itself rather than the path the CLI expects, so a run
# has to materialize it on disk first.
PLAIN_ENV_VARS = frozenset({"GOOGLE_CLOUD_PROJECT_ID"})
FILE_ENV_VARS = frozenset({"GOOGLE_APPLICATION_CREDENTIALS"})

SECRET = "secret"
PLAIN = "plain"
FILE = "file"

# Linux caps one environment variable at 128 KB, and every value here is handed
# to the eval tool that way. Refuse a paste that would break every run instead.
MAX_VALUE_BYTES = 64 * 1024

# Google Cloud's own rule: 6 to 30 characters, starting with a lowercase letter,
# lowercase letters, digits and hyphens, not ending in a hyphen.
_PROJECT_ID = re.compile(r"[a-z][a-z0-9-]{4,28}[a-z0-9]")


class ProviderKeyError(ValueError):
    """Caller-facing problem with a submitted value (bad provider, bad JSON)."""


def env_var_kind(env_var: str) -> str:
    if env_var in FILE_ENV_VARS:
        return FILE
    if env_var in PLAIN_ENV_VARS:
        return PLAIN
    return SECRET


def provider_env_vars(provider: str) -> List[str]:
    """Every environment variable a provider needs, in registry order."""
    try:
        return list(PROVIDER_ENV_VARS[provider])
    except KeyError:
        raise ProviderKeyError(f"Unknown provider '{provider}'")


def _fernet() -> Optional[Fernet]:
    raw = os.getenv(ENCRYPTION_KEY_ENV)
    if not raw:
        return None
    try:
        return Fernet(raw.encode())
    except (ValueError, TypeError) as exc:
        logger.error(
            "%s is set but not a valid Fernet key; workspace provider keys are "
            "unusable until it is fixed: %s",
            ENCRYPTION_KEY_ENV,
            exc,
        )
        return None


def encryption_available() -> bool:
    return _fernet() is not None


def encrypt_value(value: str) -> str:
    fernet = _fernet()
    if fernet is None:
        raise ProviderKeyError(
            f"{ENCRYPTION_KEY_ENV} is not configured on this server, so provider "
            "keys cannot be stored"
        )
    return fernet.encrypt(value.encode()).decode()


_decrypt_failure_reported = False


def _report_decrypt_failure(exc: Exception) -> None:
    """Send the first unreadable key to Sentry, then stay quiet.

    `capture_exception_to_sentry` blocks on a flush, and this runs once per
    stored row on request paths, so reporting every row would add seconds to
    every request and flood Sentry with one event per row per request. A lost
    or rotated key produces identical events, so the first one says everything
    the rest would.
    """
    global _decrypt_failure_reported
    if _decrypt_failure_reported:
        return
    _decrypt_failure_reported = True
    capture_exception_to_sentry(exc)


def _decrypt_value(encrypted: str) -> Optional[str]:
    """Decrypted value, or None when it cannot be read.

    Unreadable means the encryption key was rotated or lost. Callers treat that
    as "this workspace has no key for that variable" and use the server's.
    """
    fernet = _fernet()
    if fernet is None:
        return None
    try:
        return fernet.decrypt(encrypted.encode()).decode()
    except (InvalidToken, ValueError) as exc:
        logger.error("Could not decrypt a stored provider key: %s", exc)
        _report_decrypt_failure(exc)
        return None


def _display_for(env_var: str, value: str) -> str:
    """What the settings screen shows once the raw value is gone."""
    kind = env_var_kind(env_var)
    if kind == PLAIN:
        return value
    if kind == FILE:
        # The service account's own address identifies which account is wired up,
        # which is the only part of the JSON worth showing back.
        try:
            email = json.loads(value).get("client_email")
        except (ValueError, AttributeError):
            email = None
        if not isinstance(email, str) or not email:
            return "Service account JSON saved"
        return email
    return f"••••{value[-4:]}" if len(value) >= 4 else "••••"


def validate_values(provider: str, values: Dict[str, str]) -> Dict[str, str]:
    """Check a submitted set of values for one provider and return it stripped.

    Every variable the provider needs must be present and non-empty, so a
    half-configured Google Cloud cannot be saved. Credentials JSON must parse,
    because a malformed file only surfaces as a failed run much later.
    """
    required = provider_env_vars(provider)
    unknown = sorted(set(values) - set(required))
    if unknown:
        raise ProviderKeyError(
            f"{provider} does not use {', '.join(unknown)}. It uses "
            f"{', '.join(required)}"
        )

    cleaned: Dict[str, str] = {}
    missing: List[str] = []
    for env_var in required:
        value = (values.get(env_var) or "").strip()
        if not value:
            missing.append(env_var)
            continue
        if len(value.encode()) > MAX_VALUE_BYTES:
            raise ProviderKeyError(
                f"{env_var} is too long. A run passes it to the eval tool as an "
                "environment variable, which the operating system caps"
            )
        if env_var_kind(env_var) == FILE:
            try:
                parsed = json.loads(value)
            except ValueError:
                raise ProviderKeyError(f"{env_var} must be the service account JSON")
            if not isinstance(parsed, dict):
                raise ProviderKeyError(f"{env_var} must be the service account JSON")
        if env_var == "GOOGLE_CLOUD_PROJECT_ID" and not _PROJECT_ID.fullmatch(value):
            # This field is shown back in full, so a key pasted into it by mistake
            # would be on screen in the clear. A project id has a fixed shape.
            raise ProviderKeyError(
                f"{env_var} must be a Google Cloud project id, such as my-project-123"
            )
        cleaned[env_var] = value
    if missing:
        raise ProviderKeyError(
            f"{provider} also needs {', '.join(missing)}. Send every value at once"
        )
    return cleaned


def encrypted_rows_for(provider: str, values: Dict[str, str]) -> List[Dict[str, str]]:
    """Validated values → the rows the storage layer writes."""
    return [
        {
            "env_var": env_var,
            "value_encrypted": encrypt_value(value),
            "display": _display_for(env_var, value),
        }
        for env_var, value in validate_values(provider, values).items()
    ]


def workspace_env_values(org_uuid: str) -> Dict[str, str]:
    """Decrypted `{env_var: value}` a workspace has set. Unreadable rows are dropped."""
    values: Dict[str, str] = {}
    for row in get_org_provider_keys(org_uuid):
        value = _decrypt_value(row["value_encrypted"])
        if value is not None:
            values[row["env_var"]] = value
    return values


def covered_providers(org_uuid: str) -> Set[str]:
    """Providers the workspace has fully configured with its own keys."""
    values = workspace_env_values(org_uuid)
    return {
        provider
        for provider, env_vars in PROVIDER_ENV_VARS.items()
        if all(values.get(var) for var in env_vars)
    }


def available_providers_for_org(org_uuid: str) -> List[str]:
    """Providers usable by this workspace: its own keys, plus the server's."""
    covered = covered_providers(org_uuid)
    names = list(available_provider_names())
    names += [p for p in PROVIDER_ENV_VARS if p in covered and p not in names]
    return names


def provider_env(org_uuid: str, workdir: Any) -> Optional[Dict[str, str]]:
    """Environment for a run: the server's, with the workspace's values on top.

    Returns None when the workspace has set nothing, so callers pass `env=None`
    and the subprocess inherits this process's environment exactly as before.
    Credentials JSON is written into `workdir`, which every caller deletes when
    the run ends, so the file never outlives the run that needs it.
    """
    values = workspace_env_values(org_uuid) if org_uuid else {}
    if not values:
        return None

    env = dict(os.environ)
    for provider, env_vars in PROVIDER_ENV_VARS.items():
        # All of a provider's values move together or none do. Google Cloud
        # otherwise pairs the server's service account with the workspace's
        # project, which fails in a way nobody can diagnose.
        if not all(values.get(var) for var in env_vars):
            continue
        overrides: Optional[Dict[str, str]] = {}
        for env_var in env_vars:
            if env_var_kind(env_var) != FILE:
                overrides[env_var] = values[env_var]
                continue
            path = Path(workdir) / f"{env_var.lower()}.json"
            try:
                path.write_text(values[env_var])
                os.chmod(path, 0o600)
            except OSError as exc:
                logger.error(
                    "Could not write %s, falling back to the server's %s keys: %s",
                    env_var,
                    provider,
                    exc,
                )
                capture_exception_to_sentry(exc)
                overrides = None
                break
            overrides[env_var] = str(path)
        if overrides is not None:
            env.update(overrides)
    return env
