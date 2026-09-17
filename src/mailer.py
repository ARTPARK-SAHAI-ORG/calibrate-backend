"""Transactional email over Resend's HTTP API.

Named `mailer` rather than `email`: `src/` is the working directory, so a
module named `email.py` would shadow the stdlib package of that name.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import httpx

from utils import capture_exception_to_sentry, env_str

logger = logging.getLogger(__name__)

RESEND_ENDPOINT = "https://api.resend.com/emails"
SEND_TIMEOUT_SECONDS = 10

WELCOME_TEMPLATE = "calibrate-welcome"
WORKSPACE_INVITE_TEMPLATE = "calibrate-workspace-invite"

# Bounded, and its threads are joined at interpreter exit, so an unauthenticated
# endpoint cannot spawn threads without limit and a shutdown does not drop a
# message mid-send.
_SENDER = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mailer")


def _strip_markup(value: str) -> str:
    # Resend does not escape variable values, and one value can land in an HTML
    # body or a plain-text subject, where HTML escaping would show as `&amp;`.
    return str(value).replace("<", "").replace(">", "")


def send_email(to: str, template: str, variables: dict[str, str]) -> None:
    """Send one email, off the request thread so a slow provider never delays a response."""
    api_key = env_str("RESEND_API_KEY", "")
    if not api_key:
        logger.info("RESEND_API_KEY unset, email disabled: not sending to %s", to)
        return

    sender = env_str("EMAIL_FROM", "Calibrate <onboarding@resend.dev>")
    safe = {key: _strip_markup(value) for key, value in variables.items()}

    def _send() -> None:
        try:
            response = httpx.post(
                RESEND_ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "from": sender,
                    "to": [to],
                    "template": {"id": template, "variables": safe},
                },
                timeout=SEND_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.error("Failed to send email to %s: %s", to, exc)
            capture_exception_to_sentry(exc)

    _SENDER.submit(_send)


def frontend_url() -> str:
    return env_str("FRONTEND_URL", "http://localhost:3000").rstrip("/")
