#!/usr/bin/env python3
"""Create and publish Calibrate's Resend email templates.

Run directly (`uv run python scripts/create_email_templates.py`) with a
full-access `RESEND_API_KEY`. An alias that already exists is left alone, so an
edit made in the Resend dashboard is never clobbered; pass `--force` to
overwrite the templates with the wording below.

`APP_URL` sets the address the logo and the fallback links point at.
"""

from __future__ import annotations

import os
import sys

import httpx

API_ROOT = "https://api.resend.com/templates"
TIMEOUT_SECONDS = 30

APP_URL = (os.environ.get("APP_URL") or "https://calibrate.artpark.ai").rstrip("/")
LOGO_URL = f"{APP_URL}/logo.png"

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"


def layout(
    heading: str,
    button: str,
    sub: str = "",
    note: str = "",
    items: list[tuple[str, str]] | None = None,
    hero: str = "",
    button_url: str = "{{{URL}}}",
) -> str:
    """One centred card: logo, heading, optional sub-line, button, optional note.

    Tables and inline styles throughout — Outlook ignores stylesheets and
    several clients strip a <style> block entirely.
    """
    sub_row = (
        '<tr><td align="center" style="font-size:15px;line-height:1.5;'
        f'color:#6b6b66;padding-top:12px;">{sub}</td></tr>'
        if sub
        else ""
    )
    note_row = (
        '<tr><td height="34" style="font-size:0;line-height:0;">&nbsp;</td></tr>'
        '<tr><td align="center" style="font-size:13px;line-height:1.5;'
        'color:#8a8a84;padding-top:26px;border-top:1px solid #eeeeea;">'
        f"{note}</td></tr>"
        if note
        else ""
    )
    rows = "".join(
        '<tr><td style="padding-top:18px;font-size:14px;line-height:1.55;'
        f'color:#4a4a45;"><strong style="color:#111111;">{title}</strong><br>{text}</td></tr>'
        for title, text in (items or [])
    )
    items_block = (
        '<tr><td style="padding-top:30px;border-top:1px solid #eeeeea;">'
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0">{rows}</table>'
        "</td></tr>"
        if rows
        else ""
    )
    return (
        '<table width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="background:#ffffff;padding:40px 16px;font-family:{FONT};">'
        '<tr><td align="center">'
        '<table width="600" cellpadding="0" cellspacing="0" border="0" '
        'style="max-width:600px;border:1px solid #e6e6e2;border-radius:14px;'
        'padding:44px 40px;">'
        '<tr><td align="center" style="padding-bottom:26px;">'
        f'<img src="{LOGO_URL}" width="24" height="24" alt="Calibrate" '
        'style="display:block;border:0;outline:none;text-decoration:none;"></td></tr>'
        '<tr><td align="center" style="font-size:24px;line-height:1.35;'
        f'font-weight:600;color:#111111;letter-spacing:-0.01em;">{heading}</td></tr>'
        f"{sub_row}"
        + (f'<tr><td align="center" style="padding-top:24px;">{hero}</td></tr>' if hero else "")
        + '<tr><td align="center" style="padding-top:22px;">'
        '<table cellpadding="0" cellspacing="0" border="0"><tr>'
        '<td bgcolor="#111111" style="border-radius:8px;">'
        f'<a href="{button_url}" style="display:inline-block;padding:13px 30px;'
        f'font-family:{FONT};font-size:15px;font-weight:500;color:#ffffff;'
        f'text-decoration:none;">{button}</a>'
        "</td></tr></table></td></tr>"
        f"{items_block}"
        f"{note_row}"
        "</table></td></tr></table>"
    )


SKILLS_DOC = "https://docs.calibrate.artpark.ai/agents/skills"
LEARN = f"{APP_URL}/learn"
WHATSAPP = "https://chat.whatsapp.com/JygDNcZ943a3VmZDXYMg5Z"


def link(label: str, url: str) -> str:
    """A quiet text link. No underline: nothing else in the email has one."""
    return f'<a href="{url}" style="color:#6b6b66;text-decoration:none;">{label}</a>'


def textlink(label: str, url: str) -> str:
    """An inline link. Colour carries it, so nothing needs an underline."""
    return (
        f'<a href="{url}" style="color:#0e7a55;font-weight:500;'
        f'text-decoration:none;">{label}</a>'
    )


def code(command: str) -> str:
    """A command the reader copies by hand. Email has no scripting, so no copy button."""
    return (
        '<table width="100%" cellpadding="0" cellspacing="0" border="0" '
        'style="margin-top:12px;"><tr>'
        '<td bgcolor="#f4f4f1" style="border:1px solid #e4e4de;border-radius:8px;'
        'padding:12px 14px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'
        'monospace;font-size:12.5px;line-height:1.5;color:#111111;'
        f'word-break:break-word;">{command}</td>'
        "</tr></table>"
    )


def pill(label: str, url: str) -> str:
    """A small secondary button, so a link never has to be underlined."""
    return (
        '<table cellpadding="0" cellspacing="0" border="0" '
        'style="display:inline-block;margin:12px 8px 0 0;"><tr>'
        '<td bgcolor="#f4f4f1" style="border-radius:8px;border:1px solid #e4e4de;">'
        f'<a href="{url}" style="display:inline-block;padding:9px 16px;'
        f'font-family:{FONT};font-size:13.5px;font-weight:500;color:#111111;'
        f'text-decoration:none;">{label}</a>'
        "</td></tr></table>"
    )


def thumbnail(youtube_id: str) -> str:
    thumb = f"https://img.youtube.com/vi/{youtube_id}/maxresdefault.jpg"
    return (
        f'<a href="https://youtu.be/{youtube_id}" style="text-decoration:none;">'
        f'<img src="{thumb}" width="520" alt="" '
        'style="display:block;width:100%;max-width:520px;height:auto;border:0;'
        'border-radius:10px;"></a>'
    )


def video(youtube_id: str, caption: str) -> str:
    """A clickable thumbnail. Email clients strip iframes, so a real embed
    cannot play; the thumbnail links out to YouTube instead."""
    url = f"https://youtu.be/{youtube_id}"
    thumb = f"https://img.youtube.com/vi/{youtube_id}/maxresdefault.jpg"
    return (
        f'<a href="{url}" style="text-decoration:none;">'
        f'<img src="{thumb}" width="520" alt="" '
        'style="display:block;width:100%;max-width:520px;height:auto;border:0;'
        'border-radius:10px;margin:12px 0 0;"></a>'
        + '<table width="100%" cellpadding="0" cellspacing="0" border="0"><tr>'
        + '<td align="center">'
        + pill(f"&#9654;&nbsp;{caption}", url)
        + "</td></tr></table>"
    )


URL_VAR = {"key": "URL", "type": "string", "fallback": APP_URL}
WORKSPACE_VAR = {"key": "WORKSPACE", "type": "string", "fallback": "a workspace"}
INVITER_VAR = {"key": "INVITER", "type": "string", "fallback": "Someone"}

TEMPLATES = [
    {
        "alias": "calibrate-welcome",
        "name": "Calibrate welcome",
        "subject": "Welcome to Calibrate",
        "html": layout(
            heading="Welcome, {{{NAME}}}",
            sub=(
                "Calibrate is AI agent evaluation built for non-profits. It gives "
                "you a simple and repeatable way to find errors, so you can deploy "
                "changes confidently without breaking what already works."
            ),
            hero=thumbnail("F1oR8QlCnmI"),
            button="See how Calibrate works",
            button_url="https://youtu.be/F1oR8QlCnmI",
            items=[
                (
                    "Every failure in one place",
                    "Every mistake you find becomes a test. One list that keeps "
                    "growing, instead of living in someone's head or a spreadsheet.",
                ),
                (
                    "Release changes without breaking what worked",
                    "Before you ship a change, run every test again and see which "
                    "ones stopped passing.",
                ),
                (
                    "Catch failures before users do",
                    "Keep running the tests on live traffic, so you find the "
                    "mistakes instead of waiting for a user to report them.",
                ),
                (
                    "Domain experts take the lead",
                    "In sensitive domains like health, education and agriculture, "
                    "only your domain experts can say whether an answer is good. "
                    "Calibrate lets them define that and own the evaluation, "
                    "without taking up engineering bandwidth.",
                ),
                (
                    "Turn your AI tool into an eval agent",
                    "You can easily connect Calibrate to your AI tool (Claude Code, "
                    "Codex, Cursor) to convert it into your personal eval agent, "
                    "helping you analyse mistakes, collect human labels, and "
                    "automatically align LLM judges. Watch the video below to see it "
                    "in action."
                    + video("Vx3oxYKbLVw", "Watch the video")
                    + "<br><br>Install the skills with one command:"
                    + code("npx skills add dalmia/calibrate-skills "
                           "--agent claude-code -g")
                    + '<span style="display:inline-block;padding-top:10px;">'
                    + "Swap <strong>claude-code</strong> for <strong>cursor</strong> "
                    + "or <strong>codex</strong>. Full setup "
                    + textlink("here", SKILLS_DOC)
                    + ".</span>",
                ),
                (
                    "Ask us anything",
                    "Our community answers faster than we do. "
                    + textlink("Join the WhatsApp group", WHATSAPP)
                    + ".",
                ),
            ],
            note=f'Learn more about Calibrate {textlink("here", LEARN)}',
        ),
        "variables": [
            {"key": "NAME", "type": "string", "fallback": "there"},
            URL_VAR,
        ],
    },
    {
        "alias": "calibrate-workspace-invite",
        "name": "Calibrate workspace invite",
        "subject": "You are invited to {{{WORKSPACE}}} on Calibrate",
        "html": layout(
            heading="You have been invited to join {{{WORKSPACE}}} on Calibrate",
            sub="Invited by {{{INVITER}}}",
            button="Accept Invite",
        ),
        "variables": [WORKSPACE_VAR, INVITER_VAR, URL_VAR],
    },
]


def publish(headers: dict, template_id: str, alias: str, what: str) -> bool:
    response = httpx.post(
        f"{API_ROOT}/{template_id}/publish", headers=headers, timeout=TIMEOUT_SECONDS
    )
    if response.status_code >= 400:
        print(f"{alias}: {what} but publishing failed ({response.text})")
        return False
    print(f"{alias}: {what} and published")
    return True


def existing_ids(headers: dict) -> dict:
    response = httpx.get(API_ROOT, headers=headers, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return {t.get("alias"): t.get("id") for t in response.json().get("data", [])}


def main() -> int:
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        print("RESEND_API_KEY is not set. Use a full-access key, not a sending key.")
        return 1
    force = "--force" in sys.argv

    headers = {"Authorization": f"Bearer {api_key}"}
    for template in TEMPLATES:
        alias = template["alias"]
        response = httpx.post(
            API_ROOT, headers=headers, json=template, timeout=TIMEOUT_SECONDS
        )
        if response.status_code == 401:
            print(
                f"Resend refused the key while creating {alias}: {response.text}. "
                "A send-only key answers 401 restricted_api_key. Use a full-access key."
            )
            return 1
        body = response.text.lower()
        if response.status_code == 409 or "already in use" in body or "exist" in body:
            if not force:
                print(f"{alias}: already exists, left as it is")
                continue
            template_id = existing_ids(headers).get(alias)
            if template_id is None:
                print(f"{alias}: exists but could not be found to update")
                return 1
            update = httpx.patch(
                f"{API_ROOT}/{template_id}",
                headers=headers,
                json=template,
                timeout=TIMEOUT_SECONDS,
            )
            if update.status_code >= 400:
                print(f"{alias}: could not be updated ({update.text})")
                return 1
            if not publish(headers, template_id, alias, "overwritten"):
                return 1
            continue
        if response.status_code >= 400:
            print(
                f"{alias}: could not be created ({response.status_code} {response.text})"
            )
            return 1

        if not publish(headers, response.json()["id"], alias, "created"):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
