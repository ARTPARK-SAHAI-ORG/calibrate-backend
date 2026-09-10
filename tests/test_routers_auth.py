"""Tests for the auth router's email-driven flows.

Covers:
  - signup and first Google login send a welcome email
  - a second Google login does not
  - forgot-password emails a reset link and stays silent about unknown emails
  - reset-password accepts the emailed token and rejects bad input
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import main as main_mod

    with patch("main.recover_pending_jobs"):
        with TestClient(main_mod.app) as c:
            yield c


@pytest.fixture
def sent(monkeypatch):
    """Collect emails instead of sending them."""
    import routers.auth as auth_mod

    messages = []
    monkeypatch.setattr(
        auth_mod,
        "send_email",
        lambda to, subject, html: messages.append(
            {"to": to, "subject": subject, "html": html}
        ),
    )
    return messages


def _email(prefix: str = "auth") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}@example.com"


def _signup(client, email: str, password: str = "passw0rd"):
    return client.post(
        "/auth/signup",
        json={
            "first_name": "A",
            "last_name": "U",
            "email": email,
            "password": password,
        },
    )


def test_signup_sends_one_welcome_email(client, sent):
    email = _email("signup")
    assert _signup(client, email).status_code == 200

    assert len(sent) == 1
    assert sent[0]["to"] == email
    assert sent[0]["subject"] == "Welcome to Calibrate"


def test_google_login_welcomes_only_the_first_time(client, sent):
    import routers.auth as auth_mod

    email = _email("google")

    async def _fake_verify(id_token: str) -> dict:
        return {"email": email, "given_name": "G", "family_name": "U"}

    with patch.object(auth_mod, "verify_google_token", _fake_verify):
        assert client.post("/auth/google", json={"id_token": "x"}).status_code == 200
        assert len(sent) == 1
        assert sent[0]["to"] == email

        assert client.post("/auth/google", json={"id_token": "x"}).status_code == 200
        assert len(sent) == 1


def test_google_login_without_email_is_rejected(client, sent):
    import routers.auth as auth_mod

    async def _fake_verify(id_token: str) -> dict:
        return {}

    with patch.object(auth_mod, "verify_google_token", _fake_verify):
        resp = client.post("/auth/google", json={"id_token": "x"})

    assert resp.status_code == 400
    assert sent == []


def test_forgot_password_emails_a_link_with_the_token(client, sent):
    email = _email("forgot")
    _signup(client, email)
    sent.clear()

    resp = client.post("/auth/forgot-password", json={"email": email})

    assert resp.status_code == 200
    assert len(sent) == 1
    assert sent[0]["to"] == email
    assert "/reset-password?token=" in sent[0]["html"]


def test_forgot_password_for_unknown_email_looks_identical(client, sent):
    known = _email("known")
    _signup(client, known)
    sent.clear()

    real = client.post("/auth/forgot-password", json={"email": known})
    sent.clear()
    unknown = client.post("/auth/forgot-password", json={"email": _email("nobody")})

    assert unknown.status_code == real.status_code
    assert unknown.json() == real.json()
    assert sent == []


def test_reset_password_with_the_emailed_token_changes_the_login(client, sent):
    email = _email("reset")
    _signup(client, email)
    sent.clear()

    client.post("/auth/forgot-password", json={"email": email})
    token = sent[0]["html"].split("token=")[1].split('"')[0]

    resp = client.post(
        "/auth/reset-password", json={"token": token, "password": "newpassw0rd"}
    )
    assert resp.status_code == 200

    assert (
        client.post(
            "/auth/login", json={"email": email, "password": "newpassw0rd"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/auth/login", json={"email": email, "password": "passw0rd"}
        ).status_code
        == 401
    )


def test_reset_password_with_a_bad_token_is_rejected(client):
    resp = client.post(
        "/auth/reset-password", json={"token": "nope", "password": "newpassw0rd"}
    )
    assert resp.status_code == 400


def test_reset_password_rejects_a_short_password(client):
    resp = client.post("/auth/reset-password", json={"token": "nope", "password": "abc"})
    assert resp.status_code == 422


def test_google_login_does_not_welcome_someone_already_invited(client, sent):
    """An invite already emailed them, so their first sign-in is not a new account."""
    import db
    import routers.auth as auth_mod

    inviter = _signup(client, _email("inviter")).json()
    org_uuid = client.get(
        "/organizations",
        headers={"Authorization": f"Bearer {inviter['access_token']}"},
    ).json()[0]["uuid"]

    invited = _email("invited")
    db.add_organization_member(org_uuid, invited)
    sent.clear()

    async def _fake_verify(id_token: str) -> dict:
        return {"email": invited, "given_name": "I", "family_name": "U"}

    with patch.object(auth_mod, "verify_google_token", _fake_verify):
        assert client.post("/auth/google", json={"id_token": "x"}).status_code == 200

    assert sent == []


def test_google_login_without_a_name_welcomes_only_once(client, sent):
    """Google sends no name unless the profile scope was asked for."""
    import routers.auth as auth_mod

    email = _email("noname")

    async def _fake_verify(id_token: str) -> dict:
        return {"email": email}

    with patch.object(auth_mod, "verify_google_token", _fake_verify):
        for _ in range(3):
            assert client.post("/auth/google", json={"id_token": "x"}).status_code == 200

    assert [m["to"] for m in sent] == [email]


def test_forgot_password_emails_the_tidied_up_address(client, sent):
    """A stray space or capital letter must not reach the mail provider."""
    email = _email("padded")
    _signup(client, email)
    sent.clear()

    resp = client.post(
        "/auth/forgot-password", json={"email": f"  {email.upper()}  "}
    )

    assert resp.status_code == 200
    assert [m["to"] for m in sent] == [email]


def test_forgot_password_for_a_blank_address_sends_nothing(client, sent):
    """A row with an empty email can exist as the seeded default user."""
    resp = client.post("/auth/forgot-password", json={"email": "   "})

    assert resp.status_code == 200
    assert sent == []


def test_reset_password_rejects_a_token_used_in_the_same_instant(client, sent, monkeypatch):
    """The write re-checks the token, in case it was spent since the first check."""
    import routers.auth as auth_mod

    email = _email("race")
    _signup(client, email)
    sent.clear()
    client.post("/auth/forgot-password", json={"email": email})
    token = sent[0]["html"].split("token=")[1].split('"')[0]

    monkeypatch.setattr(auth_mod, "reset_password_with_token", lambda *a: False)
    resp = client.post(
        "/auth/reset-password", json={"token": token, "password": "newpassw0rd"}
    )

    assert resp.status_code == 400
