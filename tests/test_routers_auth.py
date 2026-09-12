"""Tests for the auth router's email-driven flows.

Covers:
  - signup and first Google login send a welcome email
  - a second Google login does not
  - a Google login by someone already invited does not
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
        lambda to, template, variables: messages.append(
            {"to": to, "template": template, "variables": variables}
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

    import routers.auth as auth_mod

    assert len(sent) == 1
    assert sent[0]["to"] == email
    assert sent[0]["template"] == auth_mod.WELCOME_TEMPLATE
    assert sent[0]["variables"]["NAME"] == "A"
    assert sent[0]["variables"]["URL"]


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

def test_signup_does_not_welcome_someone_already_invited(client, sent):
    """An invite already emailed them, so finishing signup is not a new account.
    Same rule as the Google path."""
    import db

    inviter = _signup(client, _email("inviter2")).json()
    org_uuid = client.get(
        "/organizations",
        headers={"Authorization": f"Bearer {inviter['access_token']}"},
    ).json()[0]["uuid"]

    invited = _email("invited2")
    db.add_organization_member(org_uuid, invited)
    sent.clear()

    assert _signup(client, invited).status_code == 200
    assert sent == []
