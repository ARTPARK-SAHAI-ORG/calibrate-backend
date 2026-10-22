"""Tests for the auth router's email-driven flows.

Covers:
  - the welcome email on signup and on a first Google login
  - confirming the email at signup when email is on, and the links it sends
  - password reset
  - the Google token checks
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


def test_signup_cannot_take_over_a_google_account(client, sent):
    """A Google account has no password. Signing up with its email must not
    set one, or anyone could log in as that person."""
    import db

    email = _email("google")
    # What a Google sign in leaves behind: a named row, confirmed, no password
    victim_uuid = db.create_user("Victim", "User", email)
    db.mark_email_verified(victim_uuid)

    assert _signup(client, email, password="attacker1").status_code == 409
    assert db.get_user(victim_uuid)["password_hash"] is None
    login = client.post(
        "/auth/login", json={"email": email, "password": "attacker1"}
    )
    assert login.status_code == 401


# ---------------------------------------------------------------------------
# Email verification and password reset (on only when RESEND_API_KEY is set)
# ---------------------------------------------------------------------------

from urllib.parse import parse_qs, urlparse  # noqa: E402


@pytest.fixture
def email_on(monkeypatch, sent):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    return sent


def _link_params(message) -> dict:
    query = parse_qs(urlparse(message["variables"]["URL"]).query)
    return {key: values[0] for key, values in query.items()}


def _backdate(email: str, column: str, seconds: int) -> None:
    import db

    with db.get_db_connection() as conn:
        conn.execute(
            f"UPDATE users SET {column} = datetime('now', ?) WHERE email = ?",
            (f"-{seconds} seconds", email),
        )
        conn.commit()


def _signup_next(client, email, password="passw0rd", next_path=None):
    body = {"first_name": "A", "last_name": "U", "email": email, "password": password}
    if next_path is not None:
        body["next"] = next_path
    return client.post("/auth/signup", json=body)


def _login(client, email, password="passw0rd"):
    return client.post("/auth/login", json={"email": email, "password": password})


def test_signup_with_email_on_sends_a_link_and_no_token(client, email_on):
    import routers.auth as auth_mod

    email = _email("verify")
    resp = _signup(client, email.upper())

    assert resp.status_code == 202
    assert resp.json() == {
        "status": "verification_sent",
        "email": email,
        "already_pending": False,
    }
    assert "access_token" not in resp.json()
    assert len(email_on) == 1
    assert email_on[0]["to"] == email
    assert email_on[0]["template"] == auth_mod.VERIFY_EMAIL_TEMPLATE
    assert email_on[0]["variables"]["NAME"] == "A"
    url = email_on[0]["variables"]["URL"]
    assert url.startswith(f"{auth_mod.frontend_url()}/verify-email?token=")
    assert "callbackUrl" not in url


def test_login_before_confirming_is_refused_and_resends(client, email_on):
    email = _email("unconfirmed")
    _signup(client, email)
    _backdate(email, "verify_sent_at", 61)
    first_token = _link_params(email_on[0])["token"]

    resp = _login(client, email)
    assert resp.status_code == 403
    assert resp.json()["detail"] == "email_not_verified"
    assert len(email_on) == 2
    assert _link_params(email_on[1])["token"] != first_token

    assert _login(client, email, "wrong-pass").status_code == 401
    assert len(email_on) == 2


def test_resend_waits_a_minute(client, email_on):
    email = _email("cooldown")
    assert _signup(client, email).status_code == 202
    token = _link_params(email_on[0])["token"]

    again = _signup(client, email, password="other-pass")
    assert again.status_code == 202
    assert _login(client, email).status_code == 403
    assert len(email_on) == 1

    # The first link still works, with the first password.
    resp = client.post("/auth/verify-email", json={"token": token})
    assert resp.status_code == 200
    assert _login(client, email).status_code == 200
    assert _login(client, email, "other-pass").status_code == 401


def test_verify_email_logs_in_and_the_password_then_works(client, email_on):
    import routers.auth as auth_mod

    email = _email("confirm")
    _signup(client, email)
    token = _link_params(email_on[0])["token"]

    resp = client.post("/auth/verify-email", json={"token": token})
    assert resp.status_code == 200
    body = resp.json()
    assert body["access_token"]
    assert body["user"]["email"] == email
    assert email_on[-1]["template"] == auth_mod.WELCOME_TEMPLATE

    assert _login(client, email).status_code == 200
    assert client.post("/auth/verify-email", json={"token": token}).status_code == 400


def test_signup_over_a_confirmed_account_is_refused(client, email_on):
    import db

    email = _email("taken")
    db.create_user_with_password("Real", "Owner", email, "hash$x")
    resp = _signup(client, email)
    assert resp.status_code == 409
    assert resp.json()["detail"] == "Email already taken"
    assert email_on == []


def test_signup_over_an_invite_needs_the_link_before_the_workspace(client, email_on):
    import db

    owner = db.create_user("Own", "Er", _email("owner"))
    org_uuid = db.create_organization("Shared team", owner)
    invited = _email("invitee")
    db.add_organization_member(org_uuid, invited)

    assert _signup(client, invited).status_code == 202
    assert _login(client, invited).status_code == 403

    token = _link_params(email_on[0])["token"]
    resp = client.post("/auth/verify-email", json={"token": token})
    assert resp.status_code == 200
    orgs = client.get(
        "/organizations",
        headers={"Authorization": f"Bearer {resp.json()['access_token']}"},
    ).json()
    assert org_uuid in [o["uuid"] for o in orgs]
    # The invite already emailed them, so no welcome.
    assert len(email_on) == 1


def test_second_signup_before_confirming_replaces_the_link(client, email_on):
    email = _email("again")
    _signup(client, email)
    old_token = _link_params(email_on[0])["token"]
    _backdate(email, "verify_sent_at", 61)

    resp = _signup(client, email, password="newer-pass")
    assert resp.status_code == 202
    assert resp.json()["already_pending"] is True
    new_token = _link_params(email_on[1])["token"]

    assert client.post("/auth/verify-email", json={"token": old_token}).status_code == 400
    assert client.post("/auth/verify-email", json={"token": new_token}).status_code == 200
    assert _login(client, email, "newer-pass").status_code == 200


def test_forgot_password_for_an_unknown_email_sends_nothing(client, email_on):
    resp = client.post("/auth/forgot-password", json={"email": _email("nobody")})
    assert resp.status_code == 202
    assert resp.json() == {"status": "sent"}
    assert email_on == []


def test_forgot_password_skips_a_bare_invite(client, email_on):
    import db

    owner = db.create_user("Own", "Er", _email("owner2"))
    org_uuid = db.create_organization("Team two", owner)
    invited = _email("stub")
    db.add_organization_member(org_uuid, invited)

    assert client.post("/auth/forgot-password", json={"email": invited}).status_code == 202
    assert email_on == []


def test_forgot_password_with_email_off_sends_nothing(client, sent):
    email = _email("off")
    assert _signup(client, email).status_code == 200
    sent.clear()
    assert client.post("/auth/forgot-password", json={"email": email}).status_code == 202
    assert sent == []


def test_reset_password_sets_the_new_password_and_logs_in(client, email_on):
    import db
    import routers.auth as auth_mod

    email = _email("reset")
    db.create_user_with_password("Re", "Set", email, auth_mod._hash_password("old-pass"))

    resp = client.post("/auth/forgot-password", json={"email": email, "next": "/agents"})
    assert resp.status_code == 202
    assert len(email_on) == 1
    assert email_on[0]["template"] == auth_mod.RESET_PASSWORD_TEMPLATE
    params = _link_params(email_on[0])
    assert email_on[0]["variables"]["URL"].startswith(
        f"{auth_mod.frontend_url()}/reset-password?token="
    )
    assert params["callbackUrl"] == "/agents"
    assert params["email"] == email

    assert client.post(
        "/auth/reset-password", json={"token": params["token"], "password": "short"}
    ).status_code == 422

    resp = client.post(
        "/auth/reset-password", json={"token": params["token"], "password": "new-pass"}
    )
    assert resp.status_code == 200
    assert resp.json()["access_token"]
    assert _login(client, email, "new-pass").status_code == 200
    assert _login(client, email, "old-pass").status_code == 401
    assert client.post(
        "/auth/reset-password", json={"token": params["token"], "password": "again-pass"}
    ).status_code == 400


def test_reset_confirms_a_pending_signup(client, email_on):
    email = _email("pendreset")
    _signup(client, email)
    verify_token = _link_params(email_on[0])["token"]

    client.post("/auth/forgot-password", json={"email": email})
    reset_token = _link_params(email_on[1])["token"]
    resp = client.post(
        "/auth/reset-password", json={"token": reset_token, "password": "chosen-pass"}
    )
    assert resp.status_code == 200
    assert _login(client, email, "chosen-pass").status_code == 200
    assert client.post("/auth/verify-email", json={"token": verify_token}).status_code == 400


def _verify_link(client, email, email_on) -> str:
    _signup(client, email)
    return _link_params(email_on[-1])["token"]


def _reset_link(client, email, email_on) -> str:
    import db

    db.create_user_with_password("Re", "Set", email, "hash$x")
    client.post("/auth/forgot-password", json={"email": email})
    return _link_params(email_on[-1])["token"]


@pytest.mark.parametrize(
    "make_link, column, endpoint, lifetime, expired_detail",
    [
        (_verify_link, "verify_sent_at", "/auth/verify-email", 24 * 3600, "expired_token"),
        (_reset_link, "reset_sent_at", "/auth/reset-password", 3600, "invalid_or_expired_token"),
    ],
)
@pytest.mark.parametrize("age_offset, works", [(-60, True), (1, False)])
def test_link_works_until_the_time_the_email_states(
    client, email_on, make_link, column, endpoint, lifetime, expired_detail, age_offset, works
):
    """The emails promise 24 hours and 1 hour: a link a minute short of that
    still works, and one a second past it does not."""
    email = _email("age")
    token = make_link(client, email, email_on)
    _backdate(email, column, lifetime + age_offset)

    resp = client.post(endpoint, json={"token": token, "password": "new-pass"})
    if works:
        assert resp.status_code == 200, resp.text
    else:
        assert resp.status_code == 400
        assert resp.json()["detail"] == expired_detail


def test_an_expired_link_says_so_and_can_ask_for_a_new_one(client, email_on):
    email = _email("resend")
    token = _verify_link(client, email, email_on)
    assert _link_params(email_on[-1])["email"] == email
    _backdate(email, "verify_sent_at", 24 * 3600 + 1)

    resp = client.post("/auth/verify-email", json={"token": token, "email": email})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "expired_token"

    email_on.clear()
    resp = client.post(
        "/auth/resend-verification", json={"email": email, "next": "/invite/tok"}
    )
    assert resp.status_code == 202, resp.text
    assert resp.json() == {"status": "sent", "email": email}
    assert len(email_on) == 1
    params = _link_params(email_on[0])
    assert params["callbackUrl"] == "/invite/tok"
    assert client.post("/auth/verify-email", json={"token": params["token"]}).status_code == 200


def test_a_replaced_link_can_still_ask_for_a_new_one(client, email_on):
    email = _email("replaced")
    old = _verify_link(client, email, email_on)
    _backdate(email, "verify_sent_at", 61)
    _signup(client, email)  # a newer link replaces the old one

    resp = client.post("/auth/verify-email", json={"token": old, "email": email})
    assert resp.json()["detail"] == "expired_token"
    _backdate(email, "verify_sent_at", 61)
    email_on.clear()
    assert client.post("/auth/resend-verification", json={"email": email}).status_code == 202
    assert len(email_on) == 1


def test_a_used_link_says_the_email_is_already_verified(client, email_on):
    email = _email("used")
    token = _verify_link(client, email, email_on)
    assert client.post("/auth/verify-email", json={"token": token}).status_code == 200

    resp = client.post("/auth/verify-email", json={"token": token, "email": email})
    assert resp.json()["detail"] == "already_verified"
    # Nothing is waiting, so asking for a new link sends nothing
    email_on.clear()
    assert client.post("/auth/resend-verification", json={"email": email}).status_code == 202
    assert email_on == []


def test_resend_for_an_unknown_address_sends_nothing(client, email_on):
    resp = client.post("/auth/resend-verification", json={"email": _email("nobody")})
    assert resp.status_code == 202
    assert email_on == []


def test_a_reset_link_can_be_checked_without_using_it(client, email_on):
    email = _email("check")
    token = _reset_link(client, email, email_on)

    assert client.post("/auth/check-reset-link", json={"token": token}).json() == {"valid": True}
    # Checking does not use it up
    assert client.post(
        "/auth/reset-password", json={"token": token, "password": "new-pass"}
    ).status_code == 200
    resp = client.post("/auth/check-reset-link", json={"token": token})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "invalid_or_expired_token"


def test_an_expired_reset_link_fails_the_check(client, email_on):
    email = _email("checkold")
    token = _reset_link(client, email, email_on)
    _backdate(email, "reset_sent_at", 3601)
    assert client.post("/auth/check-reset-link", json={"token": token}).status_code == 400


def test_a_made_up_link_is_refused(client, email_on):
    assert client.post("/auth/verify-email", json={"token": "made-up"}).status_code == 400


def test_a_signup_left_waiting_can_sign_up_again_once_email_is_off(client, sent, monkeypatch):
    """Turning email off must not lock out someone whose link never came: signing
    up again works the way it does for anyone else while email is off."""
    email = _email("stuck")
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    assert _signup(client, email).status_code == 202
    monkeypatch.setenv("RESEND_API_KEY", "")

    resp = _signup(client, email, password="fresh-pass")
    assert resp.status_code == 200, resp.text
    assert _login(client, email, "fresh-pass").status_code == 200


@pytest.mark.parametrize("endpoint", ["/auth/signup", "/auth/login"])
def test_a_password_over_72_bytes_is_a_message_not_a_server_error(client, sent, endpoint):
    email = _email("long")
    if endpoint == "/auth/login":
        assert _signup(client, email).status_code == 200  # so login checks a real password
    body = {"first_name": "A", "last_name": "U", "email": email, "password": "é" * 40}
    resp = client.post(endpoint, json=body)
    assert resp.status_code == (422 if endpoint == "/auth/signup" else 401), resp.text


def test_first_google_sign_in_after_a_strangers_signup_is_welcomed(client, email_on):
    """Someone else started a signup with this address and never confirmed it. The
    owner's first Google sign in is still their first, so they get the welcome."""
    email = _email("gwelcome")
    _signup(client, email)
    email_on.clear()
    with patch("routers.auth.verify_google_token", return_value={
        "email": email, "given_name": "Real", "family_name": "Owner"
    }):
        assert client.post("/auth/google", json={"id_token": "t"}).status_code == 200
    assert [m["template"] for m in email_on] == ["calibrate-welcome"]


@pytest.mark.parametrize(
    "next_path, expected",
    [
        ("/agents?tab=1", "/agents?tab=1"),
        ("//evil.example.com", None),
        ("/\\evil.example.com", None),
        ("https://evil.example.com", None),
        ("", None),
    ],
)
def test_callback_is_added_only_for_a_path_on_the_site(client, email_on, next_path, expected):
    _signup_next(client, _email("next"), next_path=next_path)
    assert _link_params(email_on[0]).get("callbackUrl") == expected


def _tokeninfo(info: dict):
    from unittest.mock import AsyncMock, MagicMock

    class FakeResp:
        status_code = 200
        text = "ok"

        def json(self):
            return info

    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=FakeResp())
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)
    return patch("routers.auth.httpx.AsyncClient", return_value=fake_client)


@pytest.mark.parametrize(
    "client_id, extra",
    [
        ("our-client", {"aud": "someone-else", "email_verified": "true"}),
        ("our-client", {"aud": "our-client", "email_verified": "false"}),
        ("our-client", {"aud": "our-client"}),
        ("", {"aud": "", "email_verified": "true"}),
    ],
)
def test_google_token_must_be_ours_and_verified(client, sent, monkeypatch, client_id, extra):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", client_id)
    email = _email("gaud")
    with _tokeninfo({"email": email, **extra}):
        resp = client.post("/auth/google", json={"id_token": "x"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid Google token"
    import db

    assert db.get_user_by_email(email) is None


def test_google_token_with_our_audience_logs_in(client, sent, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "our-client")
    email = _email("gok")
    with _tokeninfo({"email": email, "aud": "our-client", "email_verified": True}):
        resp = client.post("/auth/google", json={"id_token": "x"})
    assert resp.status_code == 200
    import db

    assert db.get_user_by_email(email)["email_verified_at"]


def test_google_login_voids_a_pending_signup(client, email_on):
    import db
    import routers.auth as auth_mod

    email = _email("gpending")
    _signup(client, email, password="attacker1")
    token = _link_params(email_on[0])["token"]

    async def _fake_verify(id_token: str) -> dict:
        return {"email": email, "given_name": "Real", "family_name": "Owner"}

    with patch.object(auth_mod, "verify_google_token", _fake_verify):
        assert client.post("/auth/google", json={"id_token": "x"}).status_code == 200

    assert client.post("/auth/verify-email", json={"token": token}).status_code == 400
    assert _login(client, email, "attacker1").status_code == 401
    row = db.get_user_by_email(email)
    assert row["password_hash"] is None
    assert row["verify_password_hash"] is None
