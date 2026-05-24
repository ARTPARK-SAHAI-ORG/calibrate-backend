"""Additional coverage for src/auth_utils.py."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt

import auth_utils
from auth_utils import (
    JWT_ALGORITHM,
    create_access_token,
    get_current_user_id,
    get_optional_user_id,
    require_superadmin,
)


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_get_current_user_id_valid(monkeypatch):
    monkeypatch.setattr(auth_utils, "JWT_SECRET_KEY", "test-secret-key-for-unit-tests")
    monkeypatch.setattr(auth_utils, "JWT_EXPIRATION_HOURS", 1)
    token = create_access_token("uid-1", "user@example.com")
    out = asyncio.run(get_current_user_id(_creds(token)))
    assert out == "uid-1"


def test_get_current_user_id_invalid_token(monkeypatch):
    monkeypatch.setattr(auth_utils, "JWT_SECRET_KEY", "test-secret-key-for-unit-tests")
    with pytest.raises(HTTPException) as ex:
        asyncio.run(get_current_user_id(_creds("not-a-token")))
    assert ex.value.status_code == 401


def test_get_current_user_id_missing_sub(monkeypatch):
    monkeypatch.setattr(auth_utils, "JWT_SECRET_KEY", "test-secret-key-for-unit-tests")
    payload = {
        "email": "u@example.com",
        "exp": datetime.utcnow() + timedelta(hours=1),
    }
    token = jwt.encode(payload, auth_utils.JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    with pytest.raises(HTTPException) as ex:
        asyncio.run(get_current_user_id(_creds(token)))
    assert ex.value.status_code == 401


def test_require_superadmin_allows(monkeypatch):
    monkeypatch.setattr(auth_utils, "JWT_SECRET_KEY", "test-secret-key-for-unit-tests")
    monkeypatch.setattr(auth_utils, "SUPERADMIN_EMAIL", "admin@example.com")
    token = create_access_token("admin-uid", "admin@example.com")
    out = asyncio.run(require_superadmin(_creds(token)))
    assert out == "admin-uid"


def test_require_superadmin_rejects_non_admin(monkeypatch):
    monkeypatch.setattr(auth_utils, "JWT_SECRET_KEY", "test-secret-key-for-unit-tests")
    monkeypatch.setattr(auth_utils, "SUPERADMIN_EMAIL", "admin@example.com")
    token = create_access_token("uid", "regular@example.com")
    with pytest.raises(HTTPException) as ex:
        asyncio.run(require_superadmin(_creds(token)))
    assert ex.value.status_code == 403


def test_require_superadmin_no_admin_env(monkeypatch):
    monkeypatch.setattr(auth_utils, "JWT_SECRET_KEY", "test-secret-key-for-unit-tests")
    monkeypatch.setattr(auth_utils, "SUPERADMIN_EMAIL", "")
    token = create_access_token("uid", "regular@example.com")
    with pytest.raises(HTTPException) as ex:
        asyncio.run(require_superadmin(_creds(token)))
    assert ex.value.status_code == 403


def test_get_optional_user_id_none():
    assert asyncio.run(get_optional_user_id(None)) is None


def test_get_optional_user_id_valid(monkeypatch):
    monkeypatch.setattr(auth_utils, "JWT_SECRET_KEY", "test-secret-key-for-unit-tests")
    token = create_access_token("uid-2", "u@e.com")
    out = asyncio.run(get_optional_user_id(_creds(token)))
    assert out == "uid-2"


def test_get_optional_user_id_bad_token():
    assert asyncio.run(get_optional_user_id(_creds("garbage"))) is None


def test_is_superadmin_user_no_env(monkeypatch):
    monkeypatch.setattr(auth_utils, "SUPERADMIN_EMAIL", "")
    assert auth_utils.is_superadmin_user("any-uid") is False


def test_is_superadmin_user_match_and_mismatch(monkeypatch):
    monkeypatch.setattr(auth_utils, "SUPERADMIN_EMAIL", "admin@example.com")

    def fake_get_user(uid):
        return {
            "admin-uid": {"email": "admin@example.com"},
            "regular-uid": {"email": "user@example.com"},
        }.get(uid)

    import db

    monkeypatch.setattr(db, "get_user", fake_get_user)
    assert auth_utils.is_superadmin_user("admin-uid") is True
    assert auth_utils.is_superadmin_user("regular-uid") is False
    assert auth_utils.is_superadmin_user("unknown-uid") is False


def test_get_current_org_superadmin_bypasses_membership(monkeypatch):
    from auth_utils import get_current_org

    monkeypatch.setattr(auth_utils, "JWT_SECRET_KEY", "test-secret-key-for-unit-tests")
    monkeypatch.setattr(auth_utils, "SUPERADMIN_EMAIL", "admin@example.com")
    token = create_access_token("admin-uid", "admin@example.com")

    import db

    monkeypatch.setattr(db, "get_member_role", lambda org, uid: None)
    monkeypatch.setattr(db, "get_organization", lambda org: {"uuid": org})

    ctx = asyncio.run(get_current_org(_creds(token), x_org_uuid="some-org"))
    assert ctx.org_uuid == "some-org"
    assert ctx.role == "owner"
    assert ctx.user_id == "admin-uid"


def test_get_current_org_superadmin_bypass_requires_org_to_exist(monkeypatch):
    from auth_utils import get_current_org

    monkeypatch.setattr(auth_utils, "JWT_SECRET_KEY", "test-secret-key-for-unit-tests")
    monkeypatch.setattr(auth_utils, "SUPERADMIN_EMAIL", "admin@example.com")
    token = create_access_token("admin-uid", "admin@example.com")

    import db

    monkeypatch.setattr(db, "get_member_role", lambda org, uid: None)
    monkeypatch.setattr(db, "get_organization", lambda org: None)

    with pytest.raises(HTTPException) as ex:
        asyncio.run(get_current_org(_creds(token), x_org_uuid="ghost-org"))
    assert ex.value.status_code == 404
