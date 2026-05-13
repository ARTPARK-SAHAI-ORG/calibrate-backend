"""Unit tests for JWT helpers in auth_utils."""

import auth_utils
from auth_utils import create_access_token, decode_token


def test_create_and_decode_roundtrip(monkeypatch):
    monkeypatch.setattr(auth_utils, "JWT_SECRET_KEY", "test-secret-key-for-unit-tests")
    monkeypatch.setattr(auth_utils, "JWT_EXPIRATION_HOURS", 1)
    token = create_access_token("user-uuid-123", "user@example.com")
    payload = decode_token(token)
    assert payload is not None
    assert payload["sub"] == "user-uuid-123"
    assert payload["email"] == "user@example.com"


def test_decode_invalid_token_returns_none(monkeypatch):
    monkeypatch.setattr(auth_utils, "JWT_SECRET_KEY", "test-secret-key-for-unit-tests")
    assert decode_token("not-a-valid-jwt") is None
