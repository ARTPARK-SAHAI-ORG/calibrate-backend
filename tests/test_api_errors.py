"""Tests for the shared API error envelope and Public API error docs."""

import importlib.util
from pathlib import Path

from fastapi import HTTPException

from api_errors import (
    normalize_error_content,
    http_exception_handler,
    PUBLIC_API_ERROR_RESPONSES,
)
from main import _build_public_openapi


def test_normalize_string_detail():
    assert normalize_error_content(404, "Agent not found") == {
        "error": {"code": "NOT_FOUND", "message": "Agent not found"}
    }


def test_normalize_structured_detail_preserves_extra_fields():
    body = normalize_error_content(
        404,
        {
            "code": "AGENTS_NOT_FOUND",
            "message": "Unknown agent name(s)",
            "not_found": ["missing"],
        },
    )
    assert body == {
        "error": {
            "code": "AGENTS_NOT_FOUND",
            "message": "Unknown agent name(s)",
            "not_found": ["missing"],
        }
    }


def test_normalize_legacy_code_message_dict():
    body = normalize_error_content(
        409,
        {
            "code": "ITEM_NAME_CONFLICT",
            "message": "Name taken",
            "conflicting_names": ["a"],
        },
    )
    assert body["error"]["code"] == "ITEM_NAME_CONFLICT"
    assert body["error"]["conflicting_names"] == ["a"]


def test_http_exception_handler_emits_envelope():
    import asyncio

    exc = HTTPException(status_code=401, detail="Invalid or revoked API key")
    response = asyncio.run(http_exception_handler(None, exc))
    assert response.status_code == 401
    assert response.body == (
        b'{"error":{"code":"UNAUTHORIZED","message":"Invalid or revoked API key"}}'
    )


def test_public_openapi_includes_documented_error_responses():
    spec = _build_public_openapi()
    public_ops = [
        (path, method.lower())
        for path, ops in spec["paths"].items()
        for method in ops
    ]
    assert public_ops, "expected at least one Public API operation"

    documented_statuses = {str(code) for code in PUBLIC_API_ERROR_RESPONSES}
    for path, method in public_ops:
        op = spec["paths"][path][method]
        responses = op.get("responses", {})
        missing = documented_statuses - set(responses)
        assert not missing, f"{method.upper()} {path} missing responses: {sorted(missing)}"

    assert "ErrorResponse" in spec["components"]["schemas"]
    assert "ApiErrorBody" in spec["components"]["schemas"]


def test_routers_follow_public_error_response_decorator():
    """Every Public API route must declare PUBLIC_API_ERROR_RESPONSES."""
    repo_root = Path(__file__).resolve().parents[1]
    routers_dir = repo_root / "src" / "routers"
    checker_path = repo_root / "scripts" / "check_public_api_error_docs.py"
    spec = importlib.util.spec_from_file_location("checker", checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    violations = checker.find_violations(routers_dir)
    assert violations == [], "\n".join(violations)
