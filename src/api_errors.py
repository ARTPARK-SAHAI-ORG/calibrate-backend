"""Shared API error envelope and OpenAPI response definitions.

Every application-raised error (HTTPException, NameAlreadyExistsError) is
normalized to::

    {"error": {"code": "<CODE>", "message": "<human text>", ...}}

Optional extra keys (``not_found``, ``conflicting_names``, …) sit alongside
``code`` and ``message`` inside ``error``.

FastAPI's built-in **422** validation errors are intentionally left on the
native ``{"detail": [{loc, msg, type, …}]}`` shape.
"""

from __future__ import annotations

from typing import Any, Dict, NoReturn, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

# Default machine codes when callers pass a plain string ``detail=``.
DEFAULT_ERROR_CODE_BY_STATUS: Dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    500: "INTERNAL_ERROR",
    502: "BAD_GATEWAY",
    503: "SERVICE_UNAVAILABLE",
    504: "GATEWAY_TIMEOUT",
}

DEFAULT_ERROR_MESSAGE_BY_STATUS: Dict[int, str] = {
    400: "Bad request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not found",
    409: "Conflict",
    500: "Internal server error",
    502: "Bad gateway",
    503: "Service unavailable",
    504: "Gateway timeout",
}


class ApiErrorBody(BaseModel):
    """Machine- and human-readable error payload."""

    model_config = ConfigDict(extra="allow")

    code: str = Field(
        description="Machine-readable error code (e.g. `NOT_FOUND`, `BAD_REQUEST`)"
    )
    message: str = Field(description="Human-readable explanation of the error")


class ErrorResponse(BaseModel):
    """Standard error envelope for application-raised failures."""

    error: ApiErrorBody = Field(description="Error details")


def default_error_code(status_code: int) -> str:
    return DEFAULT_ERROR_CODE_BY_STATUS.get(status_code, "ERROR")


def default_error_message(status_code: int) -> str:
    return DEFAULT_ERROR_MESSAGE_BY_STATUS.get(status_code, "Error")


def normalize_error_content(
    status_code: int, detail: Any
) -> Dict[str, Dict[str, Any]]:
    """Shape any HTTPException ``detail`` into the standard envelope."""
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        return {"error": dict(detail["error"])}

    if isinstance(detail, str):
        return {
            "error": {
                "code": default_error_code(status_code),
                "message": detail,
            }
        }

    if isinstance(detail, dict):
        fields = dict(detail)
        code = fields.pop("code", default_error_code(status_code))
        message = fields.pop("message", default_error_message(status_code))
        return {"error": {"code": code, "message": message, **fields}}

    return {
        "error": {
            "code": default_error_code(status_code),
            "message": str(detail),
        }
    }


def raise_api_error(
    status_code: int,
    message: str,
    *,
    code: Optional[str] = None,
    **extra: Any,
) -> NoReturn:
    """Raise an HTTPException whose handler will emit the standard envelope."""
    body: Dict[str, Any] = {
        "code": code or default_error_code(status_code),
        "message": message,
        **extra,
    }
    raise HTTPException(status_code=status_code, detail=body)


async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=normalize_error_content(exc.status_code, exc.detail),
        headers=getattr(exc, "headers", None),
    )


# Per-status OpenAPI entries for Public API routes (422 stays FastAPI-native).
PUBLIC_API_ERROR_RESPONSES: Dict[int, Dict[str, Any]] = {
    400: {
        "model": ErrorResponse,
        "description": "Invalid request (e.g. agent connection not verified, no linked tests).",
    },
    401: {
        "model": ErrorResponse,
        "description": "Invalid or revoked API key.",
    },
    403: {
        "model": ErrorResponse,
        "description": "Missing authentication credentials.",
    },
    404: {
        "model": ErrorResponse,
        "description": "Resource not found in your workspace.",
    },
    500: {
        "model": ErrorResponse,
        "description": "Unexpected server error.",
    },
}
