"""API keys router — credentials for programmatic API access.

A key is scoped to your workspace (resolved via `get_current_org`, i.e.
the `X-Org-UUID` header or the personal workspace). The API key is returned
exactly once, on creation; afterwards only its prefix and bcrypt hash are stored,
so it can be listed/revoked but never re-displayed. Authenticate downstream
requests with `Authorization: Bearer <api-key>` or `X-API-Key: <api-key>` — see
`auth_utils.get_org_jwt_or_api_key`.
"""

import re
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field, field_validator

from auth_utils import (
    API_KEY_PREFIX,
    OrgContext,
    generate_api_key,
    get_current_org,
    hash_api_key,
)
from db import (
    create_api_key,
    get_api_key,
    list_api_keys_for_org,
    soft_delete_api_key,
)

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


class CreateApiKeyRequest(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable label for the key (1–200 chars), shown in listings",
    )


def _masked(last_four: str) -> str:
    """Display form once the raw key is gone, e.g. `••••1a2b`."""
    return f"{API_KEY_PREFIX}••••{last_four}"


_TZ_SUFFIX = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def _to_utc_iso(ts: Optional[str]) -> Optional[str]:
    """Normalize a SQLite UTC timestamp to explicit ISO-8601 UTC.

    SQLite `CURRENT_TIMESTAMP` is naive UTC (`2026-06-05 10:11:00`). Emitting it
    without a zone makes browsers parse it as local time, skewing "Last used" by
    the viewer's offset. We swap the space for `T` and append `Z` so the FE can
    `new Date(...)` it directly. No-op if a zone is already present or value is
    None/empty.
    """
    if not ts:
        return ts
    s = str(ts).strip().replace(" ", "T")
    return s if _TZ_SUFFIX.search(s) else s + "Z"


class ApiKeyResponse(BaseModel):
    """Listing shape — never includes the raw key.

    `last_four` is the only fragment of the key kept after creation;
    `masked_key` is a ready-to-render display string built from it.
    """

    uuid: str = Field(description="API key identifier (8-char UUID)")
    name: str = Field(description="Human-readable label for the key")
    last_four: str = Field(
        description="Last 4 chars of the raw key — the only fragment retained after creation"
    )
    masked_key: str = Field(
        description="Ready-to-render display string, e.g. `••••1a2b`"
    )
    last_used_at: Optional[str] = Field(
        None,
        description="When the key last authenticated a request (ISO 8601 UTC); `null` if never used",
    )
    created_at: str = Field(description="Creation timestamp (ISO 8601 UTC)")
    updated_at: str = Field(description="Last-update timestamp (ISO 8601 UTC)")

    # Stamp timestamps as explicit UTC (…Z) so the FE doesn't read them as local.
    @field_validator("created_at", "updated_at", "last_used_at")
    @classmethod
    def _stamp_utc(cls, v: Optional[str]) -> Optional[str]:
        return _to_utc_iso(v)

    @classmethod
    def from_row(cls, row: dict, **extra) -> "ApiKeyResponse":
        """Build the response (any subclass) from a DB row, deriving the display
        fields. `extra` carries subclass-only fields, e.g. the raw `key`."""
        last_four = row.get("key_last_four", "")
        return cls(last_four=last_four, masked_key=_masked(last_four), **extra, **row)


class CreateApiKeyResponse(ApiKeyResponse):
    """Creation shape — carries the raw `key` exactly once. Show it, then never
    again; subsequent reads only ever return `masked_key` / `last_four`."""

    key: str = Field(
        description="The API key. **Returned exactly once, at creation** — store it now; it can never be retrieved again"
    )


@router.post(
    "", response_model=CreateApiKeyResponse, status_code=201, summary="Create API key"
)
async def create_key(
    request: CreateApiKeyRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Mint a new API key for your workspace. The API key is
    returned exactly once in this response and never again — store it now."""
    raw_key, key_prefix = generate_api_key()
    row = create_api_key(
        org_uuid=ctx.org_uuid,
        owner_user_id=ctx.user_id,
        name=request.name,
        key_prefix=key_prefix,
        key_last_four=raw_key[-4:],
        key_hash=hash_api_key(raw_key),
    )
    return CreateApiKeyResponse.from_row(row, key=raw_key)


@router.get("", response_model=List[ApiKeyResponse], summary="List API keys")
async def list_keys(ctx: OrgContext = Depends(get_current_org)):
    """List active API keys for your workspace. Raw keys are never
    returned — only `masked_key` / `last_four`."""
    return [ApiKeyResponse.from_row(k) for k in list_api_keys_for_org(ctx.org_uuid)]


@router.delete("/{key_uuid}", status_code=204, summary="Revoke API key")
async def revoke_key(
    key_uuid: str = Path(description="API key UUID (8-char identifier)"),
    ctx: OrgContext = Depends(get_current_org),
):
    """Revoke (soft-delete) an API key, immediately disabling it for auth.
    Returns 404 if it isn't in your workspace."""
    if get_api_key(key_uuid, ctx.org_uuid) is None:
        raise HTTPException(status_code=404, detail="API key not found")
    soft_delete_api_key(key_uuid, ctx.org_uuid)
