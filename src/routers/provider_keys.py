"""Provider API keys a workspace stores for its own runs.

Stored values are never returned. Each row carries a `display` computed when it
was stored, so listing them decrypts nothing.
"""

from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field

from auth_utils import OrgContext, get_current_org
from db import (
    get_org_provider_keys,
    set_org_provider_keys,
    soft_delete_org_provider_keys,
)
from provider_keys import (
    ENCRYPTION_KEY_ENV,
    PROVIDER_ENV_VARS,
    ProviderKeyError,
    encrypted_rows_for,
    encryption_available,
    env_var_kind,
    provider_env_vars,
)

router = APIRouter(prefix="/provider-keys", tags=["provider-keys"])

ProviderFieldKind = Literal["secret", "plain", "file"]


class ProviderKeyField(BaseModel):
    env_var: str = Field(description="Environment variable this value is stored under")
    kind: ProviderFieldKind = Field(
        description="What sort of value this is, which decides how it is shown back to you"
    )
    required: bool = Field(
        description="Whether the provider needs this value to be usable"
    )
    set: bool = Field(description="Whether your workspace has stored this value")
    display: Optional[str] = Field(
        None,
        description="Masked or shortened form of the stored value, never the value itself",
    )


class ProviderKeysResponse(BaseModel):
    provider: str = Field(description="Provider these values belong to")
    configured: bool = Field(
        description="Whether your workspace has stored every value this provider needs"
    )
    fields: List[ProviderKeyField] = Field(
        description="One entry for each value the provider needs"
    )


class SetProviderKeysRequest(BaseModel):
    values: Dict[str, str] = Field(
        description="Every value the provider needs, keyed by environment variable name"
    )


class DeleteProviderKeysResponse(BaseModel):
    message: str = Field(description="Confirmation message")


def _known_provider(provider: str) -> str:
    if provider not in PROVIDER_ENV_VARS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown provider '{provider}'. Available providers: "
                f"{', '.join(PROVIDER_ENV_VARS)}"
            ),
        )
    return provider


def _entry(provider: str, displays: Dict[str, str]) -> ProviderKeysResponse:
    fields = [
        ProviderKeyField(
            env_var=env_var,
            kind=env_var_kind(env_var),
            required=True,
            set=env_var in displays,
            display=displays.get(env_var) or None,
        )
        for env_var in provider_env_vars(provider)
    ]
    return ProviderKeysResponse(
        provider=provider,
        configured=all(f.set for f in fields),
        fields=fields,
    )


def _displays(rows: List[dict]) -> Dict[str, str]:
    return {row["env_var"]: row.get("display") or "" for row in rows}


@router.get("", response_model=List[ProviderKeysResponse], summary="List provider keys")
def list_provider_keys(ctx: OrgContext = Depends(get_current_org)):
    """List every provider you can hold keys for, and which of its values your workspace has stored."""
    displays = _displays(get_org_provider_keys(ctx.org_uuid))
    return [_entry(provider, displays) for provider in PROVIDER_ENV_VARS]


@router.put(
    "/{provider}",
    response_model=ProviderKeysResponse,
    summary="Update provider keys",
)
def set_provider_keys(
    request: SetProviderKeysRequest,
    provider: str = Path(
        description="Provider to store keys for",
        examples=["openai"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Store your workspace's own keys for one provider, replacing any values it already holds."""
    _known_provider(provider)
    if not encryption_available():
        raise HTTPException(
            status_code=503,
            detail=(
                f"{ENCRYPTION_KEY_ENV} is not configured on this server, so provider "
                "keys cannot be stored safely"
            ),
        )
    try:
        rows = encrypted_rows_for(provider, request.values)
    except ProviderKeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    stored = set_org_provider_keys(ctx.org_uuid, ctx.user_id, provider, rows)
    return _entry(provider, _displays(stored))


@router.delete(
    "/{provider}",
    response_model=DeleteProviderKeysResponse,
    summary="Delete provider keys",
)
def delete_provider_keys(
    provider: str = Path(
        description="Provider to remove keys for",
        examples=["openai"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Delete your workspace's keys for one provider so its runs use the server's keys again."""
    _known_provider(provider)
    if not soft_delete_org_provider_keys(ctx.org_uuid, provider):
        raise HTTPException(
            status_code=404, detail=f"No stored keys for '{provider}'"
        )
    return DeleteProviderKeysResponse(message=f"Removed your {provider} keys")
