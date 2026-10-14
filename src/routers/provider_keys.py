"""AI provider keys a workspace runs its evaluations with instead of Calibrate's."""

import os
from typing import List, Literal

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field, StringConstraints, field_validator
from typing_extensions import Annotated

import db
import provider_keys
from auth_utils import OrgContext, get_current_org
from provider_status import PROVIDER_ENV_VARS
from routers.api_keys import _to_utc_iso
from routers.org_limits import effective_max_scored_traces

router = APIRouter(prefix="/provider-keys", tags=["provider-keys"])

Provider = Literal[tuple(PROVIDER_ENV_VARS)]


class ProviderKeyRequest(BaseModel):
    value: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = Field(
        description="Your API key for the provider, or the service account JSON when the provider is `google`"
    )


class ProviderKeyResponse(BaseModel):
    provider: Provider = Field(description="The AI provider the key belongs to")
    masked_value: str = Field(
        description="Masked display form of the key, or the service account email when the provider is `google`"
    )
    updated_at: str = Field(description="When the key was last updated (ISO 8601 UTC)")

    @field_validator("updated_at")
    @classmethod
    def _stamp_utc(cls, v: str) -> str:
        return _to_utc_iso(v)


class ProviderKeysResponse(BaseModel):
    uses_own_keys: bool = Field(
        description="Whether your workspace's runs use only its own AI provider keys instead of Calibrate's"
    )
    keys: List[ProviderKeyResponse] = Field(description="The keys your workspace has stored, one for each provider")


@router.get("", response_model=ProviderKeysResponse, summary="List provider keys")
def list_provider_keys(ctx: OrgContext = Depends(get_current_org)):
    """List the AI provider keys your workspace runs its evaluations with"""
    keys = db.list_org_provider_keys(ctx.org_uuid)
    return {"uses_own_keys": bool(keys), "keys": keys}


@router.put("/{provider}", response_model=ProviderKeyResponse, summary="Update provider key")
async def save_provider_key(
    request: ProviderKeyRequest,
    provider: Provider = Path(description="The AI provider the key belongs to", examples=["openai"]),
    ctx: OrgContext = Depends(get_current_org),
):
    """Check that a provider key works, then store it for your workspace. Once your workspace has any key stored, its runs use only its own keys and have no limit on rows per run or scored traces"""
    if not provider_keys.encryption_configured():
        raise HTTPException(status_code=503, detail="Saving provider keys is not set up on this server.")
    if provider == "google" and provider_keys.parse_google_credentials(request.value) is None:
        raise HTTPException(
            status_code=400,
            detail="The google key must be a service account JSON file with a project_id and a client_email.",
        )
    error = await provider_keys.check_provider_key(provider, request.value)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return db.upsert_org_provider_key(
        ctx.org_uuid,
        provider,
        provider_keys.encrypt(request.value),
        provider_keys.masked_value(provider, request.value),
        ctx.user_id,
    )


@router.delete("/{provider}", status_code=204, summary="Delete provider key")
def remove_provider_key(
    provider: Provider = Path(description="The AI provider the key belongs to", examples=["openai"]),
    ctx: OrgContext = Depends(get_current_org),
):
    """Delete a provider key. Once the last one is deleted, your workspace's runs go back to Calibrate's keys and limits"""
    keys = db.list_org_provider_keys(ctx.org_uuid)
    if provider not in {k["provider"] for k in keys}:
        raise HTTPException(status_code=404, detail="Provider key not found")
    # ponytail: check and delete are separate statements, so a run created in between
    # skipped the spend limits yet runs on the server's keys; one transaction if it matters.
    if len(keys) == 1 and db.org_has_active_jobs(ctx.org_uuid):
        raise HTTPException(
            status_code=409,
            detail="Wait for this workspace's runs to finish, or stop them, before removing its last provider key.",
        )
    db.delete_org_provider_key(ctx.org_uuid, provider)
    if provider == "google":
        try:
            os.remove(provider_keys.google_creds_path(ctx.org_uuid))
        except FileNotFoundError:
            pass
    if len(keys) == 1:
        db.trim_pending_trace_runs_over_cap(ctx.org_uuid, effective_max_scored_traces(ctx.org_uuid))
