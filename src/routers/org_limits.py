import os
import sqlite3

from fastapi import APIRouter, HTTPException, Depends, Path
from pydantic import BaseModel, Field

from db import (
    create_org_limits,
    get_member_role,
    get_organization,
    get_org_limits,
    update_org_limits,
    delete_org_limits,
)
from auth_utils import get_current_org, OrgContext, require_superadmin, is_superadmin_user

router = APIRouter(prefix="/org-limits", tags=["org-limits"])

DEFAULT_MAX_ROWS_PER_EVAL = int(os.getenv("DEFAULT_MAX_ROWS_PER_EVAL", "20"))


class OrgLimits(BaseModel):
    max_rows_per_eval: int = Field(
        gt=0,
        le=10000,
        description="Maximum dataset rows a single eval run may process (1–10000)",
    )


class OrgLimitsCreate(BaseModel):
    org_uuid: str = Field(description="Target workspace UUID (8-char identifier)")
    limits: OrgLimits = Field(description="Limit values to set for the workspace")


class OrgLimitsUpdate(BaseModel):
    limits: OrgLimits = Field(description="New limit values for the workspace")


class OrgLimitsResponse(BaseModel):
    uuid: str = Field(description="Org-limits row identifier (8-char UUID)")
    org_uuid: str = Field(description="Workspace these limits apply to (8-char UUID)")
    limits: OrgLimits = Field(description="Current limit values for the workspace")
    created_at: str = Field(description="Creation timestamp (ISO 8601 UTC)")
    updated_at: str = Field(description="Last-update timestamp (ISO 8601 UTC)")


class OrgLimitsCreateResponse(BaseModel):
    uuid: str = Field(description="Identifier of the newly created org-limits row")
    message: str = Field(description="Human-readable status message")


@router.get("/me/max-rows-per-eval", summary="Get own max rows per eval")
async def get_max_rows_per_eval(ctx: OrgContext = Depends(get_current_org)):
    """Get the max rows per eval for the caller's current workspace. Falls back to
    the server default (`DEFAULT_MAX_ROWS_PER_EVAL`) when no workspace-specific
    limit is set."""
    limits = get_org_limits(ctx.org_uuid)
    if limits and "max_rows_per_eval" in limits.get("limits", {}):
        return {"max_rows_per_eval": limits["limits"]["max_rows_per_eval"]}
    return {"max_rows_per_eval": DEFAULT_MAX_ROWS_PER_EVAL}


@router.post("", response_model=OrgLimitsCreateResponse, summary="Create workspace limits")
async def create_org_limits_endpoint(
    data: OrgLimitsCreate, user_id: str = Depends(require_superadmin)
):
    """Create limits for an workspace. Superadmin only. Returns 404 if the workspace
    doesn't exist and 409 if limits already exist (use PUT to update)."""
    if not get_organization(data.org_uuid):
        raise HTTPException(status_code=404, detail="Organization not found")
    existing = get_org_limits(data.org_uuid)
    if existing:
        raise HTTPException(
            status_code=409,
            detail="Limits already exist for this organization. Use PUT to update.",
        )
    try:
        row_uuid = create_org_limits(org_uuid=data.org_uuid, limits=data.limits)
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="Limits already exist for this organization. Use PUT to update.",
        )
    return OrgLimitsCreateResponse(
        uuid=row_uuid, message="Organization limits created successfully"
    )


@router.get("/{target_org_uuid}", response_model=OrgLimitsResponse, summary="Get workspace limits")
async def get_org_limits_endpoint(
    target_org_uuid: str = Path(description="Target workspace UUID (8-char identifier)"),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get limits for an workspace. The caller must be a member of the target workspace (or
    a superadmin). Returns 404 if not permitted or no limits are set."""
    if get_member_role(target_org_uuid, ctx.user_id) is None and not is_superadmin_user(ctx.user_id):
        raise HTTPException(status_code=404, detail="Organization limits not found")
    limits = get_org_limits(target_org_uuid)
    if not limits:
        raise HTTPException(status_code=404, detail="Organization limits not found")
    return limits


@router.put("/{target_org_uuid}", response_model=OrgLimitsResponse, summary="Update workspace limits")
async def update_org_limits_endpoint(
    target_org_uuid: str = Path(description="Target workspace UUID (8-char identifier)"),
    data: OrgLimitsUpdate = ...,
    user_id: str = Depends(require_superadmin),
):
    """Update limits for an workspace. Superadmin only. Returns 404 if no limits row
    exists for the workspace."""
    updated = update_org_limits(org_uuid=target_org_uuid, limits=data.limits)
    if not updated:
        raise HTTPException(status_code=404, detail="Organization limits not found")
    return updated


@router.delete("/{target_org_uuid}", summary="Delete workspace limits")
async def delete_org_limits_endpoint(
    target_org_uuid: str = Path(description="Target workspace UUID (8-char identifier)"),
    user_id: str = Depends(require_superadmin),
):
    """Delete limits for an workspace, reverting it to the server default. Superadmin
    only. Returns 404 if no limits row exists for the workspace."""
    deleted = delete_org_limits(target_org_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Organization limits not found")
    return {"message": "Organization limits deleted successfully"}
