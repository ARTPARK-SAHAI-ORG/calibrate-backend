import os
import sqlite3

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from db import (
    create_org_limits,
    get_member_role,
    get_organization,
    get_org_limits,
    update_org_limits,
    delete_org_limits,
)
from auth_utils import get_current_org, OrgContext, require_superadmin

router = APIRouter(prefix="/org-limits", tags=["org-limits"])

DEFAULT_MAX_ROWS_PER_EVAL = int(os.getenv("DEFAULT_MAX_ROWS_PER_EVAL", "20"))


class OrgLimits(BaseModel):
    max_rows_per_eval: int = Field(gt=0, le=10000)


class OrgLimitsCreate(BaseModel):
    org_uuid: str
    limits: OrgLimits


class OrgLimitsUpdate(BaseModel):
    limits: OrgLimits


class OrgLimitsResponse(BaseModel):
    uuid: str
    org_uuid: str
    limits: OrgLimits
    created_at: str
    updated_at: str


class OrgLimitsCreateResponse(BaseModel):
    uuid: str
    message: str


@router.get("/me/max-rows-per-eval")
async def get_max_rows_per_eval(ctx: OrgContext = Depends(get_current_org)):
    """Get the max rows per eval for the caller's current org.

    Returns the org-specific value from org_limits if set,
    otherwise falls back to DEFAULT_MAX_ROWS_PER_EVAL.
    """
    limits = get_org_limits(ctx.org_uuid)
    if limits and "max_rows_per_eval" in limits.get("limits", {}):
        return {"max_rows_per_eval": limits["limits"]["max_rows_per_eval"]}
    return {"max_rows_per_eval": DEFAULT_MAX_ROWS_PER_EVAL}


@router.post("", response_model=OrgLimitsCreateResponse)
async def create_org_limits_endpoint(
    data: OrgLimitsCreate, user_id: str = Depends(require_superadmin)
):
    """Create limits for an org. Superadmin only."""
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


@router.get("/{target_org_uuid}", response_model=OrgLimitsResponse)
async def get_org_limits_endpoint(
    target_org_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    """Get limits for an org. Must be a member of the target org."""
    if get_member_role(target_org_uuid, ctx.user_id) is None:
        raise HTTPException(status_code=404, detail="Organization limits not found")
    limits = get_org_limits(target_org_uuid)
    if not limits:
        raise HTTPException(status_code=404, detail="Organization limits not found")
    return limits


@router.put("/{target_org_uuid}", response_model=OrgLimitsResponse)
async def update_org_limits_endpoint(
    target_org_uuid: str,
    data: OrgLimitsUpdate,
    user_id: str = Depends(require_superadmin),
):
    """Update limits for an org. Superadmin only."""
    updated = update_org_limits(org_uuid=target_org_uuid, limits=data.limits)
    if not updated:
        raise HTTPException(status_code=404, detail="Organization limits not found")
    return updated


@router.delete("/{target_org_uuid}")
async def delete_org_limits_endpoint(
    target_org_uuid: str, user_id: str = Depends(require_superadmin)
):
    """Delete limits for an org. Superadmin only."""
    deleted = delete_org_limits(target_org_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Organization limits not found")
    return {"message": "Organization limits deleted successfully"}
