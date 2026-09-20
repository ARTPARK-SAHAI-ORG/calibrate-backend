"""Workspace limits (superadmin configuration).

Set caps for each workspace: dataset rows per eval run, stored traces, traces
scored automatically, and trace scoring batch size and concurrency. Members read
the effective values via `/me`.
"""

import os
import sqlite3

from fastapi import APIRouter, HTTPException, Depends, Path
from typing import Dict, Optional

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
DEFAULT_MAX_SCORED_TRACES = int(os.getenv("DEFAULT_MAX_SCORED_TRACES", "100"))
DEFAULT_MAX_TRACES = int(os.getenv("DEFAULT_MAX_TRACES", "50000"))
DEFAULT_TRACE_SCORING_BATCH_SIZE = int(os.getenv("DEFAULT_TRACE_SCORING_BATCH_SIZE", "20"))
DEFAULT_MAX_CONCURRENT_TRACE_SCORING_BATCHES = int(
    os.getenv("MAX_CONCURRENT_TRACE_SCORING_BATCHES_PER_ORG", "1")
)


class OrgLimits(BaseModel):
    max_rows_per_eval: int = Field(
        gt=0,
        le=10000,
        description="Maximum dataset rows a single eval run may process",
    )
    max_scored_traces: Optional[int] = Field(
        None,
        gt=0,
        le=1_000_000,
        description="Maximum traces a workspace may score automatically, counted over its lifetime",
    )
    max_traces: Optional[int] = Field(
        None,
        gt=0,
        le=1_000_000,
        description="Maximum traces a workspace may store",
    )
    trace_scoring_batch_size: Optional[int] = Field(
        None,
        gt=0,
        le=1000,
        description="Maximum traces one scoring batch takes",
    )
    max_concurrent_trace_scoring_batches: Optional[int] = Field(
        None,
        gt=0,
        le=100,
        description="Maximum scoring batches a workspace may run at once",
    )


class OrgLimitsCreate(BaseModel):
    org_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Workspace to create limits for",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    )
    limits: OrgLimits = Field(description="Limit values to set")


class OrgLimitsUpdate(BaseModel):
    limits: OrgLimits = Field(description="New limit values")


class OrgLimitsResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Limits record ID",
    )
    org_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Workspace these limits apply to",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    )
    limits: OrgLimits = Field(description="Current limit values")
    created_at: str = Field(description="When the limits record was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the limits record was last updated (ISO 8601 UTC)")


class OrgLimitsCreateResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the newly created limits record",
    )
    message: str = Field(description="Status message")


_DEFAULT_LIMITS = {
    "max_rows_per_eval": lambda: DEFAULT_MAX_ROWS_PER_EVAL,
    "max_scored_traces": lambda: DEFAULT_MAX_SCORED_TRACES,
    "max_traces": lambda: DEFAULT_MAX_TRACES,
    "trace_scoring_batch_size": lambda: DEFAULT_TRACE_SCORING_BATCH_SIZE,
    "max_concurrent_trace_scoring_batches": (
        lambda: DEFAULT_MAX_CONCURRENT_TRACE_SCORING_BATCHES
    ),
}


def effective_limits(org_uuid: str) -> Dict[str, int]:
    """Every limit in force for a workspace, from ONE read of its row.

    Trace ingest is the highest-volume write here and needs two of these, so
    asking per key would put an extra read on the same file the scoring workers
    are writing to, for every trace.
    """
    stored = (get_org_limits(org_uuid) or {}).get("limits") or {}
    return {
        key: stored.get(key) if stored.get(key) is not None else default()
        for key, default in _DEFAULT_LIMITS.items()
    }


def _effective_limit(org_uuid: str, key: str, default: int) -> int:
    stored = ((get_org_limits(org_uuid) or {}).get("limits") or {}).get(key)
    return stored if stored is not None else default


def effective_max_rows_per_eval(org_uuid: str) -> int:
    """Workspace cap on rows per eval run, falling back to the server default."""
    return _effective_limit(org_uuid, "max_rows_per_eval", DEFAULT_MAX_ROWS_PER_EVAL)


def effective_max_scored_traces(org_uuid: str) -> int:
    """Workspace cap on automatically scored traces, falling back to the server default."""
    return _effective_limit(org_uuid, "max_scored_traces", DEFAULT_MAX_SCORED_TRACES)


def effective_max_traces(org_uuid: str) -> int:
    """Workspace cap on stored traces, falling back to the server default."""
    return _effective_limit(org_uuid, "max_traces", DEFAULT_MAX_TRACES)


def effective_trace_scoring_batch_size(org_uuid: str) -> int:
    """Workspace size of one trace scoring batch, falling back to the server default."""
    return _effective_limit(
        org_uuid, "trace_scoring_batch_size", DEFAULT_TRACE_SCORING_BATCH_SIZE
    )


def effective_max_concurrent_trace_scoring_batches(org_uuid: str) -> int:
    """Workspace cap on trace scoring batches running at once, falling back to the server default."""
    return _effective_limit(
        org_uuid,
        "max_concurrent_trace_scoring_batches",
        DEFAULT_MAX_CONCURRENT_TRACE_SCORING_BATCHES,
    )


def enforce_max_rows_per_eval(org_uuid: str, rows: int) -> None:
    """Reject a run that would process more rows than the workspace allows."""
    cap = effective_max_rows_per_eval(org_uuid)
    if rows > cap:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This run would process {rows} rows, above this workspace's "
                f"limit of {cap}. Run fewer rows, or ask an admin to raise the limit."
            ),
        )


@router.get("/me", summary="Get own workspace limits")
def get_own_limits(ctx: OrgContext = Depends(get_current_org)):
    """Get every limit in force for your workspace, whether set for it or left at the server default"""
    return effective_limits(ctx.org_uuid)


@router.get("/me/max-rows-per-eval", summary="Get own max rows per eval")
def get_max_rows_per_eval(ctx: OrgContext = Depends(get_current_org)):
    """Get the max rows per eval"""
    return {"max_rows_per_eval": effective_max_rows_per_eval(ctx.org_uuid)}


@router.get("/me/max-scored-traces", summary="Get own max scored traces")
def get_max_scored_traces(ctx: OrgContext = Depends(get_current_org)):
    """Get the max traces scored automatically"""
    return {"max_scored_traces": effective_max_scored_traces(ctx.org_uuid)}


@router.post("", response_model=OrgLimitsCreateResponse, summary="Create workspace limits")
def create_org_limits_endpoint(
    data: OrgLimitsCreate, user_id: str = Depends(require_superadmin)
):
    """Create limits for a workspace. Superadmin only"""
    # 404 if workspace missing; 409 if limits already exist (use PUT to update).
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
def get_org_limits_endpoint(
    target_org_uuid: str = Path(
        description="The workspace whose limits to read. You must be a member",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get limits for a workspace you belong to"""
    if get_member_role(target_org_uuid, ctx.user_id) is None and not is_superadmin_user(ctx.user_id):
        raise HTTPException(status_code=404, detail="Organization limits not found")
    limits = get_org_limits(target_org_uuid)
    if not limits:
        raise HTTPException(status_code=404, detail="Organization limits not found")
    return limits


@router.put("/{target_org_uuid}", response_model=OrgLimitsResponse, summary="Update workspace limits")
def update_org_limits_endpoint(
    target_org_uuid: str = Path(
        description="The workspace whose limits to update",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    data: OrgLimitsUpdate = ...,
    user_id: str = Depends(require_superadmin),
):
    """Update limits for a workspace. Superadmin only"""
    updated = update_org_limits(org_uuid=target_org_uuid, limits=data.limits)
    if not updated:
        raise HTTPException(status_code=404, detail="Organization limits not found")
    return updated


@router.delete("/{target_org_uuid}", summary="Delete workspace limits")
def delete_org_limits_endpoint(
    target_org_uuid: str = Path(
        description="The workspace whose limits to delete",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    user_id: str = Depends(require_superadmin),
):
    """Delete limits for a workspace, reverting it to the server default. Superadmin only"""
    deleted = delete_org_limits(target_org_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Organization limits not found")
    return {"message": "Organization limits deleted successfully"}
