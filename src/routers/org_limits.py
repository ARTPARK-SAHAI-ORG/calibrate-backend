"""Workspace limits (superadmin configuration).

Set caps for each workspace: dataset rows per eval run, stored traces, traces
scored automatically, and trace scoring batch size and concurrency. Members read
the effective values via `/me`.
"""

import logging
import os
import sqlite3

from fastapi import APIRouter, HTTPException, Depends, Path, Response
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

from db import (
    create_limit_hit,
    create_org_limits,
    get_member_role,
    get_organization,
    get_org_limits,
    update_org_limits,
    delete_org_limits,
)
from auth_utils import get_current_org, OrgContext, require_superadmin, is_superadmin_user
import provider_keys
from admin_analytics import EVAL_FIELDS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/org-limits", tags=["org-limits"])

# The kinds of run a limit hit can name: the same list the analytics count.
EvalKind = Literal[tuple(EVAL_FIELDS)]

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


class OwnLimitsResponse(BaseModel):
    max_rows_per_eval: Optional[int] = Field(
        description="Maximum dataset rows a single eval run may process. No limit (`null`) while your workspace uses its own AI provider keys"
    )
    max_scored_traces: Optional[int] = Field(
        description="Maximum traces your workspace may score automatically, counted over its lifetime. No limit (`null`) while your workspace uses its own AI provider keys"
    )
    max_traces: int = Field(description="Maximum traces your workspace may store")
    trace_scoring_batch_size: int = Field(
        description="Maximum traces one scoring batch takes"
    )
    max_concurrent_trace_scoring_batches: int = Field(
        description="Maximum scoring batches your workspace may run at once"
    )
    uses_own_provider_keys: bool = Field(
        description="Whether your workspace's runs use only its own AI provider keys instead of Calibrate's"
    )


class MaxRowsPerEvalResponse(BaseModel):
    max_rows_per_eval: Optional[int] = Field(
        description="Maximum dataset rows a single eval run may process. No limit (`null`) while your workspace uses its own AI provider keys"
    )


class MaxScoredTracesResponse(BaseModel):
    max_scored_traces: Optional[int] = Field(
        description="Maximum traces your workspace may score automatically, counted over its lifetime. No limit (`null`) while your workspace uses its own AI provider keys"
    )


# A workspace paying its own providers is not capped on what it spends.
_SPEND_LIMITS = ("max_rows_per_eval", "max_scored_traces")

_DEFAULT_LIMITS = {
    "max_rows_per_eval": lambda: DEFAULT_MAX_ROWS_PER_EVAL,
    "max_scored_traces": lambda: DEFAULT_MAX_SCORED_TRACES,
    "max_traces": lambda: DEFAULT_MAX_TRACES,
    "trace_scoring_batch_size": lambda: DEFAULT_TRACE_SCORING_BATCH_SIZE,
    "max_concurrent_trace_scoring_batches": (
        lambda: DEFAULT_MAX_CONCURRENT_TRACE_SCORING_BATCHES
    ),
}


def effective_limits(org_uuid: str) -> Dict[str, Optional[int]]:
    """Every limit in force for a workspace, from ONE read of its row.

    Trace ingest is the highest-volume write here and needs two of these, so
    asking per key would put an extra read on the same file the scoring workers
    are writing to, for every trace.
    """
    stored = (get_org_limits(org_uuid) or {}).get("limits") or {}
    limits = {
        key: stored.get(key) if stored.get(key) is not None else default()
        for key, default in _DEFAULT_LIMITS.items()
    }
    if provider_keys.org_uses_own_keys(org_uuid):
        limits.update(dict.fromkeys(_SPEND_LIMITS))
    return limits


def _effective_limit(org_uuid: str, key: str) -> Optional[int]:
    return effective_limits(org_uuid)[key]


def effective_max_rows_per_eval(org_uuid: str) -> Optional[int]:
    """Workspace cap on rows per eval run, falling back to the server default."""
    return _effective_limit(org_uuid, "max_rows_per_eval")


def effective_max_scored_traces(org_uuid: str) -> Optional[int]:
    """Workspace cap on automatically scored traces, falling back to the server default."""
    return _effective_limit(org_uuid, "max_scored_traces")


def effective_max_traces(org_uuid: str) -> int:
    """Workspace cap on stored traces, falling back to the server default."""
    return _effective_limit(org_uuid, "max_traces")


def effective_trace_scoring_batch_size(org_uuid: str) -> int:
    """Workspace size of one trace scoring batch, falling back to the server default."""
    return _effective_limit(org_uuid, "trace_scoring_batch_size")


def effective_max_concurrent_trace_scoring_batches(org_uuid: str) -> int:
    """Workspace cap on trace scoring batches running at once, falling back to the server default."""
    return _effective_limit(org_uuid, "max_concurrent_trace_scoring_batches")


def record_limit_hit(
    org_uuid: str,
    limit_name: str,
    kind: Optional[str],
    requested: Optional[int],
    cap: Optional[int],
) -> None:
    """Record a refusal for analytics. Never raises: the caller's own answer wins."""
    try:
        create_limit_hit(org_uuid, limit_name, kind, requested, cap)
    except Exception:
        logger.exception(f"Could not record a {limit_name} limit hit for {org_uuid}")


def enforce_max_rows_per_eval(org_uuid: str, rows: int, kind: EvalKind) -> None:
    """Reject a run that would process more rows than the workspace allows."""
    cap = effective_max_rows_per_eval(org_uuid)
    if cap is not None and rows > cap:
        record_limit_hit(org_uuid, "max_rows_per_eval", kind, rows, cap)
        raise HTTPException(
            status_code=400,
            detail=(
                f"This run would process {rows} rows, above this workspace's "
                f"limit of {cap}. Run fewer rows, or ask an admin to raise the limit."
            ),
        )


@router.get("/me", response_model=OwnLimitsResponse, summary="Get own workspace limits")
def get_own_limits(ctx: OrgContext = Depends(get_current_org)):
    """Get every limit in force for your workspace, whether set for it or left at the server default"""
    return {
        **effective_limits(ctx.org_uuid),
        "uses_own_provider_keys": provider_keys.org_uses_own_keys(ctx.org_uuid),
    }


@router.get(
    "/me/max-rows-per-eval",
    response_model=MaxRowsPerEvalResponse,
    summary="Get own maximum rows per eval run",
)
def get_max_rows_per_eval(ctx: OrgContext = Depends(get_current_org)):
    """Get the most dataset rows one eval run in your workspace may process"""
    return {"max_rows_per_eval": effective_max_rows_per_eval(ctx.org_uuid)}


@router.get(
    "/me/max-scored-traces",
    response_model=MaxScoredTracesResponse,
    summary="Get own maximum scored traces",
)
def get_max_scored_traces(ctx: OrgContext = Depends(get_current_org)):
    """Get how many traces your workspace may have scored automatically"""
    return {"max_scored_traces": effective_max_scored_traces(ctx.org_uuid)}


class LimitHitCreate(BaseModel):
    limit_name: Literal["max_rows_per_eval"] = Field(
        description="The limit the run went over",
        examples=["max_rows_per_eval"],
    )
    kind: Optional[EvalKind] = Field(
        None,
        description="What kind of evaluation was refused, or `null` when unknown",
        examples=["test_run"],
    )
    requested: int = Field(
        ge=0,
        description="How many rows the refused run would have processed",
        examples=[600],
    )


@router.post("/hits", status_code=204, summary="Record a limit hit")
def record_limit_hit_endpoint(
    data: LimitHitCreate, ctx: OrgContext = Depends(get_current_org)
):
    """Record that a run in your workspace was stopped before it started for going over a limit"""
    cap = effective_limits(ctx.org_uuid)[data.limit_name]
    # Only a run that really was over the limit is a hit.
    if cap is not None and data.requested > cap:
        record_limit_hit(ctx.org_uuid, data.limit_name, data.kind, data.requested, cap)
    return Response(status_code=204)


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
