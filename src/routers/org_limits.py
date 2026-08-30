"""Workspace eval limits (superadmin configuration).

Set caps for each workspace on dataset rows per eval run. Members can read their
workspace's effective limit via `/me/max-rows-per-eval`.

The cap exists to bound what a run costs the server. A workspace paying for the
run with its own provider keys therefore has no cap, which is why every call
site names the providers its run will actually use.
"""

import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional

from fastapi import APIRouter, HTTPException, Depends, Path, Query
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
from provider_keys import covered_providers

router = APIRouter(prefix="/org-limits", tags=["org-limits"])

DEFAULT_MAX_ROWS_PER_EVAL = int(os.getenv("DEFAULT_MAX_ROWS_PER_EVAL", "20"))


class OrgLimits(BaseModel):
    max_rows_per_eval: int = Field(
        gt=0,
        le=10000,
        description="Maximum dataset rows a single eval run may process",
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


# The judge always runs on OpenRouter, whatever the flow: `calibrate_agent`'s
# judges, and the Sarvam LLM-WER and intent bundles, all build an OpenRouter
# client. Any run that scores anything bills this.
JUDGE_PROVIDER = "openrouter"


def providers_for_agent_run(agent: Dict[str, Any]) -> List[str]:
    """Providers an agent test, benchmark, or text simulation bills.

    Reads the model provider back out of the same `config["llm"]` the runners
    hand the CLI, so the two cannot drift. A connection-type agent is the
    customer's own endpoint and costs the server nothing, leaving just the judge.
    """
    config = agent.get("config") or {}
    if config.get("agent_url"):
        return [JUDGE_PROVIDER]
    provider = (config.get("llm") or {}).get("provider") or JUDGE_PROVIDER
    return [JUDGE_PROVIDER, provider]


def _workspace_pays_for(org_uuid: str, providers: Iterable[str]) -> bool:
    """True when the workspace's own keys cover every provider named."""
    wanted = {p for p in providers if p}
    if not wanted:
        return False
    return wanted <= covered_providers(org_uuid)


def effective_max_rows_per_eval(
    org_uuid: str, providers: Optional[Iterable[str]] = None
) -> Optional[int]:
    """Cap on rows per eval run, or None when the run is uncapped.

    `providers` names what the run will actually call. Passing every one of them
    covered by the workspace's own keys lifts the cap, because the run then costs
    the server nothing. Omitting it always yields the cap.
    """
    if providers is not None and _workspace_pays_for(org_uuid, providers):
        return None
    limits = get_org_limits(org_uuid)
    if limits and "max_rows_per_eval" in limits.get("limits", {}):
        return limits["limits"]["max_rows_per_eval"]
    return DEFAULT_MAX_ROWS_PER_EVAL


def enforce_max_rows_per_eval(
    org_uuid: str, rows: int, providers: Optional[Iterable[str]] = None
) -> None:
    """Reject a run that would process more rows than the workspace allows."""
    cap = effective_max_rows_per_eval(org_uuid, providers)
    if cap is not None and rows > cap:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This run would process {rows} rows, above this workspace's "
                f"limit of {cap}. Run fewer rows, add your own provider API keys, "
                "or ask an admin to raise the limit."
            ),
        )


@router.get("/me/max-rows-per-eval", summary="Get own max rows per eval")
def get_max_rows_per_eval(
    ctx: OrgContext = Depends(get_current_org),
    providers: Optional[List[str]] = Query(
        None,
        description="Providers an upcoming run would call. Answers `null` when your own API keys cover all of them",
    ),
):
    """Get the max rows per eval"""
    return {"max_rows_per_eval": effective_max_rows_per_eval(ctx.org_uuid, providers)}


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
