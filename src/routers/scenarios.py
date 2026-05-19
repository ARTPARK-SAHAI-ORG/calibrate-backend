from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from db import (
    create_scenario,
    get_scenario,
    get_all_scenarios,
    update_scenario,
    delete_scenario,
    ensure_name_unique,
)
from auth_utils import get_current_org, OrgContext


router = APIRouter(prefix="/scenarios", tags=["scenarios"])


class ScenarioCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ScenarioUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class ScenarioResponse(BaseModel):
    uuid: str
    name: str
    description: Optional[str] = None
    created_at: str
    updated_at: str


class ScenarioCreateResponse(BaseModel):
    uuid: str
    message: str


@router.post("", response_model=ScenarioCreateResponse)
async def create_scenario_endpoint(
    scenario: ScenarioCreate, ctx: OrgContext = Depends(get_current_org)
):
    """Create a new scenario."""
    with ensure_name_unique("scenarios", scenario.name, ctx.org_uuid, entity="Scenario"):
        scenario_uuid = create_scenario(
            name=scenario.name,
            description=scenario.description,
            org_uuid=ctx.org_uuid,
            user_id=ctx.user_id,
        )
    return ScenarioCreateResponse(
        uuid=scenario_uuid, message="Scenario created successfully"
    )


@router.get("", response_model=List[ScenarioResponse])
async def list_scenarios(ctx: OrgContext = Depends(get_current_org)):
    """List all scenarios for the caller's current org."""
    scenarios = get_all_scenarios(org_uuid=ctx.org_uuid)
    return scenarios


@router.get("/{scenario_uuid}", response_model=ScenarioResponse)
async def get_scenario_endpoint(
    scenario_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    """Get a scenario by UUID."""
    scenario = get_scenario(scenario_uuid)
    if not scenario or scenario.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario


@router.put("/{scenario_uuid}", response_model=ScenarioResponse)
async def update_scenario_endpoint(
    scenario_uuid: str,
    scenario: ScenarioUpdate,
    ctx: OrgContext = Depends(get_current_org),
):
    """Update a scenario."""
    existing_scenario = get_scenario(scenario_uuid)
    if not existing_scenario or existing_scenario.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Scenario not found")

    with ensure_name_unique(
        "scenarios",
        scenario.name,
        ctx.org_uuid,
        entity="Scenario",
        exclude_uuid=scenario_uuid,
    ):
        updated = update_scenario(
            scenario_uuid=scenario_uuid,
            name=scenario.name,
            description=scenario.description,
        )

    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated_scenario = get_scenario(scenario_uuid)
    return updated_scenario


@router.delete("/{scenario_uuid}")
async def delete_scenario_endpoint(
    scenario_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    """Delete a scenario."""
    existing_scenario = get_scenario(scenario_uuid)
    if not existing_scenario or existing_scenario.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Scenario not found")

    deleted = delete_scenario(scenario_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {"message": "Scenario deleted successfully"}
