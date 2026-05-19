from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from db import (
    create_persona,
    get_persona,
    get_all_personas,
    update_persona,
    delete_persona,
    ensure_name_unique,
)
from auth_utils import get_current_org, OrgContext


router = APIRouter(prefix="/personas", tags=["personas"])


class PersonaCreate(BaseModel):
    name: str
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class PersonaUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class PersonaResponse(BaseModel):
    uuid: str
    name: str
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


class PersonaCreateResponse(BaseModel):
    uuid: str
    message: str


@router.post("", response_model=PersonaCreateResponse)
async def create_persona_endpoint(
    persona: PersonaCreate, ctx: OrgContext = Depends(get_current_org)
):
    """Create a new persona."""
    with ensure_name_unique("personas", persona.name, ctx.org_uuid, entity="Persona"):
        persona_uuid = create_persona(
            name=persona.name,
            description=persona.description,
            config=persona.config,
            org_uuid=ctx.org_uuid,
            user_id=ctx.user_id,
        )
    return PersonaCreateResponse(
        uuid=persona_uuid, message="Persona created successfully"
    )


@router.get("", response_model=List[PersonaResponse])
async def list_personas(ctx: OrgContext = Depends(get_current_org)):
    """List all personas for the caller's current org."""
    personas = get_all_personas(org_uuid=ctx.org_uuid)
    return personas


@router.get("/{persona_uuid}", response_model=PersonaResponse)
async def get_persona_endpoint(
    persona_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    """Get a persona by UUID."""
    persona = get_persona(persona_uuid)
    if not persona or persona.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Persona not found")
    return persona


@router.put("/{persona_uuid}", response_model=PersonaResponse)
async def update_persona_endpoint(
    persona_uuid: str,
    persona: PersonaUpdate,
    ctx: OrgContext = Depends(get_current_org),
):
    """Update a persona."""
    existing_persona = get_persona(persona_uuid)
    if not existing_persona or existing_persona.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Persona not found")

    with ensure_name_unique(
        "personas",
        persona.name,
        ctx.org_uuid,
        entity="Persona",
        exclude_uuid=persona_uuid,
    ):
        updated = update_persona(
            persona_uuid=persona_uuid,
            name=persona.name,
            description=persona.description,
            config=persona.config,
        )

    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated_persona = get_persona(persona_uuid)
    return updated_persona


@router.delete("/{persona_uuid}")
async def delete_persona_endpoint(
    persona_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    """Delete a persona."""
    existing_persona = get_persona(persona_uuid)
    if not existing_persona or existing_persona.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Persona not found")

    deleted = delete_persona(persona_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Persona not found")
    return {"message": "Persona deleted successfully"}
