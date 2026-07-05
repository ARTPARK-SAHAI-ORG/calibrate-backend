from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Path
from pydantic import BaseModel, Field

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
    name: str = Field(description="Human-readable persona name, unique within the org")
    description: Optional[str] = Field(
        None, description="Free-text description of the persona. Omit to leave unset"
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="Behavioral config as a free-form JSON object. Omit to leave unset"
    )


class PersonaUpdate(BaseModel):
    name: Optional[str] = Field(
        None, description="New persona name, unique within the org. Omit to leave unchanged"
    )
    description: Optional[str] = Field(
        None, description="New description. Omit to leave unchanged"
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="New behavioral config (JSON object). Omit to leave unchanged"
    )


class PersonaResponse(BaseModel):
    uuid: str = Field(description="Persona UUID (8-char identifier)")
    name: str = Field(description="Human-readable persona name")
    description: Optional[str] = Field(
        None, description="Free-text description, or null if unset"
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="Behavioral config (JSON object), or null if unset"
    )
    created_at: str = Field(description="Creation timestamp (ISO 8601 UTC)")
    updated_at: str = Field(description="Last-update timestamp (ISO 8601 UTC)")


class PersonaCreateResponse(BaseModel):
    uuid: str = Field(description="UUID (8-char identifier) of the newly created persona")
    message: str = Field(description="Human-readable success message")


@router.post("", response_model=PersonaCreateResponse, summary="Create persona")
async def create_persona_endpoint(
    persona: PersonaCreate, ctx: OrgContext = Depends(get_current_org)
):
    """Create a new persona in the caller's current org. The name must be unique within the org."""
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


@router.get("", response_model=List[PersonaResponse], summary="List personas")
async def list_personas(ctx: OrgContext = Depends(get_current_org)):
    """List all personas for the caller's current org."""
    personas = get_all_personas(org_uuid=ctx.org_uuid)
    return personas


@router.get("/{persona_uuid}", response_model=PersonaResponse, summary="Get persona")
async def get_persona_endpoint(
    persona_uuid: str = Path(description="Persona UUID (8-char identifier)"),
    ctx: OrgContext = Depends(get_current_org),
):
    """Retrieve a single persona by UUID within the caller's org."""
    persona = get_persona(persona_uuid)
    if not persona or persona.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Persona not found")
    return persona


@router.put("/{persona_uuid}", response_model=PersonaResponse, summary="Update persona")
async def update_persona_endpoint(
    persona: PersonaUpdate,
    persona_uuid: str = Path(description="Persona UUID (8-char identifier)"),
    ctx: OrgContext = Depends(get_current_org),
):
    """Update a persona's fields. Only the provided fields change; omitted fields are left as-is."""
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


@router.delete("/{persona_uuid}", summary="Delete persona")
async def delete_persona_endpoint(
    persona_uuid: str = Path(description="Persona UUID (8-char identifier)"),
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete a persona by UUID."""
    existing_persona = get_persona(persona_uuid)
    if not existing_persona or existing_persona.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Persona not found")

    deleted = delete_persona(persona_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Persona not found")
    return {"message": "Persona deleted successfully"}
