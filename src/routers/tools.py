from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Path
from pydantic import BaseModel, Field

from db import create_tool, get_tool, get_all_tools, update_tool, delete_tool, ensure_name_unique
from auth_utils import get_current_org, OrgContext


router = APIRouter(prefix="/tools", tags=["tools"])


class ToolCreate(BaseModel):
    name: str = Field(description="Human-readable tool name, unique within the org")
    description: str = Field(description="What the tool does; surfaced to agents and the UI")
    config: Optional[Dict[str, Any]] = Field(
        None, description="Tool config (e.g. JSON schema, parameters) as a free-form JSON object. Omit to leave unset"
    )


class ToolUpdate(BaseModel):
    name: Optional[str] = Field(
        None, description="New tool name, unique within the org. Omit to leave unchanged"
    )
    description: Optional[str] = Field(
        None, description="New description. Omit to leave unchanged"
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="New tool config (JSON object). Omit to leave unchanged"
    )


class ToolResponse(BaseModel):
    uuid: str = Field(description="Tool UUID (8-char identifier)")
    name: str = Field(description="Human-readable tool name")
    description: str = Field(description="What the tool does")
    config: Optional[Dict[str, Any]] = Field(
        None, description="Tool config (JSON object), or null if unset"
    )
    created_at: str = Field(description="Creation timestamp (ISO 8601 UTC)")
    updated_at: str = Field(description="Last-update timestamp (ISO 8601 UTC)")


class ToolCreateResponse(BaseModel):
    uuid: str = Field(description="UUID (8-char identifier) of the newly created tool")
    message: str = Field(description="Human-readable success message")


@router.post("", response_model=ToolCreateResponse, summary="Create tool")
async def create_tool_endpoint(
    tool: ToolCreate, ctx: OrgContext = Depends(get_current_org)
):
    """Create a new tool in the caller's current org. The name must be unique within the org."""
    with ensure_name_unique("tools", tool.name, ctx.org_uuid, entity="Tool"):
        tool_uuid = create_tool(
            name=tool.name,
            description=tool.description,
            config=tool.config,
            org_uuid=ctx.org_uuid,
            user_id=ctx.user_id,
        )
    return ToolCreateResponse(uuid=tool_uuid, message="Tool created successfully")


@router.get("", response_model=List[ToolResponse], summary="List tools")
async def list_tools(ctx: OrgContext = Depends(get_current_org)):
    """List all tools for the caller's current org."""
    tools = get_all_tools(org_uuid=ctx.org_uuid)
    return tools


@router.get("/{tool_uuid}", response_model=ToolResponse, summary="Get tool")
async def get_tool_endpoint(
    tool_uuid: str = Path(description="Tool UUID (8-char identifier)"),
    ctx: OrgContext = Depends(get_current_org),
):
    """Retrieve a single tool by UUID within the caller's org."""
    tool = get_tool(tool_uuid)
    if not tool or tool.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool


@router.put("/{tool_uuid}", response_model=ToolResponse, summary="Update tool")
async def update_tool_endpoint(
    tool: ToolUpdate,
    tool_uuid: str = Path(description="Tool UUID (8-char identifier)"),
    ctx: OrgContext = Depends(get_current_org),
):
    """Update a tool's fields. Only the provided fields change; omitted fields are left as-is."""
    existing_tool = get_tool(tool_uuid)
    if not existing_tool or existing_tool.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Tool not found")

    with ensure_name_unique(
        "tools", tool.name, ctx.org_uuid, entity="Tool", exclude_uuid=tool_uuid
    ):
        updated = update_tool(
            tool_uuid=tool_uuid,
            name=tool.name,
            description=tool.description,
            config=tool.config,
        )

    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated_tool = get_tool(tool_uuid)
    return updated_tool


@router.delete("/{tool_uuid}", summary="Delete tool")
async def delete_tool_endpoint(
    tool_uuid: str = Path(description="Tool UUID (8-char identifier)"),
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete a tool by UUID."""
    existing_tool = get_tool(tool_uuid)
    if not existing_tool or existing_tool.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Tool not found")

    deleted = delete_tool(tool_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Tool not found")
    return {"message": "Tool deleted successfully"}
