from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from db import create_tool, get_tool, get_all_tools, update_tool, delete_tool, ensure_name_unique
from auth_utils import get_current_org, OrgContext


router = APIRouter(prefix="/tools", tags=["tools"])


class ToolCreate(BaseModel):
    name: str
    description: str
    config: Optional[Dict[str, Any]] = None


class ToolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class ToolResponse(BaseModel):
    uuid: str
    name: str
    description: str
    config: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


class ToolCreateResponse(BaseModel):
    uuid: str
    message: str


@router.post("", response_model=ToolCreateResponse)
async def create_tool_endpoint(
    tool: ToolCreate, ctx: OrgContext = Depends(get_current_org)
):
    """Create a new tool."""
    with ensure_name_unique("tools", tool.name, ctx.org_uuid, entity="Tool"):
        tool_uuid = create_tool(
            name=tool.name,
            description=tool.description,
            config=tool.config,
            org_uuid=ctx.org_uuid,
            user_id=ctx.user_id,
        )
    return ToolCreateResponse(uuid=tool_uuid, message="Tool created successfully")


@router.get("", response_model=List[ToolResponse])
async def list_tools(ctx: OrgContext = Depends(get_current_org)):
    """List all tools for the caller's current org."""
    tools = get_all_tools(org_uuid=ctx.org_uuid)
    return tools


@router.get("/{tool_uuid}", response_model=ToolResponse)
async def get_tool_endpoint(
    tool_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    """Get a tool by UUID."""
    tool = get_tool(tool_uuid)
    if not tool or tool.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool


@router.put("/{tool_uuid}", response_model=ToolResponse)
async def update_tool_endpoint(
    tool_uuid: str, tool: ToolUpdate, ctx: OrgContext = Depends(get_current_org)
):
    """Update a tool."""
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


@router.delete("/{tool_uuid}")
async def delete_tool_endpoint(
    tool_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    """Delete a tool."""
    existing_tool = get_tool(tool_uuid)
    if not existing_tool or existing_tool.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Tool not found")

    deleted = delete_tool(tool_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Tool not found")
    return {"message": "Tool deleted successfully"}
