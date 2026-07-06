from typing import List, Dict, Any, Literal
from fastapi import APIRouter, HTTPException, Depends, Path
from pydantic import BaseModel, Field
from sqlite3 import IntegrityError

from auth_utils import get_current_org, OrgContext
from db import (
    add_tool_to_agent,
    remove_tool_from_agent,
    get_tools_for_agent,
    get_agents_for_tool,
    get_agent_tool_link,
    get_all_agent_tools,
    get_agent,
    get_tool,
)


router = APIRouter(prefix="/agent-tools", tags=["agent-tools"])


class AgentToolsCreate(BaseModel):
    agent_uuid: str = Field(
        description="UUID of the agent to link tools to (8-char identifier)"
    )
    tool_uuids: List[str] = Field(
        description="Tool UUIDs (8-char) to link. Already-linked tools are skipped"
    )


class AgentToolDelete(BaseModel):
    agent_uuid: str = Field(description="Agent UUID (8-char identifier)")
    tool_uuid: str = Field(
        description="UUID of the tool to unlink from the agent (8-char identifier)"
    )


class AgentToolResponse(BaseModel):
    id: int = Field(description="Auto-increment link row id")
    agent_id: str = Field(description="Linked agent UUID (8-char identifier)")
    tool_id: str = Field(description="Linked tool UUID (8-char identifier)")
    created_at: str = Field(description="Link creation timestamp (ISO 8601 UTC)")


class AgentToolsCreateResponse(BaseModel):
    ids: List[int] = Field(
        description="Link row ids created this call (excludes tools that were already linked)"
    )
    message: str = Field(description="Human-readable confirmation message")


class ToolResponse(BaseModel):
    uuid: str = Field(description="Tool UUID (8-char identifier)")
    name: str = Field(description="Human-readable tool name")
    description: str = Field(description="Tool description")
    config: Dict[str, Any] | None = Field(
        None, description="Tool configuration; null when the tool has none"
    )
    created_at: str = Field(description="Creation timestamp (ISO 8601 UTC)")
    updated_at: str = Field(description="Last-update timestamp (ISO 8601 UTC)")


class AgentResponse(BaseModel):
    uuid: str = Field(description="Agent UUID (8-char identifier)")
    name: str = Field(description="Human-readable agent name")
    type: Literal["agent", "connection"] = Field(
        description="`agent` (managed defaults) or `connection` (caller-supplied config)"
    )
    config: Dict[str, Any] | None = Field(
        None, description="Behavioral config; null when the agent has none"
    )
    created_at: str = Field(description="Creation timestamp (ISO 8601 UTC)")
    updated_at: str = Field(description="Last-update timestamp (ISO 8601 UTC)")


def _require_owned_agent(agent_uuid: str, org_uuid: str) -> Dict[str, Any]:
    agent = get_agent(agent_uuid)
    if not agent or agent.get("org_uuid") != org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


def _require_owned_tool(tool_uuid: str, org_uuid: str) -> Dict[str, Any]:
    tool = get_tool(tool_uuid)
    if not tool or tool.get("org_uuid") != org_uuid:
        raise HTTPException(status_code=404, detail=f"Tool {tool_uuid} not found")
    return tool


@router.post(
    "", response_model=AgentToolsCreateResponse, summary="Link tools to agent"
)
async def create_agent_tool_links(
    agent_tools: AgentToolsCreate,
    ctx: OrgContext = Depends(get_current_org),
):
    """Link one or more tools to an agent. The agent AND every tool must belong
    to the caller's workspace — cross-workspace linking is rejected as 404. Already-linked
    tools are skipped, so the call is idempotent."""
    _require_owned_agent(agent_tools.agent_uuid, ctx.org_uuid)
    for tool_uuid in agent_tools.tool_uuids:
        _require_owned_tool(tool_uuid, ctx.org_uuid)

    link_ids = []
    for tool_uuid in agent_tools.tool_uuids:
        existing = get_agent_tool_link(agent_tools.agent_uuid, tool_uuid)
        if existing:
            continue
        try:
            link_id = add_tool_to_agent(
                agent_id=agent_tools.agent_uuid,
                tool_id=tool_uuid,
            )
            link_ids.append(link_id)
        except IntegrityError:
            continue

    return AgentToolsCreateResponse(
        ids=link_ids, message="Tools added to agent successfully"
    )


@router.get(
    "", response_model=List[AgentToolResponse], summary="List agent-tool links"
)
async def list_agent_tools(ctx: OrgContext = Depends(get_current_org)):
    """List all agent-tool links scoped to the caller's workspace."""
    return get_all_agent_tools(org_uuid=ctx.org_uuid)


@router.get(
    "/agent/{agent_uuid}/tools",
    response_model=List[ToolResponse],
    summary="List tools for agent",
)
async def get_agent_tools(
    agent_uuid: str = Path(description="Agent UUID (8-char identifier)"),
    ctx: OrgContext = Depends(get_current_org),
):
    """List the tools linked to an agent. 404s if the agent isn't in the caller's workspace."""
    _require_owned_agent(agent_uuid, ctx.org_uuid)
    return get_tools_for_agent(agent_uuid)


@router.get(
    "/tool/{tool_uuid}/agents",
    response_model=List[AgentResponse],
    summary="List agents for tool",
)
async def get_tool_agents(
    tool_uuid: str = Path(description="Tool UUID (8-char identifier)"),
    ctx: OrgContext = Depends(get_current_org),
):
    """List the agents a tool is linked to. 404s if the tool isn't in the caller's workspace."""
    _require_owned_tool(tool_uuid, ctx.org_uuid)
    return get_agents_for_tool(tool_uuid)


@router.delete("", summary="Unlink tool from agent")
async def delete_agent_tool_link(
    agent_tool: AgentToolDelete, ctx: OrgContext = Depends(get_current_org)
):
    """Unlink a tool from an agent. Requires the agent to be in the caller's
    workspace — the tool's workspace doesn't matter on delete (the link being torn down
    was created by someone who already had access)."""
    _require_owned_agent(agent_tool.agent_uuid, ctx.org_uuid)
    deleted = remove_tool_from_agent(agent_tool.agent_uuid, agent_tool.tool_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent-tool link not found")
    return {"message": "Tool removed from agent successfully"}
