from typing import List, Dict, Any, Literal
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
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
    agent_uuid: str
    tool_uuids: List[str]


class AgentToolDelete(BaseModel):
    agent_uuid: str
    tool_uuid: str


class AgentToolResponse(BaseModel):
    id: int
    agent_id: str
    tool_id: str
    created_at: str


class AgentToolsCreateResponse(BaseModel):
    ids: List[int]
    message: str


class ToolResponse(BaseModel):
    uuid: str
    name: str
    description: str
    config: Dict[str, Any] | None = None
    created_at: str
    updated_at: str


class AgentResponse(BaseModel):
    uuid: str
    name: str
    type: Literal["agent", "connection"]
    config: Dict[str, Any] | None = None
    created_at: str
    updated_at: str


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


@router.post("", response_model=AgentToolsCreateResponse)
async def create_agent_tool_links(
    agent_tools: AgentToolsCreate,
    ctx: OrgContext = Depends(get_current_org),
):
    """Link tools to an agent. The agent AND every tool must belong to the
    caller's org — cross-org linking is rejected as 404."""
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


@router.get("", response_model=List[AgentToolResponse])
async def list_agent_tools(ctx: OrgContext = Depends(get_current_org)):
    """List agent-tool links scoped to the caller's org."""
    return get_all_agent_tools(org_uuid=ctx.org_uuid)


@router.get("/agent/{agent_uuid}/tools", response_model=List[ToolResponse])
async def get_agent_tools(
    agent_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    _require_owned_agent(agent_uuid, ctx.org_uuid)
    return get_tools_for_agent(agent_uuid)


@router.get("/tool/{tool_uuid}/agents", response_model=List[AgentResponse])
async def get_tool_agents(
    tool_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    _require_owned_tool(tool_uuid, ctx.org_uuid)
    return get_agents_for_tool(tool_uuid)


@router.delete("")
async def delete_agent_tool_link(
    agent_tool: AgentToolDelete, ctx: OrgContext = Depends(get_current_org)
):
    """Unlink a tool from an agent. Requires the agent to be in the caller's
    org — the tool's org doesn't matter on delete (the link being torn down
    was created by someone who already had access)."""
    _require_owned_agent(agent_tool.agent_uuid, ctx.org_uuid)
    deleted = remove_tool_from_agent(agent_tool.agent_uuid, agent_tool.tool_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent-tool link not found")
    return {"message": "Tool removed from agent successfully"}
