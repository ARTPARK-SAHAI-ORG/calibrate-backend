"""Production trace ingestion.

Customer backends POST one trace per agent turn: the conversation history as
`input` plus the produced `output`. Rows persist as a normal `traces` table in
pense.db.

The stored shape deliberately mirrors test creation. `input` is
`tests.config.history` verbatim, and `output.tool_calls` matches the
expected-tool-call shape, so the deferred "turn traces into tests" step will
need no transformation. That endpoint is not here yet. It is parked on branch
`traces-convert-to-tests-parked`.

New contract needs go into `metadata` keys, not new top-level fields.
Customers integrate against this shape, and every field deepens the eventual
OTel-gateway migration.
"""

import os
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from auth_utils import OrgContext, get_current_org, get_org_jwt_or_api_key
from db import (
    count_live_traces,
    create_trace,
    get_trace,
    list_traces,
    soft_delete_traces,
)
from org_scope import ensure_owned_agent
from pagination import PaginatedResponse, PaginationParams, page_envelope

router = APIRouter(prefix="/traces", tags=["traces"])

# A fixed ceiling, not a per-workspace setting: one number is enough until a
# customer actually needs a different one.
MAX_TRACES_PER_WORKSPACE = int(os.getenv("DEFAULT_MAX_TRACES", "50000"))

# How many traces one delete call accepts. Independent of the storage cap:
# lowering that must not shrink a user's ability to delete their way back
# under it.
MAX_DELETE_IDS = 50_000

# A list row parses each trace's whole stored conversation to build its
# previews, so pages stay small. The shared PaginationParams allows a million,
# which would load every trace in a full workspace into memory at once.
MAX_LIST_LIMIT = 200

MAX_INPUT_TURNS = 500
MAX_TURN_CONTENT_CHARS = 50_000
MAX_TOOL_CALLS = 50
MAX_METADATA_ENTRIES = 100
_EXAMPLE_TRACE_UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"

_TRACE_UUID_DESCRIPTION = "Unique ID for the trace"

_AGENT_ID_DESCRIPTION = "ID of the agent that produced the turn"

# Bounds each entry so a malformed list is rejected before it reaches the
# database rather than being bound into a query.
TraceUuid = Annotated[str, StringConstraints(min_length=36, max_length=36)]


class TraceTurn(BaseModel):
    # Extra keys (OpenAI `tool_calls`, `tool_call_id`, `name`, ...) are stored
    # verbatim so the history stays lossless for test conversion.
    model_config = ConfigDict(extra="allow")

    role: str = Field(
        min_length=1,
        max_length=64,
        description="Message author role in the conversation history",
    )
    content: Optional[str] = Field(
        None,
        max_length=MAX_TURN_CONTENT_CHARS,
        description="Message text. Omit for turns that only carry tool calls",
    )


class TraceToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = Field(
        min_length=1,
        max_length=255,
        description="Name of the tool the agent called",
    )
    arguments: Optional[Dict[str, Any]] = Field(
        None,
        description="Argument values the agent passed to the tool. Omit when the call had none",
    )


class TraceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: Optional[str] = Field(
        None,
        max_length=MAX_TURN_CONTENT_CHARS,
        description="The assistant reply text for this turn. Omit for turns that only issued tool calls",
    )
    tool_calls: Optional[List[TraceToolCall]] = Field(
        None,
        max_length=MAX_TOOL_CALLS,
        description="Tool calls the agent issued for this turn. Omit for plain text replies",
    )

    @model_validator(mode="after")
    def _require_response_or_tool_calls(self):
        if not (self.response and self.response.strip()) and not self.tool_calls:
            raise ValueError("output must include a response or at least one tool call")
        return self


class TraceMetadataEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(
        min_length=1,
        max_length=256,
        description="Name of the metadata entry",
    )
    value: str = Field(
        max_length=8192,
        description="Value of the metadata entry",
    )


class TraceIngest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(
        min_length=1,
        max_length=36,
        description=_AGENT_ID_DESCRIPTION + ". Must be an agent in your workspace",
    )
    message_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Your own ID for the last user message in `input`, stored for reference only. Omit if you have none",
    )
    conversation_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Your own ID for the conversation this turn belongs to, stored for reference only. Omit if you have none",
    )
    input: List[TraceTurn] = Field(
        min_length=1,
        max_length=MAX_INPUT_TURNS,
        description="Conversation history up to the reported output, oldest turn first, in OpenAI chat format",
    )
    output: TraceOutput = Field(description="What the agent produced for this turn")
    metadata: Optional[List[TraceMetadataEntry]] = Field(
        None,
        max_length=MAX_METADATA_ENTRIES,
        description="Key-value pairs stored with the trace. Prefer OTel `gen_ai.*` key names where they fit. Omit if you have none",
    )


class TraceIngestResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description=_TRACE_UUID_DESCRIPTION,
        examples=[_EXAMPLE_TRACE_UUID],
    )
    message_id: Optional[str] = Field(
        None, description="The message ID you sent, if any"
    )
    conversation_id: Optional[str] = Field(
        None, description="The conversation ID you sent, if any"
    )
    created_at: str = Field(description="When the trace was created (ISO 8601 UTC)")


class TraceSummary(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description=_TRACE_UUID_DESCRIPTION,
        examples=[_EXAMPLE_TRACE_UUID],
    )
    agent_id: str = Field(description=_AGENT_ID_DESCRIPTION)
    message_id: Optional[str] = Field(
        None, description="The message ID you sent, if any"
    )
    conversation_id: Optional[str] = Field(
        None, description="The conversation ID you sent, if any"
    )
    input_preview: Optional[str] = Field(
        None, description="The last user message, truncated for display"
    )
    response_preview: Optional[str] = Field(
        None, description="The agent reply, truncated for display"
    )
    tool_names: List[str] = Field(
        description="Names of the tools the agent issued on this turn, in order"
    )
    tool_calls: List[TraceToolCall] = Field(
        description="Tools the agent issued on this turn, with the arguments it passed"
    )
    turn_count: int = Field(
        description="Number of turns in the stored conversation history"
    )
    tool_call_count: int = Field(
        description="Number of tool calls the agent issued for this turn"
    )
    metadata_count: int = Field(
        description="Number of metadata entries stored with the trace"
    )
    created_at: str = Field(description="When the trace was created (ISO 8601 UTC)")


class TraceResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description=_TRACE_UUID_DESCRIPTION,
        examples=[_EXAMPLE_TRACE_UUID],
    )
    agent_id: str = Field(description=_AGENT_ID_DESCRIPTION)
    message_id: Optional[str] = Field(
        None, description="The message ID you sent, if any"
    )
    conversation_id: Optional[str] = Field(
        None, description="The conversation ID you sent, if any"
    )
    input: List[TraceTurn] = Field(
        description="Conversation history stored for this trace, oldest turn first"
    )
    output: TraceOutput = Field(description="What the agent produced for this turn")
    metadata: Optional[List[TraceMetadataEntry]] = Field(
        None, description="Key-value pairs stored with the trace"
    )
    created_at: str = Field(description="When the trace was created (ISO 8601 UTC)")
    updated_at: str = Field(
        description="When the trace was last updated (ISO 8601 UTC)"
    )


class BulkDeleteTracesRequest(BaseModel):
    # Unknown keys must not be silently dropped: a misspelled field would
    # otherwise look like it filtered something.
    model_config = ConfigDict(extra="forbid")

    trace_ids: List[TraceUuid] = Field(
        min_length=1,
        max_length=MAX_DELETE_IDS,
        description="IDs of the traces to delete",
    )


class BulkDeleteTracesResponse(BaseModel):
    deleted: int = Field(description="Number of traces deleted")


_PREVIEW_CHARS = 160


def _preview(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    if len(text) <= _PREVIEW_CHARS:
        return text
    return text[: _PREVIEW_CHARS - 1] + "…"


def _last_user_content(input_turns: List[Dict[str, Any]]) -> Optional[str]:
    for turn in reversed(input_turns or []):
        if turn.get("role") == "user" and isinstance(turn.get("content"), str):
            return turn["content"]
    return None


def _to_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    output = row.get("output") or {}
    calls = [
        call
        for call in (output.get("tool_calls") or [])
        if isinstance(call, dict) and call.get("tool")
    ]
    return {
        "uuid": row["uuid"],
        "agent_id": row["agent_id"],
        "message_id": row["message_id"],
        "conversation_id": row["conversation_id"],
        "input_preview": _preview(_last_user_content(row.get("input") or [])),
        "response_preview": _preview(output.get("response")),
        "tool_names": [call["tool"] for call in calls],
        "tool_calls": [
            {"tool": call["tool"], "arguments": call.get("arguments")}
            for call in calls
        ],
        "turn_count": len(row.get("input") or []),
        "tool_call_count": len(calls),
        "metadata_count": len(row.get("metadata") or []),
        "created_at": row["created_at"],
    }


@router.post("", response_model=TraceIngestResponse, summary="Create trace")
async def ingest_trace(
    payload: TraceIngest, ctx: OrgContext = Depends(get_org_jwt_or_api_key)
):
    """Store a production agent turn and its conversation history for later review"""
    ensure_owned_agent(payload.agent_id, ctx.org_uuid)

    cap = MAX_TRACES_PER_WORKSPACE
    current = count_live_traces(ctx.org_uuid)
    if current >= cap:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Trace limit reached for this workspace",
                "current": current,
                "max_traces": cap,
                "hint": "Delete traces to free capacity",
            },
        )

    row = create_trace(
        org_uuid=ctx.org_uuid,
        agent_id=payload.agent_id,
        message_id=payload.message_id,
        conversation_id=payload.conversation_id,
        input=[turn.model_dump(exclude_none=True) for turn in payload.input],
        output=payload.output.model_dump(exclude_none=True),
        metadata=(
            [entry.model_dump() for entry in payload.metadata]
            if payload.metadata
            else None
        ),
    )
    return {
        "uuid": row["uuid"],
        "message_id": row["message_id"],
        "conversation_id": row["conversation_id"],
        "created_at": row["created_at"],
    }


@router.get("", response_model=PaginatedResponse[TraceSummary], summary="List traces")
async def list_traces_endpoint(
    ctx: OrgContext = Depends(get_current_org),
    pagination: PaginationParams = Depends(),
    agent_id: Optional[str] = Query(
        None, description="Return only traces from this agent"
    ),
):
    """List ingested traces, newest first"""
    if pagination.limit > MAX_LIST_LIMIT:
        raise HTTPException(
            status_code=422, detail=f"limit must be {MAX_LIST_LIMIT} or less"
        )
    # Search/filter/count run in SQL (db.list_traces), not the post-fetch
    # pagination helpers, and paging uses the bounded PaginationParams rather
    # than the unbounded OptionalPaginationParams: traces are machine-written
    # and outgrow in-memory filtering fast.
    rows, total = list_traces(
        ctx.org_uuid,
        limit=pagination.limit,
        offset=pagination.offset,
        agent_id=agent_id,
    )
    return page_envelope([_to_summary(row) for row in rows], total, pagination)


@router.post(
    "/bulk-delete",
    response_model=BulkDeleteTracesResponse,
    summary="Bulk delete traces",
)
async def bulk_delete_traces(
    payload: BulkDeleteTracesRequest, ctx: OrgContext = Depends(get_current_org)
):
    """Soft-delete traces, freeing their capacity"""
    deleted = soft_delete_traces(ctx.org_uuid, trace_ids=payload.trace_ids)
    return {"deleted": deleted}


@router.get("/{trace_uuid}", response_model=TraceResponse, summary="Get trace")
async def get_trace_endpoint(
    trace_uuid: str = Path(
        description="The trace to retrieve",
        examples=[_EXAMPLE_TRACE_UUID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get one trace by its ID"""
    row = get_trace(ctx.org_uuid, trace_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Trace not found")
    return row
