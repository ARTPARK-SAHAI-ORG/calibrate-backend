"""Production trace ingestion and curation into tests.

Customer backends POST one trace per agent turn: the conversation history as
`input` plus the produced `output`. Rows persist as a normal `traces` table in
pense.db.

The stored shape deliberately mirrors test creation. `input` is
`tests.config.history` verbatim for a conversational agent and
`tests.config.input` verbatim for a `general` one, and `output.tool_calls`
matches the expected-tool-call shape, so `POST /traces/convert-to-tests` needs
no transformation.

New contract needs go into `metadata` keys, not new top-level fields.
Customers integrate against this shape, and every field deepens the eventual
OTel-gateway migration.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from auth_utils import OrgContext, get_current_org, get_org_jwt_or_api_key
from db import (
    _trace_iso,
    add_test_to_agent,
    bulk_create_tests,
    bulk_delete_tests,
    average_trace_scores,
    count_live_traces,
    count_scored_traces,
    create_trace_rescore_run,
    create_trace_with_eval_run,
    get_agent,
    resolve_live_evaluators,
    get_all_tests_summary,
    get_evaluator_versions_by_uuids,
    get_evaluators_by_uuids,
    get_latest_trace_run_summaries,
    get_trace,
    set_trace_feedback,
    get_traces_by_uuids,
    list_trace_conversation_ids,
    list_trace_conversations,
    list_trace_labels,
    list_trace_scoring_runs,
    list_traces,
    set_test_evaluators,
    soft_delete_traces,
    soft_delete_traces_matching,
    TraceFilters,
    TraceScoreFilter,
)
from org_scope import ensure_owned_agent
from pagination import PaginatedResponse, PaginationParams, page_envelope
from routers.org_limits import effective_limits, record_limit_hit

# Reuse the tests router's validation so a converted test accepts exactly what
# POST /tests does (evaluator visible to the workspace, evaluator_type matches).
from shared_enums import (
    DEFAULT_AGENT_INTERACTION_TYPE,
    EvaluatorType,
    required_agent_interaction_type,
)
from trace_scoring import (
    MAX_METADATA_KEY_CHARS,
    TraceEvalRunStatus,
    evaluator_trace_sources,
    trace_evaluators,
)
from routers.tests import (
    EvaluatorRef,
    _validate_evaluators,
)
from utils import (
    EVALUATOR_TYPE_DESCRIPTION,
    EXAMPLE_TEST_UUID,
    EvaluatorUuid,
    OUTPUT_TYPE_DESCRIPTION,
    OutputTypeLiteral,
    trace_scoring_enabled,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/traces", tags=["traces"])

# How many traces one delete call accepts. Independent of the storage cap:
# lowering that must not shrink a user's ability to delete their way back
# under it.
MAX_DELETE_IDS = 50_000

# A list row parses each trace's whole stored conversation to build its
# previews, so pages stay small. The shared PaginationParams allows a million,
# which would load every trace in a full workspace into memory at once.
MAX_LIST_LIMIT = 200

# How many traces one conversion accepts. Each becomes a test row plus its
# evaluator and agent links, all in one request.
MAX_CONVERT_TRACES = 500

MAX_INPUT_TURNS = 500
MAX_TURN_CONTENT_CHARS = 50_000
MAX_TOOL_CALLS = 50
MAX_METADATA_ENTRIES = 100
# Caps both how many labels one trace carries and how many a filter may name,
# so listing, deleting and converting all accept the same set.
MAX_LABELS = 50
MAX_LABEL_CHARS = 128

# Caps how many score conditions one request may combine. Each one costs a
# lookup of the trace's latest run, so an unbounded repeat is an unbounded
# query. Enforced on the list and on both bulk actions, so a selection the
# list can produce is one the delete after it accepts.
MAX_SCORE_FILTERS = 20

_EXAMPLE_TRACE_UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"

_TRACE_UUID_DESCRIPTION = "Unique ID for the trace"

_AGENT_ID_DESCRIPTION = "ID of the agent that produced the turn"

_LABELS_DESCRIPTION = (
    "Your own tags for this turn, such as an environment or a release. "
    "Matched exactly when filtering, so keep the spelling stable"
)

# Trailing spaces and duplicates would each read as a separate tag in the
# filter, so a label is stored in the one form a caller can search for.
TraceLabel = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=MAX_LABEL_CHARS
    ),
]

# Bounds each entry so a malformed list is rejected before it reaches the
# database rather than being bound into a query.
TraceUuid = Annotated[str, StringConstraints(min_length=36, max_length=36)]

# Longest first, so `>=` is read as one operator rather than `>` followed by
# a number starting with `=`.
_SCORE_COMPARISONS = (">=", "<=", "!=", ">", "<", "=")

_SCORE_CONDITION_SYNTAX = (
    "Each is written as `<evaluator ID>:<condition>`, where the condition is "
    "`passed`, `failed`, or one of `=`, `!=`, `<`, `<=`, `>`, `>=` followed by "
    "a number, such as `>=4`"
)

_SCORE_FILTER_DESCRIPTION = (
    "Return only traces whose latest scoring run matches this condition. "
    + _SCORE_CONDITION_SYNTAX
    + ". Repeat the parameter to require every condition at once"
)

_SORT_BY_EVALUATOR_DESCRIPTION = (
    "Order traces by this evaluator's score on each trace's latest scoring "
    "run. Traces this evaluator has not scored come last whichever direction you pick, and "
    "an evaluator not linked to the agent counts as having scored none"
)

TraceSource = Literal["api", "chat"]
TraceFeedbackFilter = Literal["correct", "wrong", "any"]

_SOURCE_VALUES = (
    "\n\n- `api`: sent to `POST /traces`\n"
    "- `chat`: a reply the agent gave on its chat link"
)
_FEEDBACK_VALUES = (
    "\n\n- `correct`: marked correct\n"
    "- `wrong`: marked wrong\n"
    "- `any`: marked either way"
)

_SORT_ORDER_DESCRIPTION = (
    "Whether `sort_by_evaluator` puts the highest scores first or the lowest. "
    "Ignored without it"
)


def _parse_score_filters(values: Optional[List[str]]) -> List[TraceScoreFilter]:
    """Turn each `<evaluator ID>:<condition>` into what db.list_traces takes."""
    raw_values = values or []
    if len(raw_values) > MAX_SCORE_FILTERS:
        raise HTTPException(
            status_code=422,
            detail=f"score accepts at most {MAX_SCORE_FILTERS} conditions",
        )
    parsed = []
    for raw in raw_values:
        evaluator_uuid, separator, condition = raw.partition(":")
        evaluator_uuid, condition = evaluator_uuid.strip(), condition.strip()
        if not separator or not evaluator_uuid or not condition:
            raise _bad_score_filter(raw)
        if condition in ("passed", "failed"):
            parsed.append((evaluator_uuid, condition, None))
            continue
        operator = next(
            (op for op in _SCORE_COMPARISONS if condition.startswith(op)), None
        )
        if operator is None:
            raise _bad_score_filter(raw)
        try:
            number = float(condition[len(operator) :].strip())
        except ValueError:
            raise _bad_score_filter(raw)
        # float() reads "nan" and "inf". SQLite has no NaN, so a NaN binds as
        # NULL and the comparison matches nothing, turning a typo into an
        # empty page rather than an error.
        if not math.isfinite(number):
            raise _bad_score_filter(raw)
        parsed.append((evaluator_uuid, operator, number))
    return parsed


def _bad_score_filter(raw: str) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail=(
            f"{raw!r} is not a score filter. Write an evaluator ID, a colon, "
            "then passed, failed, or a comparison such as >=4"
        ),
    )


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


# A trace's input takes the shape its agent is called with: one standalone
# prompt for a `general` agent, a conversation for the rest. Separate aliases
# rather than one Field so each side keeps its own bound (turn count vs
# characters) instead of sharing whichever one is declared.
TraceInputText = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_TURN_CONTENT_CHARS)
]
TraceInputHistory = Annotated[
    List[TraceTurn], Field(min_length=1, max_length=MAX_INPUT_TURNS)
]
TraceInput = Union[TraceInputText, TraceInputHistory]

_TRACE_INPUT_DESCRIPTION = (
    "What the agent was given for this turn. For a `general` agent, the "
    "standalone prompt as a string. For a `conversation` agent, the history up "
    "to the reported output, oldest turn first, in OpenAI chat format"
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
    # Stored and displayed only. Conversion asserts the call, not its result,
    # so `POST /traces/convert-to-tests` never reads this.
    output: Any = Field(
        None,
        description="What the tool returned for this call. Any JSON value. Omit when you do not record it",
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
    stopped_early: Optional[bool] = Field(
        None,
        description="True when the agent was stopped after calling tools as many times in a row as it may, before it replied",
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
        max_length=MAX_METADATA_KEY_CHARS,
        description="Name of the metadata entry",
    )
    value: str = Field(
        max_length=MAX_TURN_CONTENT_CHARS,
        description="Value of the metadata entry",
    )


_IS_CORRECT_DESCRIPTION = "Whether the agent's output for this turn was correct"
_FEEDBACK_COMMENT_DESCRIPTION = "What was right or wrong about the agent's output"


class TraceFeedbackUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_correct: Optional[bool] = Field(
        description=_IS_CORRECT_DESCRIPTION + ". `null` clears the feedback"
    )
    comment: Optional[str] = Field(
        None, max_length=2000, description=_FEEDBACK_COMMENT_DESCRIPTION + ". Omit if you have none"
    )


class TraceFeedbackIngest(TraceFeedbackUpdate):
    """Feedback sent with a new trace, which has nothing to clear."""

    is_correct: bool = Field(description=_IS_CORRECT_DESCRIPTION)


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
        description="Your own ID for the conversation this turn belongs to. Turns sharing it are grouped into one conversation. Omit if you have none",
    )
    input: TraceInput = Field(description=_TRACE_INPUT_DESCRIPTION)
    output: TraceOutput = Field(description="What the agent produced for this turn")
    metadata: Optional[List[TraceMetadataEntry]] = Field(
        None,
        max_length=MAX_METADATA_ENTRIES,
        description="Key-value pairs stored with the trace. Prefer OTel `gen_ai.*` key names where they fit. Omit if you have none",
    )
    labels: List[TraceLabel] = Field(
        default_factory=list,
        max_length=MAX_LABELS,
        description=_LABELS_DESCRIPTION + ". Omit if you have none",
    )
    feedback: Optional[TraceFeedbackIngest] = Field(
        None,
        description="A verdict on the agent's output, if you already have one. Omit if you have none",
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


class TraceFeedback(TraceFeedbackIngest):
    at: str = Field(description="When the feedback was last given (ISO 8601 UTC)")


_TRACE_SOURCE_DESCRIPTION = "Where the trace came from:" + _SOURCE_VALUES
_TRACE_FEEDBACK_DESCRIPTION = (
    "The verdict and comment given on the trace, through the API or on the chat link"
)


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
        None,
        description="The standalone prompt or the last user message, truncated for display",
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
        description="Number of turns in the stored input. A standalone prompt counts as 1"
    )
    tool_call_count: int = Field(
        description="Number of tool calls the agent issued for this turn"
    )
    metadata_count: int = Field(
        description="Number of metadata entries stored with the trace"
    )
    labels: List[str] = Field(description=_LABELS_DESCRIPTION)
    source: TraceSource = Field(description=_TRACE_SOURCE_DESCRIPTION)
    feedback: Optional[TraceFeedback] = Field(
        None, description=_TRACE_FEEDBACK_DESCRIPTION
    )
    input: Optional[TraceInput] = Field(
        None,
        description=_TRACE_INPUT_DESCRIPTION + ". Sent only with `include_content`",
    )
    output: Optional[TraceOutput] = Field(
        None,
        description="What the agent produced for this turn. Sent only with `include_content`",
    )
    metadata: Optional[List[TraceMetadataEntry]] = Field(
        None,
        description="Key-value pairs stored with the trace. Sent only with `include_content`",
    )
    created_at: str = Field(description="When the trace was created (ISO 8601 UTC)")
    latest_run_status: Optional[TraceEvalRunStatus] = Field(
        None,
        description="Status of the latest scoring run for this trace",
    )
    latest_run_error: Optional[str] = Field(
        None, description="Why the latest scoring run was skipped or failed"
    )
    results: List["TraceScoreResult"] = Field(
        default_factory=list,
        description="Results on the latest scoring run from the evaluators linked to the agent",
    )


class TraceScoreResult(BaseModel):
    evaluator_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the evaluator that produced this result",
        examples=[_EXAMPLE_TRACE_UUID],
    )
    name: str = Field(
        description="Name of the evaluator. Uses the last stored name if it has been deleted"
    )
    evaluator_type: Optional[EvaluatorType] = Field(
        None, description=EVALUATOR_TYPE_DESCRIPTION
    )
    output_type: OutputTypeLiteral = Field(description=OUTPUT_TYPE_DESCRIPTION)
    scale_min: Optional[float] = Field(
        None, description="Lowest value on a rating scale"
    )
    scale_max: Optional[float] = Field(
        None, description="Highest value on a rating scale"
    )
    value: float = Field(
        description="The judged result. Binary stores 0 or 1. Rating stores the numeric score"
    )
    reasoning: Optional[str] = Field(
        None, description="The judge's rationale for this result"
    )
    evaluator_version_id: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the evaluator version that produced this result",
        examples=[_EXAMPLE_TRACE_UUID],
    )
    passed: bool = Field(
        description=(
            "Whether this evaluator passed. A binary result passes on 1. "
            "A rating passes at the top of its scale"
        )
    )


class TraceUnconfiguredEvaluator(BaseModel):
    evaluator_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the evaluator that did not score this trace",
        examples=[_EXAMPLE_TRACE_UUID],
    )
    name: str = Field(
        description="Name of the evaluator. Uses the last stored name if it has been deleted"
    )


class TraceMissingValue(TraceUnconfiguredEvaluator):
    variable: str = Field(description="The evaluator variable that had no value")
    source: str = Field(description="The part of the trace chosen to fill the variable")


class TraceScoringRun(BaseModel):
    run_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of this scoring run",
        examples=[_EXAMPLE_TRACE_UUID],
    )
    status: TraceEvalRunStatus = Field(
        description=(
            "Status of the scoring run:\n\n"
            "- `pending`: waiting to be claimed\n"
            "- `processing`: a worker holds the lease\n"
            "- `completed`: every snapshot evaluator has a result\n"
            "- `failed`: attempts exhausted with at least one evaluator unscored\n"
            "- `skipped`: nothing to score. `error` holds the reason"
        )
    )
    created_at: str = Field(description="When the run was created (ISO 8601 UTC)")
    completed_at: Optional[str] = Field(
        None, description="When the run reached a terminal status (ISO 8601 UTC)"
    )
    error: Optional[str] = Field(
        None, description="Why this run was skipped or failed"
    )
    results: List[TraceScoreResult] = Field(
        description="Results on this run from the evaluators linked to the agent"
    )
    missing_values: List[TraceMissingValue] = Field(
        description="Variables this trace had no value for, so their evaluators did not score it"
    )
    unconfigured_evaluators: List[TraceUnconfiguredEvaluator] = Field(
        description="Evaluators that did not score this trace because a variable had no part of the trace chosen to fill it"
    )
    newly_scored_evaluators: List[TraceUnconfiguredEvaluator] = Field(
        description="The only evaluators this run scored, because they were not configured for traces when the trace was scored before. Every other score was copied from that earlier run. Empty when the run scored with every evaluator"
    )


class TraceScoresResponse(BaseModel):
    runs: List[TraceScoringRun] = Field(
        description="Scoring runs for this trace, newest first"
    )


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
    input: TraceInput = Field(description=_TRACE_INPUT_DESCRIPTION)
    output: TraceOutput = Field(description="What the agent produced for this turn")
    metadata: Optional[List[TraceMetadataEntry]] = Field(
        None, description="Key-value pairs stored with the trace"
    )
    labels: List[str] = Field(description=_LABELS_DESCRIPTION)
    source: TraceSource = Field(description=_TRACE_SOURCE_DESCRIPTION)
    feedback: Optional[TraceFeedback] = Field(
        None, description=_TRACE_FEEDBACK_DESCRIPTION
    )
    created_at: str = Field(description="When the trace was created (ISO 8601 UTC)")
    updated_at: str = Field(
        description="When the trace was last updated (ISO 8601 UTC)"
    )


class TraceConversationIdsResponse(BaseModel):
    conversation_ids: List[str] = Field(
        description="Every conversation ID on your live traces, newest activity first"
    )


class TraceLabelsResponse(BaseModel):
    labels: List[str] = Field(
        description="Every label in use on your live traces, A to Z"
    )


_SELECT_ALL_DESCRIPTION = (
    "Act on every trace matching the filters below instead of a list of IDs. "
    "`trace_ids` is ignored when this is on"
)
_FILTER_AGENT_DESCRIPTION = (
    "With `select_all` on, act only on traces from this agent"
)
_FILTER_Q_DESCRIPTION = (
    "With `select_all` on, act only on traces containing this text in their "
    "message ID, conversation ID, conversation history, output, or metadata"
)
_FILTER_OUTPUT_TYPE_DESCRIPTION = (
    "With `select_all` on, act only on traces whose output is of this kind. "
    "`response` covers every trace carrying a reply, including one that also "
    "issued tool calls. `tool_call` covers traces that only issued tool calls"
)


_FILTER_LABELS_DESCRIPTION = (
    "With `select_all` on, act only on traces carrying at least one of these labels"
)

_FILTER_SCORE_DESCRIPTION = (
    "With `select_all` on, act only on traces whose latest scoring run matches "
    "every one of these conditions. " + _SCORE_CONDITION_SYNTAX
)


# Two agents can send the same conversation ID; without the agent, a filter on
# it would mix their traces.
_CONVERSATION_NEEDS_AGENT = "conversation_id needs agent_id"


def _stored_time(value: Optional[datetime]) -> Optional[str]:
    """Write a time the way `traces.created_at` stores it, so the two compare as text."""
    if value is None:
        return None
    if value.tzinfo is not None:
        try:
            value = value.astimezone(timezone.utc)
        except OverflowError:
            raise HTTPException(status_code=422, detail="time is out of range")
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _to_trace_filters(source: Any) -> TraceFilters:
    """Read a request's filters, parsing the score conditions before the times."""
    scores = _parse_score_filters(source.score)
    return TraceFilters(
        agent_id=source.agent_id,
        conversation_id=source.conversation_id,
        q=source.q,
        output_type=source.output_type,
        labels=source.labels,
        scores=scores,
        source=source.source,
        feedback=source.feedback,
        created_after=_stored_time(source.created_after),
        created_before=_stored_time(source.created_before),
    )


def _checked_query_filters(params: Any, pagination: PaginationParams) -> TraceFilters:
    """Run the checks both GET list routes share, in the order they report errors."""
    if params.conversation_id and not params.agent_id:
        raise HTTPException(status_code=422, detail=_CONVERSATION_NEEDS_AGENT)
    if pagination.limit > MAX_LIST_LIMIT:
        raise HTTPException(
            status_code=422, detail=f"limit must be {MAX_LIST_LIMIT} or less"
        )
    if params.labels and len(params.labels) > MAX_LABELS:
        raise HTTPException(
            status_code=422, detail=f"labels accepts at most {MAX_LABELS} labels"
        )
    return _to_trace_filters(params)


class _TraceSelection(BaseModel):
    """Either a list of IDs or `select_all` plus the same filters `GET /traces`
    takes, so a caller acting on a whole filtered set never has to name its rows."""

    # Unknown keys must not be silently dropped: a misspelled field would
    # otherwise look like it filtered something.
    model_config = ConfigDict(extra="forbid")

    # The cap is enforced in the validator, not as `max_length`, so a caller
    # that posts its whole selection alongside `select_all` is not rejected on
    # a list the handler never reads.
    max_trace_ids: ClassVar[int] = 0

    trace_ids: List[TraceUuid] = Field(
        default_factory=list, description="IDs of the traces to act on"
    )
    select_all: bool = Field(False, description=_SELECT_ALL_DESCRIPTION)
    agent_id: Optional[str] = Field(
        None, min_length=1, description=_FILTER_AGENT_DESCRIPTION
    )
    conversation_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="With `select_all` on, act only on the turns of this conversation. Requires `agent_id`",
    )
    q: Optional[str] = Field(None, description=_FILTER_Q_DESCRIPTION)
    output_type: Optional[Literal["response", "tool_call"]] = Field(
        None, description=_FILTER_OUTPUT_TYPE_DESCRIPTION
    )
    labels: List[TraceLabel] = Field(
        default_factory=list,
        max_length=MAX_LABELS,
        description=_FILTER_LABELS_DESCRIPTION,
    )
    score: List[str] = Field(
        default_factory=list,
        max_length=MAX_SCORE_FILTERS,
        description=_FILTER_SCORE_DESCRIPTION,
    )
    created_after: Optional[datetime] = Field(
        None,
        description="With `select_all` on, act only on traces created at or after this time (ISO 8601). A time with no timezone is read as UTC. Omit for no earliest date",
    )
    created_before: Optional[datetime] = Field(
        None,
        description="With `select_all` on, act only on traces created before this time (ISO 8601). A time with no timezone is read as UTC. Omit for no latest date",
    )
    source: Optional[TraceSource] = Field(
        None,
        description="With `select_all` on, act only on traces from this source:" + _SOURCE_VALUES,
    )
    feedback: Optional[TraceFeedbackFilter] = Field(
        None,
        description="With `select_all` on, act only on traces with this feedback:" + _FEEDBACK_VALUES,
    )

    @model_validator(mode="after")
    def _require_a_selection(self):
        if self.conversation_id and not self.agent_id:
            raise ValueError(_CONVERSATION_NEEDS_AGENT)
        if self.select_all:
            return self
        if not self.trace_ids:
            raise ValueError("provide trace_ids, or select_all=true")
        if len(self.trace_ids) > self.max_trace_ids:
            raise ValueError(
                f"trace_ids accepts at most {self.max_trace_ids} IDs"
            )
        return self


class BulkDeleteTracesRequest(_TraceSelection):
    max_trace_ids: ClassVar[int] = MAX_DELETE_IDS

    trace_ids: List[TraceUuid] = Field(
        default_factory=list,
        description="IDs of the traces to delete. Omit when `select_all` is on",
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


def _last_user_content(stored_input: Any) -> Optional[str]:
    if isinstance(stored_input, str):
        return stored_input
    for turn in reversed(stored_input or []):
        if turn.get("role") == "user" and isinstance(turn.get("content"), str):
            return turn["content"]
    return None


def _test_name_for_trace(trace: Dict[str, Any]) -> str:
    """`message_id`, else the last thing the user said (cut to 60 characters),
    else the trace uuid."""
    if trace["message_id"]:
        return trace["message_id"]
    text = (_last_user_content(trace["input"]) or "").strip()
    if not text:
        return trace["uuid"]
    return text if len(text) <= 60 else text[:60].rstrip() + "…"


def _turn_count(stored_input: Any) -> int:
    """A standalone prompt is the single turn it stands for."""
    return 1 if isinstance(stored_input, str) else len(stored_input or [])


def _to_summary(
    row: Dict[str, Any], scoring: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    output = row.get("output") or {}
    calls = [
        call
        for call in (output.get("tool_calls") or [])
        if isinstance(call, dict) and call.get("tool")
    ]
    scoring = scoring or {}
    return {
        "uuid": row["uuid"],
        "agent_id": row["agent_id"],
        "message_id": row["message_id"],
        "conversation_id": row["conversation_id"],
        "input_preview": _preview(_last_user_content(row.get("input"))),
        "response_preview": _preview(output.get("response")),
        "tool_names": [call["tool"] for call in calls],
        "tool_calls": [
            {
                "tool": call["tool"],
                "arguments": call.get("arguments"),
                "output": call.get("output"),
            }
            for call in calls
        ],
        "turn_count": _turn_count(row.get("input")),
        "tool_call_count": len(calls),
        "metadata_count": len(row.get("metadata") or []),
        "labels": row.get("labels") or [],
        "source": row["source"],
        "feedback": row["feedback"],
        "created_at": row["created_at"],
        "latest_run_status": scoring.get("status"),
        "latest_run_error": scoring.get("error"),
        "results": scoring.get("results") or [],
    }


def _ensure_input_matches_agent(stored_input: Any, agent: Dict[str, Any]) -> None:
    """Reject an input whose shape the trace's agent is never called with.

    Tying the two together at ingest is what lets `convert-to-tests` build each
    test from the trace alone: the stored shape already matches what the agent
    can be given, so a converted test can only ever fit it.
    """
    interaction_type = (
        agent.get("interaction_type") or DEFAULT_AGENT_INTERACTION_TYPE
    )
    is_text = isinstance(stored_input, str)
    if is_text and interaction_type != "general":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Agent has interaction_type='{interaction_type}' and is called "
                "with a conversation, so input must be a list of turns, not a string."
            ),
        )
    if not is_text and interaction_type == "general":
        raise HTTPException(
            status_code=400,
            detail=(
                "Agent has interaction_type='general' and is called with a "
                "standalone prompt, so input must be a string, not a list of turns."
            ),
        )
    if is_text and not stored_input.strip():
        raise HTTPException(status_code=400, detail="input must not be blank")


@router.post(
    "",
    response_model=TraceIngestResponse,
    summary="Create trace",
    tags=["Public API"],
)
def ingest_trace(
    payload: TraceIngest, ctx: OrgContext = Depends(get_org_jwt_or_api_key)
):
    """Store a production agent turn and its conversation history for later review"""
    agent = ensure_owned_agent(payload.agent_id, ctx.org_uuid)
    _ensure_input_matches_agent(payload.input, agent)

    limits = effective_limits(ctx.org_uuid)
    cap = limits["max_traces"]
    current = count_live_traces(ctx.org_uuid)
    if current >= cap:
        record_limit_hit(ctx.org_uuid, "max_traces", "trace", current + 1, cap)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Trace limit reached for this workspace",
                "current": current,
                "max_traces": cap,
                "hint": "Delete traces to free capacity",
            },
        )

    row = create_trace_with_eval_run(
        org_uuid=ctx.org_uuid,
        agent=agent,
        max_scored_traces=limits["max_scored_traces"],
        batch_size=limits["trace_scoring_batch_size"],
        message_id=payload.message_id,
        conversation_id=payload.conversation_id,
        input=(
            payload.input
            if isinstance(payload.input, str)
            else [turn.model_dump(exclude_none=True) for turn in payload.input]
        ),
        output=payload.output.model_dump(exclude_none=True),
        metadata=(
            [entry.model_dump() for entry in payload.metadata]
            if payload.metadata
            else None
        ),
        labels=list(dict.fromkeys(payload.labels)),
        feedback=payload.feedback.model_dump() if payload.feedback else None,
    )
    return {
        "uuid": row["uuid"],
        "message_id": row["message_id"],
        "conversation_id": row["conversation_id"],
        "created_at": row["created_at"],
    }


class TraceScoreAverage(BaseModel):
    evaluator_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the evaluator these numbers are for",
        examples=[_EXAMPLE_TRACE_UUID],
    )
    name: str = Field(
        description="Name of the evaluator. Uses the last stored name if it has been deleted"
    )
    output_type: OutputTypeLiteral = Field(description=OUTPUT_TYPE_DESCRIPTION)
    traces_scored: int = Field(
        description="How many of the matching traces this evaluator scored"
    )
    average: float = Field(
        description="Mean of those scores. A binary evaluator averages 0 and 1, so its mean is the share that passed"
    )
    scale_min: Optional[float] = Field(
        None, description="Lowest value on a rating scale"
    )
    scale_max: Optional[float] = Field(
        None, description="Highest value on a rating scale"
    )


class TraceListResponse(PaginatedResponse[TraceSummary]):
    """The page, plus the averages when `include_score_averages` asked for them."""

    score_averages: Optional[List[TraceScoreAverage]] = Field(
        None,
        description="One average for each evaluator linked to the agent that scored at least one trace matching the filters, covering all the matching traces it scored rather than this page. Ask for it with `include_score_averages`",
    )


# Plain dataclasses: a pydantic query model would change the published parameters.
@dataclass
class _TraceListFilters:
    agent_id: Optional[str] = Query(
        None, description="Return only traces from this agent"
    )
    conversation_id: Optional[str] = Query(
        None,
        min_length=1,
        max_length=255,
        description="Return only the turns of this conversation. Requires `agent_id`",
    )
    q: Optional[str] = Query(
        None,
        description="Return only traces containing this text in their message ID, conversation ID, conversation history, output, or metadata",
    )
    output_type: Optional[Literal["response", "tool_call"]] = Query(
        None,
        description="Return only traces whose output is of this kind. `response` covers every trace carrying a reply, including one that also issued tool calls. `tool_call` covers traces that only issued tool calls",
    )
    labels: Optional[List[TraceLabel]] = Query(
        None,
        description="Return only traces carrying at least one of these labels. Repeat the parameter for each label",
    )
    score: Optional[List[str]] = Query(
        None, description=_SCORE_FILTER_DESCRIPTION
    )
    created_after: Optional[datetime] = Query(
        None,
        description="Return only traces created at or after this time (ISO 8601). A time with no timezone is read as UTC. Omit for no earliest date",
    )
    created_before: Optional[datetime] = Query(
        None,
        description="Return only traces created before this time (ISO 8601). A time with no timezone is read as UTC. Omit for no latest date",
    )
    source: Optional[TraceSource] = Query(
        None,
        description="Return only traces from this source:" + _SOURCE_VALUES,
    )
    feedback: Optional[TraceFeedbackFilter] = Query(
        None,
        description="Return only traces with this feedback:" + _FEEDBACK_VALUES,
    )


@dataclass
class _TraceListOptions:
    sort_by_evaluator: Optional[str] = Query(
        None, description=_SORT_BY_EVALUATOR_DESCRIPTION
    )
    sort_order: Literal["asc", "desc"] = Query(
        "desc", description=_SORT_ORDER_DESCRIPTION
    )
    include_content: bool = Query(
        False,
        description="Also return each trace's full input, output and metadata. Use `limit` and `offset` to page through them",
    )
    include_score_averages: bool = Query(
        False,
        description="Also return each evaluator's mean score over every matching trace, not just this page. Costs a pass over those traces, so ask for it when the filters change rather than on every page",
    )


@router.get(
    "",
    response_model=TraceListResponse,
    summary="List traces",
    tags=["Public API"],
)
def list_traces_endpoint(
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
    filters: _TraceListFilters = Depends(),
    options: _TraceListOptions = Depends(),
    pagination: PaginationParams = Depends(),
):
    """List ingested traces, newest first"""
    trace_filters = _checked_query_filters(filters, pagination)
    # Search/filter/count run in SQL (db.list_traces), not the post-fetch
    # pagination helpers, and paging uses the bounded PaginationParams rather
    # than the unbounded OptionalPaginationParams: traces are machine-written
    # and outgrow in-memory filtering fast.
    rows, total = list_traces(
        ctx.org_uuid,
        limit=pagination.limit,
        offset=pagination.offset,
        filters=trace_filters,
        sort_evaluator=options.sort_by_evaluator,
        sort_order=options.sort_order,
    )
    scoring = get_latest_trace_run_summaries(
        ctx.org_uuid, [row["uuid"] for row in rows]
    )
    items = [_to_summary(row, scoring.get(row["uuid"])) for row in rows]
    if options.include_content:
        for item, row in zip(items, rows):
            item.update(input=row["input"], output=row["output"], metadata=row["metadata"])
    body = page_envelope(
        items,
        total,
        pagination,
    )
    if options.include_score_averages:
        body["score_averages"] = average_trace_scores(
            ctx.org_uuid, filters=trace_filters
        )
    return body


class TraceConversationSummary(BaseModel):
    conversation_id: str = Field(description="The conversation ID you sent with its turns")
    agent_id: str = Field(description=_AGENT_ID_DESCRIPTION)
    first_turn_at: str = Field(description="When the first trace in this conversation was created (ISO 8601 UTC)")
    last_turn_at: str = Field(description="When the newest trace in this conversation was created (ISO 8601 UTC)")
    first_message_preview: Optional[str] = Field(
        None,
        description="The user message of the conversation's first trace, truncated for display",
    )
    last_turn: TraceSummary = Field(description="The newest trace in this conversation")


@dataclass
class _ConversationFilters:
    agent_id: Optional[str] = Query(
        None, description="Return only conversations with this agent"
    )
    conversation_id: Optional[str] = Query(
        None,
        min_length=1,
        max_length=255,
        description="Return only the conversation with this ID. Requires `agent_id`",
    )
    q: Optional[str] = Query(
        None,
        description="Return only conversations with a turn containing this text in its message ID, conversation ID, conversation history, output, or metadata",
    )
    output_type: Optional[Literal["response", "tool_call"]] = Query(
        None,
        description="Return only conversations with a turn whose output is of this kind",
    )
    labels: Optional[List[TraceLabel]] = Query(
        None,
        description="Return only conversations with a turn carrying at least one of these labels. Repeat the parameter for each label",
    )
    score: Optional[List[str]] = Query(
        None,
        description="Return only conversations with a turn whose latest scoring run matches every one of these conditions. " + _SCORE_CONDITION_SYNTAX,
    )
    source: Optional[TraceSource] = Query(
        None,
        description="Return only conversations with a turn from this source:" + _SOURCE_VALUES,
    )
    feedback: Optional[TraceFeedbackFilter] = Query(
        None,
        description="Return only conversations with a turn with this feedback:" + _FEEDBACK_VALUES,
    )
    # Not parameters: this route has no date filters.
    created_after: ClassVar[None] = None
    created_before: ClassVar[None] = None


@router.get(
    "/conversations",
    response_model=PaginatedResponse[TraceConversationSummary],
    summary="List trace conversations",
)
def list_trace_conversations_endpoint(
    ctx: OrgContext = Depends(get_current_org),
    filters: _ConversationFilters = Depends(),
    pagination: PaginationParams = Depends(),
):
    """List conversations, newest turn first. Traces sent without a conversation ID are left out"""
    trace_filters = _checked_query_filters(filters, pagination)
    conversations, total = list_trace_conversations(
        ctx.org_uuid,
        limit=pagination.limit,
        offset=pagination.offset,
        filters=trace_filters,
    )
    scoring = get_latest_trace_run_summaries(
        ctx.org_uuid, [c["last_turn"]["uuid"] for c in conversations]
    )
    for c in conversations:
        c["first_message_preview"] = _preview(
            _last_user_content(c.pop("first_input"))
        )
        c["last_turn"] = _to_summary(c["last_turn"], scoring.get(c["last_turn"]["uuid"]))
    return page_envelope(conversations, total, pagination)


class TraceUsageResponse(BaseModel):
    traces_stored: int = Field(description="How many traces you are storing")
    max_traces: int = Field(description="How many traces you may store")
    traces_scored: int = Field(
        description="How many scoring runs count against your scoring limit, one per trace scored. A run that failed does not count"
    )
    max_scored_traces: Optional[int] = Field(
        description="How many traces you may have scored. No limit (`null`) while your workspace uses its own AI provider keys"
    )


@router.get(
    "/usage",
    response_model=TraceUsageResponse,
    summary="Get your trace usage against your limits",
)
def get_trace_usage_endpoint(ctx: OrgContext = Depends(get_current_org)):
    """Report how much of each trace limit you have used"""
    usage_limits = effective_limits(ctx.org_uuid)
    return {
        "traces_stored": count_live_traces(ctx.org_uuid),
        "max_traces": usage_limits["max_traces"],
        "traces_scored": count_scored_traces(ctx.org_uuid),
        "max_scored_traces": usage_limits["max_scored_traces"],
    }


@router.get(
    "/labels",
    response_model=TraceLabelsResponse,
    summary="List trace labels",
)
def list_trace_labels_endpoint(
    ctx: OrgContext = Depends(get_current_org),
    agent_id: Optional[str] = Query(
        None, description="Return only labels used by this agent's traces"
    ),
):
    """List every label in use, so a filter can offer the whole set"""
    return {"labels": list_trace_labels(ctx.org_uuid, agent_id=agent_id)}


@router.get(
    "/conversation-ids",
    response_model=TraceConversationIdsResponse,
    summary="List trace conversation IDs",
)
def list_trace_conversation_ids_endpoint(
    ctx: OrgContext = Depends(get_current_org),
    agent_id: Optional[str] = Query(
        None, description="Return only conversation IDs from this agent's traces"
    ),
):
    """List every conversation ID in use, so a filter can offer the whole set"""
    return {
        "conversation_ids": list_trace_conversation_ids(
            ctx.org_uuid, agent_id=agent_id
        )
    }


@router.post(
    "/bulk-delete",
    response_model=BulkDeleteTracesResponse,
    summary="Bulk delete traces",
)
def bulk_delete_traces(
    payload: BulkDeleteTracesRequest, ctx: OrgContext = Depends(get_current_org)
):
    """Soft-delete traces, freeing their capacity"""
    if payload.select_all:
        deleted = soft_delete_traces_matching(
            ctx.org_uuid, filters=_to_trace_filters(payload)
        )
    else:
        deleted = soft_delete_traces(ctx.org_uuid, trace_ids=payload.trace_ids)
    return {"deleted": deleted}


_CONVERT_TYPE_DESCRIPTION = (
    "What the created tests judge:\n\n"
    "- `response`: re-run the agent on the trace's conversation and judge the fresh reply against the linked evaluators\n"
    "- `general`: re-run the agent on the trace's standalone prompt and judge the fresh reply against the linked evaluators\n"
    "- `tool_call`: re-run the agent and diff its tool calls against the ones the trace recorded\n\n"
    "`response` needs traces carrying a conversation and `general` needs traces "
    "carrying a standalone prompt, so each one fits the agent it came from. "
    "`tool_call` takes either and keeps the shape the trace has"
)


class ConvertTracesToTestsRequest(_TraceSelection):
    max_trace_ids: ClassVar[int] = MAX_CONVERT_TRACES

    trace_ids: List[TraceUuid] = Field(
        default_factory=list,
        description="IDs of the traces to convert, one test per trace. Omit when `select_all` is on",
    )
    type: Literal["response", "general", "tool_call"] = Field(
        description=_CONVERT_TYPE_DESCRIPTION
    )
    evaluators: Optional[List[EvaluatorUuid]] = Field(
        None,
        description="IDs of the evaluators to link to every created test. **Required for `response` and `general`**, rejected for `tool_call`, which compares the recorded calls instead of judging. Each evaluator must match the created tests: `llm` for `response`, `llm-general` for `general`. An evaluator with `{{placeholder}}` variables needs a part of a trace chosen for each of them on the agent of every trace, and each test keeps the values its own trace holds",
    )
    skip_missing_values: bool = Field(
        False,
        description="Leave an evaluator off the tests whose trace holds no value for one of its variables, and leave out a trace that is left with no evaluator at all. Omit to create nothing when any trace is missing a value",
    )
    accept_any_arguments: bool = Field(
        False,
        description="For `tool_call`, match only the tool name and ignore the arguments the trace recorded",
    )
    name: Optional[str] = Field(
        None,
        description="The name to give the test. Only when converting exactly one trace by its ID. Omit to name it after the trace's `message_id`, else the last user message",
    )


class ConvertedEvaluatorLeftOff(BaseModel):
    evaluator_id: str = Field(description="ID of the evaluator left off")
    evaluator_name: str = Field(description="Name of the evaluator left off")
    test_uuids: List[str] = Field(
        description="IDs of the new tests created without this evaluator"
    )


class ConvertTracesToTestsResponse(BaseModel):
    created: int = Field(description="Number of tests created")
    test_uuids: List[str] = Field(
        description="IDs of the created tests, in the order their traces were given. A trace that could not be added is left out and named in warnings",
        examples=[[EXAMPLE_TEST_UUID]],
    )
    warnings: Optional[List[str]] = Field(
        None,
        description="Traces that could not be added to tests, and tests created without an evaluator, named with what to do about them",
    )
    evaluators_left_off: Optional[List[ConvertedEvaluatorLeftOff]] = Field(
        None,
        description="With `skip_missing_values`, each evaluator left off some of the new tests because their traces hold no value for its variables",
    )


def _resolve_evaluators(
    evaluator_uuids: List[str], org_uuid: str, test_type: str
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    """Validate as POST /tests does and reject an evaluator with no live version.
    Returns the refs, each evaluator's name, and the evaluators whose live
    version declares variables, which `_evaluator_refs_per_trace` fills."""
    refs = _validate_evaluators(
        [EvaluatorRef(evaluator_uuid=u) for u in evaluator_uuids], org_uuid, test_type
    )
    evaluators = get_evaluators_by_uuids(evaluator_uuids)
    live_ids = {
        u: (evaluators.get(u) or {}).get("live_version_id") for u in evaluator_uuids
    }
    versions = get_evaluator_versions_by_uuids([v for v in live_ids.values() if v])

    problems: List[str] = []
    names: Dict[str, str] = {}
    with_variables: List[str] = []
    for evaluator_uuid in evaluator_uuids:
        # Deleted between the two reads: report it rather than raising KeyError.
        name = (evaluators.get(evaluator_uuid) or {}).get("name", evaluator_uuid)
        names[evaluator_uuid] = name
        version = versions.get(live_ids[evaluator_uuid] or "")
        if not version:
            problems.append(f'Evaluator "{name}" has no live version to run.')
        elif version.get("variables"):
            with_variables.append(evaluator_uuid)
    _raise_evaluator_problems(problems)
    return refs, names, with_variables


def _raise_evaluator_problems(problems: List[str]) -> None:
    if problems:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Some evaluators cannot be used for converted tests",
                "evaluators": problems,
            },
        )


def agent_evaluator_sources(
    agent_id: str, evaluator_ids: List[str]
) -> Tuple[Dict[str, Dict[str, str]], List[Tuple[str, Optional[Dict[str, Any]], List[str]]]]:
    """The part of a trace chosen on the agent for each variable of each linked
    evaluator, and the evaluators that cannot be filled, as (evaluator id, the
    evaluator or None when it is not linked, the variables with no part chosen)."""
    linked = {ev["uuid"]: (ev, v) for ev, v in resolve_live_evaluators(agent_id)}
    sources: Dict[str, Dict[str, str]] = {}
    unfillable: List[Tuple[str, Optional[Dict[str, Any]], List[str]]] = []
    for evaluator_id in evaluator_ids:
        if evaluator_id not in linked:
            unfillable.append((evaluator_id, None, []))
            continue
        ev, version = linked[evaluator_id]
        sources[evaluator_id], unset = evaluator_trace_sources(ev, version)
        if unset:
            unfillable.append((evaluator_id, ev, unset))
    return sources, unfillable


def traces_or_404(
    org_uuid: str, trace_ids: List[str], agent_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """The traces with these ids (of `agent_id` only, when given), or a 404
    naming the ones not found."""
    requested = list(dict.fromkeys(trace_ids))
    traces = [
        trace
        for trace in get_traces_by_uuids(org_uuid, requested)
        if agent_id is None or trace["agent_id"] == agent_id
    ]
    found = {trace["uuid"] for trace in traces}
    missing = [trace_id for trace_id in requested if trace_id not in found]
    if missing:
        raise HTTPException(
            status_code=404,
            detail={"error": "Some traces were not found", "trace_ids": missing},
        )
    return traces


def _evaluator_refs_per_trace(
    traces: List[Dict[str, Any]],
    refs: List[Dict[str, Any]],
    names: Dict[str, str],
    with_variables: List[str],
    skip_missing_values: bool,
    agents: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Each trace's evaluator refs. An evaluator with variables carries the values
    its trace holds, read through the parts of a trace chosen on that trace's
    agent, and is left off a trace missing one when `skip_missing_values` is on.
    A trace whose agent is gone (None in `agents`) keeps `refs` as they are: its
    test cannot be linked and is dropped with a warning later."""
    if not with_variables:
        return {trace["uuid"]: refs for trace in traces}
    sources_by_agent: Dict[str, List[Dict[str, Any]]] = {}
    problems: List[str] = []
    for agent_id in dict.fromkeys(trace["agent_id"] for trace in traces):
        if not agents[agent_id]:
            continue
        agent_name = agents[agent_id].get("name") or agent_id
        sources, unfillable = agent_evaluator_sources(agent_id, with_variables)
        for evaluator_uuid, ev, unset in unfillable:
            name = names[evaluator_uuid]
            problems.append(
                f'Evaluator "{name}" is not linked to agent "{agent_name}". Link '
                "it and choose which part of a trace fills each of its variables."
                if ev is None
                else f'Evaluator "{name}" has no part of a trace chosen on agent '
                f'"{agent_name}" for: {", ".join(unset)}.'
            )
        sources_by_agent[agent_id] = [
            {"uuid": uuid, "trace_variables": s} for uuid, s in sources.items()
        ]
    _raise_evaluator_problems(problems)

    out: Dict[str, List[Dict[str, Any]]] = {}
    missing: List[Dict[str, str]] = []
    for trace in traces:
        if trace["agent_id"] not in sources_by_agent:
            out[trace["uuid"]] = refs
            continue
        included, gaps = trace_evaluators(trace, sources_by_agent[trace["agent_id"]])
        values = {ev["uuid"]: ev["variable_values"] for ev in included}
        missing.extend(
            {
                "trace_id": trace["uuid"],
                "evaluator_id": gap["evaluator_uuid"],
                "evaluator_name": names[gap["evaluator_uuid"]],
                "variable": gap["variable"],
                "source": gap["source"],
            }
            for gap in gaps
        )
        out[trace["uuid"]] = [
            {**ref, "variable_values": values[ref["evaluator_id"]]}
            if ref["evaluator_id"] in values
            else ref
            for ref in refs
            if ref["evaluator_id"] not in with_variables
            or ref["evaluator_id"] in values
        ]
    if missing and not skip_missing_values:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Some traces hold no value for an evaluator's variables, so "
                    "nothing was created. Pass skip_missing_values to leave the "
                    "evaluator off those tests"
                ),
                "missing": missing,
            },
        )
    return out


# How many offending IDs a conflict lists before it just reports the count. A
# `select_all` conversion can hit hundreds, and a wall of UUIDs buries the hint
# that actually resolves it.
_MAX_REPORTED_CONFLICTS = 20


def _shape_conflict_detail(
    error: str,
    offending: List[str],
    considered: int,
    *,
    select_all: bool,
    hint: str,
) -> Dict[str, Any]:
    """Build the 400 body for traces the requested test type cannot take."""
    detail: Dict[str, Any] = {
        "error": error,
        "trace_ids": offending[:_MAX_REPORTED_CONFLICTS],
    }
    if select_all:
        detail["error"] = (
            f"{error}. {len(offending)} of the {considered} matching traces "
            "cannot be converted, so nothing was created"
        )
        detail["hint"] = hint
        if len(offending) > _MAX_REPORTED_CONFLICTS:
            detail["trace_ids_truncated"] = True
    return detail


def _dedupe_test_names(candidates: List[str], taken: set) -> List[str]:
    """Make each candidate unique against `taken` (the workspace's existing test
    names, mutated as names are claimed) by appending ` (2)`, ` (3)`, … so
    converting the same traces twice creates new tests instead of failing."""
    out: List[str] = []
    for base in candidates:
        name = base
        n = 2
        while name in taken:
            name = f"{base} ({n})"
            n += 1
        taken.add(name)
        out.append(name)
    return out


def _test_names_line(names: List[str], shown: int = 5) -> str:
    """The first few test names, then how many more, for a warning line."""
    line = ", ".join(names[:shown])
    if len(names) > shown:
        line += f" and {len(names) - shown} more"
    return line


@router.post(
    "/convert-to-tests",
    response_model=ConvertTracesToTestsResponse,
    summary="Convert traces to tests",
)
def convert_traces_to_tests(
    payload: ConvertTracesToTestsRequest, ctx: OrgContext = Depends(get_current_org)
):
    """Turn production traces into regression tests you can run and benchmark"""
    # A converted response or general test re-runs the agent and judges the fresh
    # reply, so it has no fallback judge. A tool_call run only diffs the recorded
    # calls (see the row-type skip in agent_tests._build_calibrate_config), so an
    # evaluator attached here would never judge anything: refuse it rather than
    # store a judge the user never sees run.
    if payload.type in ("response", "general") and not payload.evaluators:
        raise HTTPException(
            status_code=400,
            detail=f"{payload.type} tests require at least one evaluator",
        )
    if payload.type == "tool_call" and payload.evaluators:
        raise HTTPException(
            status_code=400,
            detail="tool_call tests compare the recorded tool calls and cannot take evaluators",
        )
    name = payload.name.strip() if payload.name is not None else None
    if payload.name is not None and (
        not name or payload.select_all or len(set(payload.trace_ids)) != 1
    ):
        raise HTTPException(
            status_code=400,
            detail="A name can only be given when converting exactly one trace, and cannot be blank",
        )
    resolved_refs: List[Dict[str, Any]] = []
    evaluator_names: Dict[str, str] = {}
    with_variables: List[str] = []
    if payload.evaluators:
        resolved_refs, evaluator_names, with_variables = _resolve_evaluators(
            list(dict.fromkeys(payload.evaluators)), ctx.org_uuid, payload.type
        )

    if payload.select_all:
        traces, total = list_traces(
            ctx.org_uuid,
            limit=MAX_CONVERT_TRACES,
            offset=0,
            filters=_to_trace_filters(payload),
        )
        if total > MAX_CONVERT_TRACES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{total} traces match, more than the {MAX_CONVERT_TRACES} "
                    "one conversion accepts. Narrow the filters and convert in batches."
                ),
            )
        if not traces:
            raise HTTPException(
                status_code=400, detail="No traces matched the filters"
            )
    else:
        traces = traces_or_404(ctx.org_uuid, payload.trace_ids)

    # `general` tests carry a standalone `input` and `response` tests a
    # `history`; calibrate refuses a row holding the wrong one. Since ingest ties
    # each trace's shape to its agent, this also keeps every created test
    # linkable to the agent it came from.
    if payload.type in ("response", "general"):
        wants_text = payload.type == "general"
        wrong_shape = [
            trace["uuid"]
            for trace in traces
            if isinstance(trace["input"], str) is not wants_text
        ]
        if wrong_shape:
            raise HTTPException(
                status_code=400,
                detail=_shape_conflict_detail(
                    (
                        "general tests take a standalone prompt, but these traces "
                        "carry a conversation"
                        if wants_text
                        else "response tests take a conversation, but these traces "
                        "carry a standalone prompt"
                    ),
                    wrong_shape,
                    len(traces),
                    select_all=payload.select_all,
                    hint=(
                        "Pass agent_id to convert one agent's traces at a time. "
                        "A trace's shape follows the agent that produced it, so a "
                        f"single {'general' if wants_text else 'conversation'} "
                        "agent's traces are all the right shape."
                    ),
                ),
            )

    if payload.type == "tool_call":
        no_calls = [
            trace["uuid"]
            for trace in traces
            if not (trace["output"] or {}).get("tool_calls")
        ]
        if no_calls:
            raise HTTPException(
                status_code=400,
                detail=_shape_conflict_detail(
                    "Some traces recorded no tool calls to assert",
                    no_calls,
                    len(traces),
                    select_all=payload.select_all,
                    hint=(
                        "Pass output_type=tool_call to convert only the traces "
                        "that issued tool calls."
                    ),
                ),
            )

    # The agent each trace came from, or None once it is deleted or moved to
    # another workspace. Read once here for the variable values and the linking.
    agents_seen: Dict[str, Optional[Dict[str, Any]]] = {}
    for agent_id in dict.fromkeys(trace["agent_id"] for trace in traces):
        found = get_agent(agent_id)
        agents_seen[agent_id] = (
            found if found and found["org_uuid"] == ctx.org_uuid else None
        )

    refs_by_trace = _evaluator_refs_per_trace(
        traces,
        resolved_refs,
        evaluator_names,
        with_variables,
        payload.skip_missing_values,
        agents_seen,
    )

    # calibrate refuses a response or general test with no evaluator, so a trace
    # whose every chosen evaluator was left off for a missing value is left out.
    no_evaluator = [t for t in traces if resolved_refs and not refs_by_trace[t["uuid"]]]
    if no_evaluator:
        traces = [t for t in traces if refs_by_trace[t["uuid"]]]
        left_out_line = (
            f"{len(no_evaluator)} of the {len(traces) + len(no_evaluator)} traces "
            "were left out, because they hold no value for the variables of any "
            "chosen evaluator: "
            f"{_test_names_line([_test_name_for_trace(t) for t in no_evaluator])}."
        )
        if not traces:
            raise HTTPException(status_code=400, detail=left_out_line)

    existing_names = {t["name"] for t in get_all_tests_summary(org_uuid=ctx.org_uuid)}
    names = _dedupe_test_names(
        [name or _test_name_for_trace(trace) for trace in traces], existing_names
    )

    db_tests: List[Dict[str, Any]] = []
    for trace, name in zip(traces, names):
        evaluation: Dict[str, Any] = {"type": payload.type}
        if payload.type == "tool_call":
            evaluation["tool_calls"] = [
                {
                    "tool": call["tool"],
                    "arguments": call.get("arguments"),
                    "accept_any_arguments": payload.accept_any_arguments,
                }
                for call in trace["output"]["tool_calls"]
            ]
        # A trace's `input` is already the test config field it maps to: a
        # standalone prompt becomes `input`, a conversation becomes `history`.
        # The recorded reply is dropped for a response or general test (the agent
        # is re-run) and captured as the assertion for a tool_call test above.
        stored_input = trace["input"]
        given = (
            {"input": stored_input}
            if isinstance(stored_input, str)
            else {"history": stored_input}
        )
        db_tests.append(
            {
                "name": name,
                "type": payload.type,
                "config": {**given, "evaluation": evaluation},
            }
        )

    try:
        test_uuids = bulk_create_tests(
            tests=db_tests, org_uuid=ctx.org_uuid, user_id=ctx.user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Each test links to the agent that produced its trace. A test that cannot
    # be linked (its agent deleted since ingest, switched to an interaction_type
    # the test no longer fits, or a failed write) would sit in the workspace
    # where no screen lists it, so it is deleted and the reader is told to retry.
    unlinked: List[str] = []
    unlinked_uuids: List[str] = []
    kept: List[Tuple[str, str, List[Dict[str, Any]]]] = []
    for trace, test_uuid, db_test in zip(traces, test_uuids, db_tests):
        agent_id = trace["agent_id"]
        agent = agents_seen[agent_id]
        linked = bool(agent) and required_agent_interaction_type(
            payload.type, db_test["config"]
        ) == (agent.get("interaction_type") or DEFAULT_AGENT_INTERACTION_TYPE)
        if linked:
            try:
                add_test_to_agent(agent_id, test_uuid)
            except Exception as e:
                linked = False
                logger.warning(f"Failed to link test {test_uuid} to agent {agent_id}: {e}")
        if linked:
            kept.append((test_uuid, db_test["name"], refs_by_trace[trace["uuid"]]))
        else:
            unlinked.append(db_test["name"])
            unlinked_uuids.append(test_uuid)
    bulk_delete_tests(unlinked_uuids, ctx.org_uuid)
    if not kept:
        raise HTTPException(
            status_code=400,
            detail="None of these traces could be added to their agent's tests. "
            "Try again.",
        )

    # The kept tests are committed and linked, so a failed evaluator link must
    # not 500 and lose them: name them so the reader can add one in place.
    warnings: List[str] = [left_out_line] if no_evaluator else []
    if unlinked:
        warnings.append(
            f"{len(unlinked)} of the {len(test_uuids)} traces could not be added "
            f"to tests: {_test_names_line(unlinked)}. Add those traces to tests "
            "again."
        )
    unjudged: List[str] = []
    if resolved_refs:
        for test_uuid, name, refs in kept:
            try:
                set_test_evaluators(test_uuid, refs)
            except Exception as e:
                unjudged.append(name)
                logger.warning(f"Failed to link evaluators to test {test_uuid}: {e}")
    if unjudged:
        warnings.append(
            f"The evaluators could not be added to {len(unjudged)} of the "
            f"{len(kept)} new tests: {_test_names_line(unjudged)}. Open each one "
            "on the agent's Tests tab and add the evaluators again."
        )
    left_off = []
    for evaluator_uuid in with_variables:
        test_uuids = [
            test_uuid
            for test_uuid, _, refs in kept
            if all(r["evaluator_id"] != evaluator_uuid for r in refs)
        ]
        if test_uuids:
            left_off.append(
                {
                    "evaluator_id": evaluator_uuid,
                    "evaluator_name": evaluator_names[evaluator_uuid],
                    "test_uuids": test_uuids,
                }
            )

    return {
        "created": len(kept),
        "test_uuids": [test_uuid for test_uuid, _, _ in kept],
        "warnings": warnings or None,
        "evaluators_left_off": left_off or None,
    }


@router.get(
    "/{trace_uuid}/scores",
    response_model=TraceScoresResponse,
    summary="Get trace scores",
)
def get_trace_scores_endpoint(
    trace_uuid: str = Path(
        description="The trace whose scoring runs to return",
        examples=[_EXAMPLE_TRACE_UUID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get every scoring run for this trace, newest first"""
    if not get_trace(ctx.org_uuid, trace_uuid):
        raise HTTPException(status_code=404, detail="Trace not found")
    runs = list_trace_scoring_runs(ctx.org_uuid, trace_uuid)
    return {
        "runs": [
            {
                "run_uuid": run["run_uuid"],
                "status": run["status"],
                "created_at": _trace_iso(run["created_at"]),
                "completed_at": _trace_iso(run["completed_at"]),
                "error": run["error"],
                "results": run["results"],
                "missing_values": run["missing_values"],
                "unconfigured_evaluators": run["unconfigured_evaluators"],
                "newly_scored_evaluators": run["newly_scored_evaluators"],
            }
            for run in runs
        ]
    }


class TraceScoreRunResponse(BaseModel):
    run_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the new scoring run",
        examples=["9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"],
    )
    status: Literal["pending"] = Field(
        description="Status of the new run, which waits for a worker to judge it"
    )


@router.post(
    "/{trace_uuid}/score",
    response_model=TraceScoreRunResponse,
    summary="Score trace",
)
def score_trace_endpoint(
    trace_uuid: str = Path(
        description="The trace to score again",
        examples=[_EXAMPLE_TRACE_UUID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Score a trace again: with every evaluator when it was skipped or failed, or with only the evaluators that were not set up when it was scored"""
    trace = get_trace(ctx.org_uuid, trace_uuid)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    agent = ensure_owned_agent(trace["agent_id"], ctx.org_uuid)
    if not trace_scoring_enabled(agent):
        raise HTTPException(
            status_code=400, detail="Trace scoring is turned off for this agent"
        )
    run_uuid, refusal = create_trace_rescore_run(
        org_uuid=ctx.org_uuid,
        trace_uuid=trace_uuid,
        agent_id=agent["uuid"],
        max_scored_traces=effective_limits(ctx.org_uuid)["max_scored_traces"],
    )
    if refusal == "open_run":
        raise HTTPException(
            status_code=409, detail="This trace is already being scored"
        )
    if refusal == "not_retryable":
        raise HTTPException(
            status_code=409,
            detail="Only a trace that was skipped or failed, or that has an evaluator set up since it was scored, can be scored again",
        )
    if refusal == "over_limit":
        raise HTTPException(
            status_code=429,
            detail="This workspace has reached its limit of scored traces",
        )
    return {"run_uuid": run_uuid, "status": "pending"}


@router.get("/{trace_uuid}", response_model=TraceResponse, summary="Get trace")
def get_trace_endpoint(
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


class TraceFeedbackResponse(BaseModel):
    feedback: Optional[TraceFeedback] = Field(
        description="The feedback now on the trace"
    )


@router.put(
    "/{trace_uuid}/feedback",
    response_model=TraceFeedbackResponse,
    summary="Set trace feedback",
    tags=["Public API"],
)
def set_trace_feedback_endpoint(
    body: TraceFeedbackUpdate,
    trace_uuid: str = Path(
        description="The trace to give feedback on",
        examples=[_EXAMPLE_TRACE_UUID],
    ),
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """Set whether the agent's output on a trace was correct, replacing any earlier feedback, or clear it"""
    if not set_trace_feedback(ctx.org_uuid, trace_uuid, body.is_correct, body.comment):
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"feedback": get_trace(ctx.org_uuid, trace_uuid)["feedback"]}
