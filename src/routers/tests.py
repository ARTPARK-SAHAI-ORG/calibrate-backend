from typing import ClassVar, Optional, List, Dict, Any, Literal
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, ConfigDict, model_validator

from db import (
    create_test,
    ensure_name_unique,
    get_test,
    get_all_tests,
    get_all_tools,
    update_test,
    delete_test,
    bulk_create_tests,
    bulk_delete_tests,
    get_agent,
    add_test_to_agent,
    get_evaluator,
    get_evaluators_for_test,
    set_test_evaluators,
    iter_tool_call_entries,
    inject_tool_uuids_into_config,
    refresh_tool_call_names_in_config,
)
from auth_utils import get_current_org, OrgContext

import logging

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/tests", tags=["tests"])


TestType = Literal["response", "tool_call", "conversation"]

# Each test type pins the evaluator_type it accepts. `conversation` tests judge whole
# simulated conversations, so only `conversation` evaluators apply; `response`/`tool_call`
# tests judge a single LLM reply, so only `llm` evaluators apply.
REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE: Dict[str, str] = {
    "response": "llm",
    "tool_call": "llm",
    "conversation": "conversation",
}


class EvaluatorRef(BaseModel):
    """Reference to an evaluator attached to a test. The pinned version is always the
    evaluator's live version at write time (`set_test_evaluators` in `db.py`)."""

    model_config = ConfigDict(extra="forbid")

    evaluator_uuid: str
    variable_values: Optional[Dict[str, Any]] = None


class TestCreate(BaseModel):
    name: str
    type: TestType
    config: Optional[Dict[str, Any]] = None
    evaluators: Optional[List[EvaluatorRef]] = None


class TestUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[TestType] = None
    config: Optional[Dict[str, Any]] = None
    evaluators: Optional[List[EvaluatorRef]] = None


class TestResponse(BaseModel):
    uuid: str
    name: str
    type: str
    config: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str
    evaluators: List[Dict[str, Any]] = []


class TestCreateResponse(BaseModel):
    uuid: str
    message: str


# --- Bulk upload models ---

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ExpectedToolCall(BaseModel):
    tool: str
    # Durable link to the `tools` entity. Stamped server-side at write time by
    # matching `tool` to a live tool in the org; the live name is re-resolved from
    # it on read so a renamed tool propagates. `tool` stays as a fallback snapshot.
    tool_uuid: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    accept_any_arguments: bool = False


class BulkTestItem(BaseModel):
    name: str
    conversation_history: List[ChatMessage]
    evaluators: Optional[List[EvaluatorRef]] = None
    tool_calls: Optional[List[ExpectedToolCall]] = None


class BulkTestUpload(BaseModel):
    type: TestType
    tests: List[BulkTestItem]
    agent_uuids: Optional[List[str]] = None
    language: Optional[str] = None

    MAX_BATCH_SIZE: ClassVar[int] = 500

    @model_validator(mode="after")
    def validate_tests(self):
        if not self.tests:
            raise ValueError("tests list must not be empty")

        if len(self.tests) > self.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(self.tests)} exceeds maximum of {self.MAX_BATCH_SIZE}")

        names = [t.name for t in self.tests]
        if len(names) != len(set(names)):
            seen = set()
            dupes = sorted({n for n in names if n in seen or seen.add(n)})
            raise ValueError(f"Duplicate test names in request: {', '.join(dupes)}")

        for t in self.tests:
            if not t.conversation_history:
                raise ValueError(f"Test '{t.name}' must have at least one message in conversation_history")
            if self.type == "response":
                if not t.evaluators:
                    raise ValueError(
                        f"Test '{t.name}' must have at least one evaluator for response type"
                    )
            elif self.type == "tool_call":
                if not t.tool_calls:
                    raise ValueError(f"Test '{t.name}' must have 'tool_calls' for tool_call type")
            elif self.type == "conversation":
                if not t.evaluators:
                    raise ValueError(
                        f"Test '{t.name}' must have at least one evaluator for conversation type"
                    )

        return self


class BulkTestUploadResponse(BaseModel):
    uuids: List[str]
    count: int
    message: str
    warnings: Optional[List[str]] = None


class BulkTestDelete(BaseModel):
    test_uuids: List[str]


class BulkTestDeleteResponse(BaseModel):
    deleted_count: int
    message: str


def _validate_evaluators(
    refs: List[EvaluatorRef], org_uuid: str, test_type: str
) -> List[Dict[str, Any]]:
    """Validate that each referenced evaluator is visible to the org and that its
    `evaluator_type` matches the test's type (`response`/`tool_call` ⇒ `llm`,
    `conversation` ⇒ `simulation`). Returns db-ready refs."""
    required_evaluator_type = REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE.get(test_type)
    if required_evaluator_type is None:
        raise HTTPException(
            status_code=400, detail=f"Unknown test type '{test_type}'"
        )
    out: List[Dict[str, Any]] = []
    for ref in refs:
        evaluator = get_evaluator(ref.evaluator_uuid)
        if not evaluator:
            raise HTTPException(status_code=404, detail=f"Evaluator {ref.evaluator_uuid} not found")
        if evaluator.get("org_uuid") is not None and evaluator["org_uuid"] != org_uuid:
            raise HTTPException(status_code=404, detail=f"Evaluator {ref.evaluator_uuid} not found")
        if evaluator.get("evaluator_type") != required_evaluator_type:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Evaluator {ref.evaluator_uuid} has evaluator_type="
                    f"'{evaluator.get('evaluator_type')}'. Tests of type "
                    f"'{test_type}' only accept '{required_evaluator_type}' evaluators."
                ),
            )
        out.append(
            {
                "evaluator_id": ref.evaluator_uuid,
                "evaluator_version_id": None,
                "variable_values": ref.variable_values,
            }
        )
    return out


def _org_tool_indexes(org_uuid: str) -> tuple[Dict[str, str], Dict[str, str]]:
    """Return (name→uuid, uuid→name) maps for the org's live (non-deleted) tools.
    Tool names are unique per org, so name→uuid is unambiguous."""
    tools = get_all_tools(org_uuid=org_uuid)
    name_to_uuid = {t["name"]: t["uuid"] for t in tools}
    uuid_to_name = {t["uuid"]: t["name"] for t in tools}
    return name_to_uuid, uuid_to_name


def _resolve_tool_call_uuids(
    config: Optional[Dict[str, Any]], uuid_to_name: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """Resolution for interactive writes (`POST`/`PUT /tests`): `tool_uuid` is
    **optional but validated**.

    - Entry WITH a `tool_uuid` → it must resolve to a live tool in the caller's org
      (404 otherwise); the live name is stamped into `tool`. These are library tools
      and get rename-tracking.
    - Entry WITHOUT a `tool_uuid` → allowed as a name-only snapshot. This covers
      built-in / agent-owned tools (agent-connection mode, framework tools) that have
      no `tools` row and therefore no uuid to send.

    Mutates and returns `config`. No-op for non-`tool_call` configs.
    """
    for tc in iter_tool_call_entries(config):
        tool_uuid = tc.get("tool_uuid")
        if not tool_uuid:
            # Built-in / agent-owned tool: no uuid, keep the name snapshot.
            continue
        name = uuid_to_name.get(tool_uuid)
        if name is None:
            raise HTTPException(
                status_code=404, detail=f"Tool {tool_uuid} not found"
            )
        tc["tool"] = name
    return config


def _normalize_evaluation_type(
    config: Optional[Dict[str, Any]], row_type: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Derive `config.evaluation.type` from the immutable row `type` — the single
    source of truth. The blob copy only exists because the whole config is stored
    verbatim and shipped to the calibrate CLI; deriving it from the row type on every
    write and read keeps the two from drifting (the run path re-stamps it too).
    Mutates and returns `config`. No-op when there's no `evaluation` block."""
    if isinstance(config, dict) and row_type:
        evaluation = config.get("evaluation")
        if isinstance(evaluation, dict):
            evaluation["type"] = row_type
    return config


def _link_tool_calls(
    config: Optional[Dict[str, Any]],
    row_type: Optional[str],
    name_to_uuid: Dict[str, str],
    uuid_to_name: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """On write — identical for `POST`/`PUT /tests` and `POST /tests/bulk`:
      1. Derive `evaluation.type` from the row `type` (single source of truth).
      2. For tool_call tests only: validate any caller-supplied `tool_uuid` (404 on a
         bad/foreign id, stamp the live name) and auto-link name-only entries to a
         tool by org name-match.
    Dispatch is driven by the row `type`, never the (drift-prone) blob copy. Built-in
    / unmatched names stay name-only. Mutates and returns `config`.
    """
    _normalize_evaluation_type(config, row_type)
    if row_type == "tool_call":
        _resolve_tool_call_uuids(config, uuid_to_name)
        inject_tool_uuids_into_config(config, name_to_uuid)
    return config


def _with_evaluators(
    test_dict: Dict[str, Any], uuid_to_name: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Attach linked evaluators to a test dict, derive `evaluation.type` from the row
    `type`, and (for tool_call tests) refresh each expected tool call's `tool` name
    from its `tool_uuid` so a renamed tool is reflected without rewriting the test.
    Pass the org's `uuid_to_name` index to enable the name refresh."""
    row_type = test_dict.get("type")
    _normalize_evaluation_type(test_dict.get("config"), row_type)
    if uuid_to_name and row_type == "tool_call":
        refresh_tool_call_names_in_config(test_dict.get("config"), uuid_to_name)
    evaluators = get_evaluators_for_test(test_dict["uuid"])
    return {**test_dict, "evaluators": evaluators}


@router.post("/bulk-delete", response_model=BulkTestDeleteResponse)
async def bulk_delete_tests_endpoint(
    payload: BulkTestDelete, ctx: OrgContext = Depends(get_current_org)
):
    """Bulk delete tests by UUIDs. Only deletes tests in the caller's current org."""
    if not payload.test_uuids:
        raise HTTPException(status_code=400, detail="test_uuids must not be empty")

    deleted_count = bulk_delete_tests(test_uuids=payload.test_uuids, org_uuid=ctx.org_uuid)

    return BulkTestDeleteResponse(
        deleted_count=deleted_count,
        message=f"Successfully deleted {deleted_count} test(s)",
    )


@router.post("/bulk", response_model=BulkTestUploadResponse)
async def bulk_upload_tests(
    payload: BulkTestUpload, ctx: OrgContext = Depends(get_current_org)
):
    """Bulk upload LLM tests. All tests must be the same type (response or tool_call)."""
    if payload.agent_uuids:
        for agent_uuid in payload.agent_uuids:
            agent = get_agent(agent_uuid)
            if not agent or agent.get("org_uuid") != ctx.org_uuid:
                raise HTTPException(status_code=404, detail=f"Agent {agent_uuid} not found")

    resolved_evaluator_refs: List[Optional[List[Dict[str, Any]]]] = []
    for t in payload.tests:
        if t.evaluators:
            resolved_evaluator_refs.append(
                _validate_evaluators(t.evaluators, ctx.org_uuid, payload.type)
            )
        else:
            resolved_evaluator_refs.append(None)

    name_to_uuid, uuid_to_name = _org_tool_indexes(ctx.org_uuid)

    db_tests = []
    for t in payload.tests:
        evaluation: Dict[str, Any] = {"type": payload.type}
        if payload.type == "tool_call":
            evaluation["tool_calls"] = [tc.model_dump() for tc in t.tool_calls]

        config: Dict[str, Any] = {
            "history": [msg.model_dump(exclude_none=True) for msg in t.conversation_history],
            "evaluation": evaluation,
        }
        if payload.language:
            config["settings"] = {"language": payload.language}

        _link_tool_calls(config, payload.type, name_to_uuid, uuid_to_name)

        db_tests.append({
            "name": t.name,
            "type": payload.type,
            "config": config,
        })

    try:
        uuids = bulk_create_tests(
            tests=db_tests, org_uuid=ctx.org_uuid, user_id=ctx.user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    for test_uuid, refs in zip(uuids, resolved_evaluator_refs):
        if refs:
            set_test_evaluators(test_uuid, refs)

    warnings: List[str] = []
    if payload.agent_uuids:
        linked_agents = set()
        for agent_uuid in payload.agent_uuids:
            agent_failed = False
            for test_uuid in uuids:
                try:
                    add_test_to_agent(agent_uuid, test_uuid)
                    linked_agents.add(agent_uuid)
                except Exception as e:
                    agent_failed = True
                    logger.warning(f"Failed to link test {test_uuid} to agent {agent_uuid}: {e}")
            if agent_failed:
                warnings.append(f"Some tests could not be linked to agent {agent_uuid}")

    message = f"Successfully created {len(uuids)} tests"
    if payload.agent_uuids:
        message += f" and linked to {len(linked_agents)} agent(s)"

    return BulkTestUploadResponse(
        uuids=uuids,
        count=len(uuids),
        message=message,
        warnings=warnings or None,
    )


@router.post("", response_model=TestCreateResponse)
async def create_test_endpoint(
    test: TestCreate, ctx: OrgContext = Depends(get_current_org)
):
    """Create a new test."""
    # Conversation tests have no evaluator fallback (unlike `response`, which can
    # synthesize the default LLM judge from legacy string criteria) — without a
    # linked simulation evaluator a run produces an empty calibrate config with
    # nothing to judge. Require at least one up front. (The bulk endpoint already
    # enforces this; this closes the single-create gap.)
    if test.type == "conversation" and not test.evaluators:
        raise HTTPException(
            status_code=400,
            detail="Conversation tests require at least one evaluator.",
        )
    resolved = (
        _validate_evaluators(test.evaluators, ctx.org_uuid, test.type)
        if test.evaluators
        else None
    )
    config = test.config
    if config is not None:
        name_to_uuid, uuid_to_name = _org_tool_indexes(ctx.org_uuid)
        _link_tool_calls(config, test.type, name_to_uuid, uuid_to_name)
    with ensure_name_unique("tests", test.name, ctx.org_uuid, entity="Test"):
        test_uuid = create_test(
            name=test.name,
            type=test.type,
            config=config,
            org_uuid=ctx.org_uuid,
            user_id=ctx.user_id,
        )
    if resolved:
        set_test_evaluators(test_uuid, resolved)
    return TestCreateResponse(uuid=test_uuid, message="Test created successfully")


@router.get("", response_model=List[TestResponse])
async def list_tests(ctx: OrgContext = Depends(get_current_org)):
    """List all tests for the caller's current org."""
    tests = get_all_tests(org_uuid=ctx.org_uuid)
    _, uuid_to_name = _org_tool_indexes(ctx.org_uuid)
    return [_with_evaluators(t, uuid_to_name) for t in tests]


@router.get("/{test_uuid}", response_model=TestResponse)
async def get_test_endpoint(
    test_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    """Get a test by UUID."""
    test = get_test(test_uuid)
    if not test or test.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Test not found")
    _, uuid_to_name = _org_tool_indexes(ctx.org_uuid)
    return _with_evaluators(test, uuid_to_name)


@router.put("/{test_uuid}", response_model=TestResponse)
async def update_test_endpoint(
    test_uuid: str, test: TestUpdate, ctx: OrgContext = Depends(get_current_org)
):
    """Update a test."""
    existing_test = get_test(test_uuid)
    if not existing_test or existing_test.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Test not found")

    # A test's `type` is immutable after creation. Allowing a change would
    # strand already-linked evaluators whose `evaluator_type` was validated
    # against the original type (e.g. a `response` test's `llm` evaluator
    # surviving a switch to `conversation`, which only accepts `simulation`).
    # Echoing back the same value is a no-op; a different value is rejected.
    existing_type = existing_test.get("type")
    if test.type is not None and test.type != existing_type:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Test type is immutable; cannot change from "
                f"'{existing_type}' to '{test.type}'. Create a new test instead."
            ),
        )

    # Conversation tests must keep at least one evaluator (see create-endpoint
    # note). Reject an update that would clear them all.
    if (
        existing_type == "conversation"
        and test.evaluators is not None
        and len(test.evaluators) == 0
    ):
        raise HTTPException(
            status_code=400,
            detail="Conversation tests require at least one evaluator; cannot remove all.",
        )

    resolved = (
        _validate_evaluators(test.evaluators, ctx.org_uuid, existing_type)
        if test.evaluators is not None
        else None
    )

    name_to_uuid, uuid_to_name = _org_tool_indexes(ctx.org_uuid)
    config_to_save = test.config
    if config_to_save is not None:
        _link_tool_calls(config_to_save, existing_type, name_to_uuid, uuid_to_name)

    has_core_updates = any(
        v is not None for v in (test.name, test.type, test.config)
    )
    if has_core_updates:
        with ensure_name_unique(
            "tests", test.name, ctx.org_uuid, entity="Test", exclude_uuid=test_uuid
        ):
            updated = update_test(
                test_uuid=test_uuid,
                name=test.name,
                type=test.type,
                config=config_to_save,
            )
        if not updated and resolved is None:
            raise HTTPException(status_code=400, detail="No fields to update")

    if resolved is not None:
        set_test_evaluators(test_uuid, resolved)

    return _with_evaluators(get_test(test_uuid), uuid_to_name)


@router.delete("/{test_uuid}")
async def delete_test_endpoint(
    test_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    """Delete a test."""
    existing_test = get_test(test_uuid)
    if not existing_test or existing_test.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Test not found")

    deleted = delete_test(test_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Test not found")
    return {"message": "Test deleted successfully"}
