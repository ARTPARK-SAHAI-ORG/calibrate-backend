"""Closed test, agent, and evaluator type sets, and the maps between them.

Lives outside `routers/` so `db.py` and other modules can import them without a db→router cycle.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

TestType = Literal["response", "tool_call", "conversation", "general"]
EvaluatorType = Literal["tts", "stt", "llm", "llm-general", "conversation", "tool-call"]
# An agent's interaction type must match calibrate_agent.connections.AGENT_TYPES,
# as it is validated by the CLI before the run starts.
AgentInteractionType = Literal["conversation", "general"]
AGENT_INTERACTION_TYPES: tuple[AgentInteractionType, ...] = get_args(AgentInteractionType)
DEFAULT_AGENT_INTERACTION_TYPE: AgentInteractionType = "conversation"

# Each test type pins the evaluator_type it accepts.
# `response`/`tool_call` tests judge a single LLM reply, so only `llm` evaluators apply;
# `conversation` tests judge whole simulated conversations, so only `conversation` evaluators apply;
# `general` tests judge a standalone, non-conversational input/output pair, so only `llm-general` evaluators apply.
REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE: dict[TestType, EvaluatorType] = {
    "response": "llm",
    "tool_call": "llm",
    "conversation": "conversation",
    "general": "llm-general",
}

# Each test type requires a matching agent `interaction_type`:
# `general` tests have no conversation history to feed a conversational agent,
# and conversation-style tests (response/tool_call/conversation) have nothing to feed a `general` agent.
# Single source of truth for the gate enforced in `POST /agent-tests` and `POST /tests/bulk`.
REQUIRED_AGENT_INTERACTION_TYPE_BY_TEST_TYPE: dict[TestType, AgentInteractionType] = {
    "response": "conversation",
    "tool_call": "conversation",
    "conversation": "conversation",
    "general": "general",
}


def required_agent_interaction_type(
    test_type: str | None, config: dict[str, Any] | None
) -> AgentInteractionType:
    """Return the agent `interaction_type` a test can be linked to.

    `tool_call` is the one type that spans both: it asserts on the tool calls the
    agent generated, which a one-shot agent produces just as well as a
    conversational one. Its config shape decides which: `input` (a standalone
    prompt, the same field a `general` test carries) means a `general` agent,
    `history` means a conversational one. A config carrying both is read as
    conversational, matching which of the two `_build_calibrate_config` sends.
    """
    if (
        test_type == "tool_call"
        and isinstance(config, dict)
        and config.get("history") is None
        and config.get("input") is not None
    ):
        return "general"
    if test_type in REQUIRED_AGENT_INTERACTION_TYPE_BY_TEST_TYPE:
        return REQUIRED_AGENT_INTERACTION_TYPE_BY_TEST_TYPE[test_type]
    return DEFAULT_AGENT_INTERACTION_TYPE
