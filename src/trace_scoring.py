"""Trace-scoring eligibility and plan resolution.

Shared by the agent opt-in API and ingest-time run creation. Lives outside
`routers/` so `db.py` can call it without importing a router. Callers load
`(evaluator, live_version)` pairs via `db.resolve_live_evaluators`; this
module does not import `db`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from utils import AgentInteractionType

# interaction_type → (evaluation.type, required evaluator_type). Kept here
# (not imported from routers.tests) so resolution never creates a db→router
# cycle. Must stay aligned with REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE for
# `response`/`general`.
EvaluationType = Literal["response", "general"]
RequiredEvaluatorType = Literal["llm", "llm-general"]
TRACE_SCORING_MODE_BY_INTERACTION_TYPE: dict[
    AgentInteractionType,
    tuple[EvaluationType, RequiredEvaluatorType]
] = {
    "conversation": ("response", "llm"),
    "general": ("general", "llm-general"),
}


class IneligibleReason(str, Enum):
    """Why a linked evaluator cannot score this agent's traces."""

    WRONG_TYPE = "wrong_type_for_agent"
    NO_LIVE_VERSION = "no_live_version"
    DECLARES_VARIABLES = "declares_variables"


@dataclass(frozen=True)
class ScoringPlanPin:
    """One evaluator pin stored within a `trace_eval_runs.scoring_plan`."""

    evaluator_uuid: str
    evaluator_version_id: str


@dataclass(frozen=True)
class ScoringPlan:
    """JSON envelope pinning evaluators to use in a runnable `trace_eval_runs` row."""

    evaluation_type: EvaluationType
    evaluators: list[ScoringPlanPin]


@dataclass(frozen=True)
class ScoringPlanSkip:
    """Why ingest wrote a `skipped` run instead of a runnable plan."""

    skip: Literal["unsupported_interaction_type", "no_usable_evaluators"]


@dataclass(frozen=True)
class TraceScoringEligible:
    """Eligible snapshot pin plus the evaluator name for eligibility responses."""

    pin: ScoringPlanPin
    name: str


@dataclass(frozen=True)
class TraceScoringIneligible:
    """One evaluator ineligible to score this agent's traces."""

    evaluator_uuid: str
    name: str
    reason: IneligibleReason


@dataclass(frozen=True)
class TraceScoringResolution:
    """The result of resolving a linked evaluator set for a particular agent."""

    # Scoring subset of TestType. From TRACE_SCORING_MODE_BY_INTERACTION_TYPE
    # (None if interaction_type is unsupported).
    evaluation_type: EvaluationType | None
    # Required EvaluatorTypeLiteral for that mode, same map as
    # REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE. Not the full VALID_EVALUATOR_TYPES
    # column on evaluators.
    evaluator_type: RequiredEvaluatorType | None
    eligible: list[TraceScoringEligible] = field(default_factory=list)
    ineligible: list[TraceScoringIneligible] = field(default_factory=list)

    def as_plan(self) -> ScoringPlan | ScoringPlanSkip:
        """Snapshot written at ingest, or a skip reason if nothing can score."""
        if self.evaluation_type is None:
            return ScoringPlanSkip(skip="unsupported_interaction_type")
        if not self.eligible:
            return ScoringPlanSkip(skip="no_usable_evaluators")
        return ScoringPlan(
            evaluation_type=self.evaluation_type,
            evaluators=[item.pin for item in self.eligible],
        )


def resolve_trace_scoring(
    interaction_type: AgentInteractionType | None,
    live_evaluators: list[tuple[dict[str, Any], dict[str, Any] | None]],
) -> TraceScoringResolution:
    """Split linked evaluators into eligible pins and ineligible-with-reason.

    Args:
        interaction_type: `agents.interaction_type` (supports: `conversation` | `general`)
        live_evaluators: each pair is an `evaluators` row (`_parse_evaluator_row`)
            and its live `evaluator_versions` row (`_parse_evaluator_version_row`),
            or `None` when `live_version_id` is unset or that version row is gone.
            See also `db.resolve_live_evaluators`.
        
    Type is checked before live-version / variable checks so a mixed linked
    set is never handed to `_validate_evaluators`, which raises on the first
    mismatch. Never raises.
    """
    mode = TRACE_SCORING_MODE_BY_INTERACTION_TYPE.get(interaction_type) if interaction_type is not None else None
    if mode is None:
        return TraceScoringResolution(
            evaluation_type=None,
            evaluator_type=None,
            eligible=[],
            ineligible=[
                TraceScoringIneligible(
                    evaluator_uuid=ev["uuid"],
                    name=ev.get("name") or ev["uuid"],
                    reason=IneligibleReason.WRONG_TYPE,
                )
                for ev, _ in live_evaluators
            ],
        )

    evaluation_type, required_evaluator_type = mode
    eligible: list[TraceScoringEligible] = []
    ineligible: list[TraceScoringIneligible] = []
    for ev, version in live_evaluators:
        name = ev.get("name") or ev["uuid"]
        if ev.get("evaluator_type") != required_evaluator_type:
            ineligible.append(
                TraceScoringIneligible(
                    evaluator_uuid=ev["uuid"],
                    name=name,
                    reason=IneligibleReason.WRONG_TYPE,
                )
            )
            continue
        if not version:
            ineligible.append(
                TraceScoringIneligible(
                    evaluator_uuid=ev["uuid"],
                    name=name,
                    reason=IneligibleReason.NO_LIVE_VERSION,
                )
            )
            continue
        if version.get("variables"):
            ineligible.append(
                TraceScoringIneligible(
                    evaluator_uuid=ev["uuid"],
                    name=name,
                    reason=IneligibleReason.DECLARES_VARIABLES,
                )
            )
            continue
        eligible.append(
            TraceScoringEligible(
                pin=ScoringPlanPin(
                    evaluator_uuid=ev["uuid"],
                    evaluator_version_id=version["uuid"],
                ),
                name=name,
            )
        )
    return TraceScoringResolution(
        evaluation_type=evaluation_type,
        evaluator_type=required_evaluator_type,
        eligible=eligible,
        ineligible=ineligible,
    )
