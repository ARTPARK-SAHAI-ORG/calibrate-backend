"""Trace-scoring eligibility and plan resolution.

Shared by agent opt-in and ingest-time run creation. Lives outside `routers/`
so `db.py` can import it without a db→router cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from shared_enums import (
    AgentInteractionType,
    EvaluatorType,
    REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE,
)

# Subset of TestType that traces can score.
EvaluationType = Literal["response", "general"]
TRACE_SCORING_MODE_BY_INTERACTION_TYPE: dict[
    AgentInteractionType,
    tuple[EvaluationType, EvaluatorType],
] = {
    "conversation": ("response", REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE["response"]),
    "general": ("general", REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE["general"]),
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
    evaluator_uuid: str
    name: str
    reason: IneligibleReason


@dataclass(frozen=True)
class TraceScoringResolution:
    evaluation_type: EvaluationType | None
    evaluator_type: EvaluatorType | None
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
    """Partition linked evaluators for this interaction type.

    `live_evaluators` is `(evaluators row, live evaluator_versions row or None)`
    from `resolve_live_evaluators`. Type is checked before live-version /
    variable checks so a mixed set never reaches `_validate_evaluators`.
    Never raises.
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
