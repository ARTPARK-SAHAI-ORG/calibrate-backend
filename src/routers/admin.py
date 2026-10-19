"""Usage numbers across every workspace. Superadmin only."""

from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

import admin_analytics
from auth_utils import get_current_user_id, is_superadmin_user, require_superadmin

router = APIRouter(prefix="/admin", tags=["admin"])


class AccessResponse(BaseModel):
    is_superadmin: bool = Field(description="Whether you can open the superadmin pages")


class Totals(BaseModel):
    users: int = Field(description="Users who have signed up, all time")
    workspaces: int = Field(description="Workspaces in use, all time")
    evaluations_running: int = Field(description="Evaluations queued or in progress right now")


class Activity(BaseModel):
    new_users: int = Field(description="Users who signed up in the period")
    active_workspaces: int = Field(
        description="Workspaces that started an evaluation, received a trace, or collected a label in the period"
    )
    evaluations: int = Field(description="Evaluations started in the period, of every kind")
    failed_evaluations: int = Field(description="Evaluations started in the period that failed")
    test_runs: int = Field(description="Test runs started in the period")
    model_comparisons: int = Field(description="Model comparisons started in the period")
    simulations: int = Field(description="Simulation runs started in the period")
    stt: int = Field(description="Speech to text evaluations started in the period")
    tts: int = Field(description="Text to speech evaluations started in the period")
    labelling_evaluator_runs: int = Field(
        description="Evaluator runs over labelling items started in the period"
    )
    cost: Optional[float] = Field(
        description="Cost in USD of the test runs and model comparisons started in the period"
    )
    traces_received: int = Field(description="Traces received in the period")
    traces_scored: int = Field(description="Traces scored in the period")
    labelling_jobs_finished: int = Field(description="Labelling jobs finished in the period")
    labels_collected: int = Field(description="Labels collected in the period")
    items_labelled: int = Field(description="Distinct items that got at least one label in the period")
    limit_hits: int = Field(description="Times a workspace was stopped by a limit in the period")


class DayCounts(BaseModel):
    date: str = Field(description="The day, in the requested time offset (YYYY-MM-DD)")
    new_users: int = Field(description="Users who signed up that day")
    active_workspaces: int = Field(description="Workspaces active that day")
    test_runs: int = Field(description="Test runs started that day")
    model_comparisons: int = Field(description="Model comparisons started that day")
    simulations: int = Field(description="Simulation runs started that day")
    stt: int = Field(description="Speech to text evaluations started that day")
    tts: int = Field(description="Text to speech evaluations started that day")
    labelling_evaluator_runs: int = Field(description="Evaluator runs over labelling items started that day")
    failed_evaluations: int = Field(description="Evaluations started that day that failed")
    traces_received: int = Field(description="Traces received that day")
    traces_scored: int = Field(description="Traces scored that day")
    labels_collected: int = Field(description="Labels collected that day")
    labelling_jobs_finished: int = Field(description="Labelling jobs finished that day")


class WorkspaceRow(BaseModel):
    org_uuid: str = Field(description="Workspace ID")
    name: str = Field(description="Workspace name")
    is_personal: bool = Field(description="Whether this is someone's personal workspace")
    members: int = Field(description="Members of the workspace now")
    last_active_at: Optional[str] = Field(
        description="When the workspace last started an evaluation, received a trace, or collected a label, all time (ISO 8601 UTC)"
    )
    evaluations: int = Field(description="Evaluations started in the period")
    failed_evaluations: int = Field(description="Evaluations started in the period that failed")
    test_runs: int = Field(description="Test runs started in the period")
    model_comparisons: int = Field(description="Model comparisons started in the period")
    simulations: int = Field(description="Simulation runs started in the period")
    stt: int = Field(description="Speech to text evaluations started in the period")
    tts: int = Field(description="Text to speech evaluations started in the period")
    labelling_evaluator_runs: int = Field(description="Evaluator runs over labelling items started in the period")
    cost: Optional[float] = Field(
        description="Cost in USD of the test runs and model comparisons started in the period"
    )
    traces_received: int = Field(description="Traces received in the period")
    traces_scored: int = Field(description="Traces scored in the period")
    labelling_jobs_finished: int = Field(description="Labelling jobs finished in the period")
    labels_collected: int = Field(description="Labels collected in the period")
    items_labelled: int = Field(description="Distinct items that got at least one label in the period")
    limit_hits: int = Field(description="Times the workspace was stopped by a limit in the period")


class Failure(BaseModel):
    id: str = Field(description="ID of the run that failed")
    created_at: str = Field(description="When the run was created (ISO 8601 UTC)")
    org_uuid: str = Field(description="Workspace ID")
    org_name: str = Field(description="Workspace name")
    kind: str = Field(
        description="What failed: `test_run`, `model_comparison`, `simulation`, `stt`, `tts` or `labelling_evaluator_run`"
    )
    error: Optional[str] = Field(description="The error recorded for the run")


class LimitHit(BaseModel):
    created_at: str = Field(description="When the limit was hit (ISO 8601 UTC)")
    org_uuid: str = Field(description="Workspace ID")
    org_name: str = Field(description="Workspace name")
    limit_name: str = Field(
        description="Which limit: `max_rows_per_eval`, `max_traces` or `max_scored_traces`"
    )
    kind: Optional[str] = Field(description="What was being started when the limit was hit")
    requested: Optional[int] = Field(description="How many rows or traces were asked for")
    cap: Optional[int] = Field(description="The limit at the time")


class AnalyticsResponse(BaseModel):
    totals: Totals = Field(description="All-time totals")
    activity: Activity = Field(description="What happened in the period")
    previous: Optional[Activity] = Field(
        description="The same numbers for the period of the same length just before"
    )
    per_day: List[DayCounts] = Field(
        description="Counts for every day in the period, including days with nothing. With no start date, from the first day with any activity"
    )
    workspaces: List[WorkspaceRow] = Field(description="Every workspace, most recently active first")
    failures: List[Failure] = Field(description=f"The latest {admin_analytics.LIST_LIMIT} failed runs in the period")
    failures_total: int = Field(description="All failed runs in the period")
    limit_hits: List[LimitHit] = Field(description=f"The latest {admin_analytics.LIST_LIMIT} limit hits in the period")
    limit_hits_total: int = Field(description="All limit hits in the period")


@router.get("/access", response_model=AccessResponse, summary="Check superadmin access")
def get_access(user_id: str = Depends(get_current_user_id)):
    """Say whether you can open the superadmin pages"""
    return {"is_superadmin": is_superadmin_user(user_id)}


@router.get("/analytics", response_model=AnalyticsResponse, summary="Get usage analytics")
def get_analytics(
    start: Optional[date] = Query(
        None, description="First day of the period (YYYY-MM-DD). Omit for all time"
    ),
    end: Optional[date] = Query(
        None, description="Last day of the period, included (YYYY-MM-DD). Defaults to today"
    ),
    tz_offset_minutes: int = Query(
        0,
        ge=-14 * 60,
        le=14 * 60,
        description="Offset from UTC used to decide which day something happened on, for example 330 for India",
    ),
    user_id: str = Depends(require_superadmin),
):
    """Get usage across every workspace for a period. Superadmin only"""
    end = end or admin_analytics.today_in_offset(tz_offset_minutes)
    if start and end < start:
        raise HTTPException(status_code=422, detail="end must not be before start")
    return admin_analytics.get_analytics(start, end, tz_offset_minutes)
