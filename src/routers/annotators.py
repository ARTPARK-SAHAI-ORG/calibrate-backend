from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Path, Query
from pydantic import BaseModel, Field

from db import (
    create_annotator,
    get_annotator,
    get_all_annotators,
    update_annotator,
    delete_annotator,
    ensure_name_unique,
    get_jobs_for_annotator_detailed,
    get_job_counts_for_org_annotators,
    get_annotations_for_org,
    get_annotations_for_annotator_overlap_slots,
)
from auth_utils import get_current_org, OrgContext
from annotation_metrics import (
    aggregate_agreement_for_annotator,
    trend_series_for_annotator,
)


router = APIRouter(prefix="/annotators", tags=["annotators"])


class AnnotatorCreate(BaseModel):
    name: str = Field(description="Human-readable annotator name, unique within the workspace")


class AnnotatorUpdate(BaseModel):
    name: Optional[str] = Field(
        None, description="New annotator name (unique within the workspace). Omit to leave unchanged"
    )


class AnnotatorResponse(BaseModel):
    uuid: str = Field(description="Annotator UUID (8-char identifier)")
    name: str = Field(description="Human-readable annotator name")
    created_at: str = Field(description="Creation timestamp (ISO 8601 UTC)")
    updated_at: str = Field(description="Last-update timestamp (ISO 8601 UTC)")
    jobs_count: Optional[int] = Field(
        None, description="Number of labelling jobs assigned to this annotator. `null` when not computed"
    )
    current_agreement: Optional[float] = Field(
        None,
        description="Latest pairwise mean agreement `[0, 1]` vs other annotators. `null` when there's no overlap to compute",
    )
    pair_count: Optional[int] = Field(
        None, description="Number of comparable annotation pairs behind `current_agreement`. `null` when none exist"
    )


class AnnotatorCreateResponse(BaseModel):
    uuid: str = Field(description="UUID of the newly created annotator (8-char identifier)")
    message: str = Field(description="Human-readable success message")


def _ensure_owned_annotator(annotator_uuid: str, org_uuid: str):
    annotator = get_annotator(annotator_uuid)
    if not annotator or annotator.get("org_uuid") != org_uuid:
        raise HTTPException(status_code=404, detail="Annotator not found")
    return annotator


@router.post("", response_model=AnnotatorCreateResponse, summary="Create annotator")
async def create_annotator_endpoint(
    payload: AnnotatorCreate,
    ctx: OrgContext = Depends(get_current_org),
):
    """Create a new annotator in your workspace. Name must be unique per workspace."""
    try:
        with ensure_name_unique(
            "annotators", payload.name, ctx.org_uuid, entity="Annotator"
        ):
            annotator_uuid = create_annotator(
                name=payload.name, org_uuid=ctx.org_uuid, user_id=ctx.user_id
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AnnotatorCreateResponse(
        uuid=annotator_uuid, message="Annotator created successfully"
    )


@router.get("", response_model=List[AnnotatorResponse], summary="List annotators")
async def list_annotators(ctx: OrgContext = Depends(get_current_org)):
    """List all annotators in your workspace with per-annotator stats:
    `jobs_count` and `current_agreement` (pairwise mean vs other annotators).
    Both are `null` when there's nothing to compute (no jobs / no overlap).
    """
    annotators = get_all_annotators(org_uuid=ctx.org_uuid)
    if not annotators:
        return []
    jobs_count_by_annotator = get_job_counts_for_org_annotators(ctx.org_uuid)
    all_annotations = get_annotations_for_org(ctx.org_uuid)
    out: List[Dict[str, Any]] = []
    for a in annotators:
        agreement, pairs = aggregate_agreement_for_annotator(
            all_annotations, a["uuid"]
        )
        out.append(
            {
                **a,
                "jobs_count": jobs_count_by_annotator.get(a["uuid"], 0),
                "current_agreement": agreement,
                "pair_count": pairs if pairs else None,
            }
        )
    return out


@router.get("/{annotator_uuid}", summary="Get annotator")
async def get_annotator_endpoint(
    annotator_uuid: str = Path(description="Annotator UUID (8-char identifier)"),
    bucket: str = Query(
        "month",
        pattern="^(week|month|year)$",
        description="Time bucket for the agreement trend series (`week`, `month`, or `year`)",
    ),
    days: int = Query(
        365, ge=1, le=3650, description="Trailing window in days for the trend series"
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get one annotator's detail: basic info, jobs assigned to it (with
    task name + item/annotation counts), latest agreement vs other annotators,
    and the agreement trend series."""
    annotator = _ensure_owned_annotator(annotator_uuid, ctx.org_uuid)

    jobs = get_jobs_for_annotator_detailed(annotator_uuid)

    annotations = get_annotations_for_annotator_overlap_slots(
        org_uuid=ctx.org_uuid, annotator_id=annotator_uuid
    )
    current, pair_count = aggregate_agreement_for_annotator(
        annotations, annotator_uuid
    )
    series = trend_series_for_annotator(
        annotations, annotator_uuid, bucket=bucket, days=days
    )

    return {
        "annotator": {
            "uuid": annotator["uuid"],
            "name": annotator["name"],
            "created_at": annotator["created_at"],
            "updated_at": annotator["updated_at"],
        },
        "stats": {
            "current_agreement": current,
            "pair_count": pair_count,
            "jobs_count": len(jobs),
        },
        "trend": {
            "bucket": bucket,
            "days": days,
            "series": series,
        },
        "jobs": jobs,
    }


@router.put("/{annotator_uuid}", response_model=AnnotatorResponse, summary="Update annotator")
async def update_annotator_endpoint(
    annotator_uuid: str = Path(description="Annotator UUID (8-char identifier)"),
    payload: AnnotatorUpdate = ...,
    ctx: OrgContext = Depends(get_current_org),
):
    """Update an annotator's name. Returns 400 when nothing is provided to change."""
    _ensure_owned_annotator(annotator_uuid, ctx.org_uuid)
    try:
        with ensure_name_unique(
            "annotators",
            payload.name,
            ctx.org_uuid,
            entity="Annotator",
            exclude_uuid=annotator_uuid,
        ):
            updated = update_annotator(annotator_uuid=annotator_uuid, name=payload.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")
    return get_annotator(annotator_uuid)


@router.delete("/{annotator_uuid}", summary="Delete annotator")
async def delete_annotator_endpoint(
    annotator_uuid: str = Path(description="Annotator UUID (8-char identifier)"),
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete an annotator by UUID."""
    _ensure_owned_annotator(annotator_uuid, ctx.org_uuid)
    deleted = delete_annotator(annotator_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Annotator not found")
    return {"message": "Annotator deleted successfully"}
