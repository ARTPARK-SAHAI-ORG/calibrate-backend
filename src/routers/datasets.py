import logging
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from auth_utils import get_current_org, OrgContext
from utils import presign_audio_path
from db import (
    create_dataset,
    get_dataset,
    get_all_datasets,
    get_dataset_item_counts,
    get_dataset_eval_counts,
    update_dataset_name,
    delete_dataset,
    add_dataset_items,
    get_dataset_item,
    get_dataset_items,
    get_dataset_items_by_uuids,
    update_dataset_item,
    delete_dataset_item,
    delete_dataset_items,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


# ── Request / Response models ────────────────────────────────────────────────


class DatasetCreateRequest(BaseModel):
    name: str
    dataset_type: Literal["stt", "tts"]


class DatasetRenameRequest(BaseModel):
    name: str


class DatasetItemIn(BaseModel):
    audio_path: Optional[str] = None  # required for STT datasets
    text: str


class DatasetItemUpdate(BaseModel):
    audio_path: Optional[str] = None
    text: Optional[str] = None


class DatasetItemResponse(BaseModel):
    uuid: str
    audio_path: Optional[str]
    text: str
    order_index: int
    created_at: str
    updated_at: Optional[str] = None


class DatasetResponse(BaseModel):
    uuid: str
    name: str
    dataset_type: str
    item_count: int
    eval_count: int
    created_at: str
    updated_at: str


class DatasetDetailResponse(DatasetResponse):
    items: List[DatasetItemResponse]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _validate_items_for_type(dataset_type: str, items: List[DatasetItemIn]) -> None:
    """Raise HTTPException if items are inconsistent with the dataset type."""
    for item in items:
        if dataset_type == "stt" and not item.audio_path:
            raise HTTPException(
                status_code=400,
                detail="STT dataset items must include audio_path",
            )
        if dataset_type == "tts" and item.audio_path:
            raise HTTPException(
                status_code=400,
                detail="TTS dataset items must not include audio_path",
            )


def _item_row_to_response(row: dict) -> DatasetItemResponse:
    return DatasetItemResponse(
        uuid=row["uuid"],
        audio_path=presign_audio_path(row.get("audio_path")),
        text=row["text"],
        order_index=row["order_index"],
        created_at=row["created_at"],
        updated_at=row.get("updated_at"),
    )


def _dataset_row_to_response(
    row: dict, item_count: int, eval_count: int = 0
) -> DatasetResponse:
    return DatasetResponse(
        uuid=row["uuid"],
        name=row["name"],
        dataset_type=row["type"],
        item_count=item_count,
        eval_count=eval_count,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("", response_model=DatasetResponse, status_code=201)
async def create_new_dataset(
    request: DatasetCreateRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Create a new empty dataset."""
    dataset_uuid = create_dataset(
        name=request.name,
        dataset_type=request.dataset_type,
        org_uuid=ctx.org_uuid,
        user_id=ctx.user_id,
    )
    row = get_dataset(dataset_uuid, org_uuid=ctx.org_uuid)
    return _dataset_row_to_response(row, item_count=0, eval_count=0)


@router.get("", response_model=List[DatasetResponse])
async def list_datasets(
    dataset_type: Optional[str] = None,
    ctx: OrgContext = Depends(get_current_org),
):
    """List all datasets for the caller's current org, optionally filtered by type."""
    if dataset_type and dataset_type not in ("stt", "tts"):
        raise HTTPException(
            status_code=400, detail="dataset_type must be 'stt' or 'tts'"
        )

    rows = get_all_datasets(org_uuid=ctx.org_uuid, dataset_type=dataset_type)
    uuids = [row["uuid"] for row in rows]
    counts = get_dataset_item_counts(uuids)
    eval_counts = get_dataset_eval_counts(uuids)
    return [
        _dataset_row_to_response(
            row,
            item_count=counts.get(row["uuid"], 0),
            eval_count=eval_counts.get(row["uuid"], 0),
        )
        for row in rows
    ]


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset_detail(
    dataset_id: str,
    ctx: OrgContext = Depends(get_current_org),
):
    """Get a dataset with all its items."""
    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    items = get_dataset_items(dataset_id)
    eval_counts = get_dataset_eval_counts([dataset_id])
    return DatasetDetailResponse(
        uuid=row["uuid"],
        name=row["name"],
        dataset_type=row["type"],
        item_count=len(items),
        eval_count=eval_counts.get(dataset_id, 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        items=[_item_row_to_response(i) for i in items],
    )


@router.patch("/{dataset_id}", response_model=DatasetResponse)
async def rename_dataset(
    dataset_id: str,
    request: DatasetRenameRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Rename a dataset."""
    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    update_dataset_name(dataset_id, org_uuid=ctx.org_uuid, name=request.name)
    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    counts = get_dataset_item_counts([dataset_id])
    eval_counts = get_dataset_eval_counts([dataset_id])
    return _dataset_row_to_response(
        row,
        item_count=counts.get(dataset_id, 0),
        eval_count=eval_counts.get(dataset_id, 0),
    )


@router.delete("/{dataset_id}", status_code=204)
async def remove_dataset(
    dataset_id: str,
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft delete a dataset and all its items."""
    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    delete_dataset(dataset_id, org_uuid=ctx.org_uuid)


@router.post(
    "/{dataset_id}/items", response_model=List[DatasetItemResponse], status_code=201
)
async def add_items(
    dataset_id: str,
    items: List[DatasetItemIn],
    ctx: OrgContext = Depends(get_current_org),
):
    """Add one or more items to a dataset."""
    if not items:
        raise HTTPException(status_code=400, detail="items list cannot be empty")
    if len(items) > 1000:
        raise HTTPException(
            status_code=400, detail="Cannot add more than 1000 items per request"
        )

    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    _validate_items_for_type(row["type"], items)

    item_dicts = [{"audio_path": i.audio_path, "text": i.text} for i in items]
    new_uuids = add_dataset_items(dataset_id, item_dicts)

    return [_item_row_to_response(i) for i in get_dataset_items_by_uuids(new_uuids)]


@router.patch("/{dataset_id}/items/{item_uuid}", response_model=DatasetItemResponse)
async def update_item(
    dataset_id: str,
    item_uuid: str,
    request: DatasetItemUpdate,
    ctx: OrgContext = Depends(get_current_org),
):
    """Update a dataset item's text or audio_path."""
    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not {"text", "audio_path"} & request.model_fields_set:
        raise HTTPException(status_code=400, detail="Nothing to update")

    if "audio_path" in request.model_fields_set:
        if row["type"] == "stt" and not request.audio_path:
            raise HTTPException(
                status_code=400,
                detail="STT dataset items must include audio_path",
            )
        if row["type"] == "tts" and request.audio_path:
            raise HTTPException(
                status_code=400,
                detail="TTS dataset items must not include audio_path",
            )

    updated = update_dataset_item(
        item_uuid,
        dataset_id,
        text=request.text if "text" in request.model_fields_set else None,
        audio_path=(
            request.audio_path if "audio_path" in request.model_fields_set else ...
        ),
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Item not found")

    item = get_dataset_item(item_uuid, dataset_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return _item_row_to_response(item)


class BulkDeleteItemsRequest(BaseModel):
    # When `select_all=True`, `item_uuids` is ignored and every non-deleted
    # item in the dataset is targeted. Otherwise `item_uuids` must be non-empty.
    item_uuids: List[str] = []
    select_all: bool = False


@router.delete("/{dataset_id}/items")
async def remove_items(
    dataset_id: str,
    request: BulkDeleteItemsRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete multiple items from a dataset in one request.

    Pass `item_uuids` to delete a specific set, or `select_all=true` to clear
    every item in the dataset (in which case `item_uuids` is ignored). UUIDs not
    in this dataset (or already deleted) are skipped silently, so `deleted_count`
    reflects how many rows actually transitioned to deleted.
    """
    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not request.select_all and not request.item_uuids:
        raise HTTPException(
            status_code=400,
            detail="provide item_uuids, or set select_all=true",
        )

    deleted_count = delete_dataset_items(
        dataset_id,
        item_uuids=None if request.select_all else request.item_uuids,
    )
    return {"deleted_count": deleted_count}


@router.delete("/{dataset_id}/items/{item_uuid}", status_code=204)
async def remove_item(
    dataset_id: str,
    item_uuid: str,
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft delete a single item from a dataset."""
    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    deleted = delete_dataset_item(item_uuid, dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Item not found")
