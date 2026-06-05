import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from db import get_user
from auth_utils import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


class UserResponse(BaseModel):
    """User response model."""

    uuid: str
    first_name: str
    last_name: str
    email: str
    created_at: str
    updated_at: str


@router.get("/{user_uuid}", response_model=UserResponse)
async def get_user_endpoint(
    user_uuid: str,
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Get user information by UUID.

    Requires a valid JWT, and a user may only fetch their own record — any
    other UUID returns 404 (existence-leak parity with the rest of the API).

    Args:
        user_uuid: The user's UUID

    Returns:
        User information
    """
    if user_uuid != current_user_id:
        raise HTTPException(status_code=404, detail="User not found")

    user = get_user(user_uuid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        uuid=user["uuid"],
        first_name=user["first_name"],
        last_name=user["last_name"],
        email=user["email"],
        created_at=user["created_at"],
        updated_at=user["updated_at"],
    )
