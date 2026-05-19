import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import httpx
import bcrypt

from db import (
    get_or_create_user,
    get_user_by_email,
    create_user_with_password,
    get_user,
)
from auth_utils import create_access_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


class GoogleLoginRequest(BaseModel):
    """Request body for Google login."""

    id_token: str


class UserResponse(BaseModel):
    """User response model."""

    uuid: str
    first_name: str
    last_name: str
    email: str
    created_at: str
    updated_at: str


class LoginResponse(BaseModel):
    """Response for successful login."""

    access_token: str
    token_type: str = "bearer"
    user: UserResponse
    message: str


async def verify_google_token(id_token: str) -> dict:
    """
    Verify Google ID token using Google's tokeninfo endpoint.

    Args:
        id_token: The Google ID token from the frontend

    Returns:
        Dict containing user info (email, given_name, family_name, etc.)

    Raises:
        HTTPException: If token verification fails
    """
    google_client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not google_client_id:
        logger.warning("GOOGLE_CLIENT_ID not set, skipping client_id validation")

    # Use Google's tokeninfo endpoint to verify the token
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
            )

            if response.status_code != 200:
                logger.error(f"Google token verification failed: {response.text}")
                raise HTTPException(status_code=401, detail="Invalid Google token")

            token_info = response.json()

            return token_info

        except httpx.RequestError as e:
            logger.error(f"Error verifying Google token: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to verify Google token")


@router.post("/google", response_model=LoginResponse)
async def google_login(request: GoogleLoginRequest):
    """
    Authenticate a user with Google OAuth.

    This endpoint:
    1. Verifies the Google ID token with Google's servers
    2. Extracts user info (email, name) from the token
    3. Creates a new user or retrieves existing user by email
    4. Returns the user info

    Args:
        request: Contains the Google ID token from the frontend

    Returns:
        LoginResponse with user info and success message
    """
    # Verify the Google token
    token_info = await verify_google_token(request.id_token)

    # Extract user info from token
    email = token_info.get("email")
    given_name = token_info.get("given_name", "")
    family_name = token_info.get("family_name", "")

    if not email:
        raise HTTPException(
            status_code=400, detail="Email not provided in Google token"
        )

    # Get or create user
    user = get_or_create_user(
        email=email,
        first_name=given_name,
        last_name=family_name,
    )

    # Generate JWT access token
    access_token = create_access_token(user["uuid"], user["email"])

    logger.info(f"User logged in: {email} (UUID: {user['uuid']})")

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            uuid=user["uuid"],
            first_name=user["first_name"],
            last_name=user["last_name"],
            email=user["email"],
            created_at=user["created_at"],
            updated_at=user["updated_at"],
        ),
        message="Login successful",
    )


class SignupRequest(BaseModel):
    first_name: str = Field(..., min_length=1)
    last_name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)


class CredentialLoginRequest(BaseModel):
    email: str
    password: str


@router.post("/signup", response_model=LoginResponse)
async def signup(request: SignupRequest):
    """
    Register a new user with email and password.

    Returns a JWT access token and user info on success.
    """
    # A row may already exist as a stub created by an org invite (no
    # password_hash set). `create_user_with_password` hydrates that stub in
    # place; it raises ValueError("email already registered") if the row
    # already has a password set.
    existing = get_user_by_email(request.email)
    if existing and existing.get("password_hash"):
        raise HTTPException(status_code=409, detail="Email already taken")

    password_hash = bcrypt.hashpw(
        request.password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

    try:
        user_uuid = create_user_with_password(
            first_name=request.first_name,
            last_name=request.last_name,
            email=request.email,
            password_hash=password_hash,
        )
    except ValueError:
        raise HTTPException(status_code=409, detail="Email already taken")

    user = get_user(user_uuid)

    access_token = create_access_token(user["uuid"], user["email"])
    logger.info(f"User signed up: {request.email} (UUID: {user['uuid']})")

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            uuid=user["uuid"],
            first_name=user["first_name"],
            last_name=user["last_name"],
            email=user["email"],
            created_at=user["created_at"],
            updated_at=user["updated_at"],
        ),
        message="Signup successful",
    )


@router.post("/login", response_model=LoginResponse)
async def login(request: CredentialLoginRequest):
    """
    Authenticate a user with email and password.

    Returns a JWT access token and user info on success.
    """
    user = get_user_by_email(request.email)
    if not user or not user.get("password_hash"):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not bcrypt.checkpw(
        request.password.encode("utf-8"),
        user["password_hash"].encode("utf-8"),
    ):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token(user["uuid"], user["email"])
    logger.info(f"User logged in: {request.email} (UUID: {user['uuid']})")

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            uuid=user["uuid"],
            first_name=user["first_name"],
            last_name=user["last_name"],
            email=user["email"],
            created_at=user["created_at"],
            updated_at=user["updated_at"],
        ),
        message="Login successful",
    )
