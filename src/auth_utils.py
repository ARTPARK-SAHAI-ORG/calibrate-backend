"""JWT Authentication Utilities.

This module provides JWT token creation/validation for securing API endpoints.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret-key-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "168"))  # 7 days default

# Security scheme for Bearer token authentication
security = HTTPBearer(auto_error=True)


def create_access_token(user_uuid: str, email: str) -> str:
    """
    Create a JWT access token containing the user's UUID.

    Args:
        user_uuid: The unique identifier of the user
        email: The user's email (for logging/debugging)

    Returns:
        Encoded JWT token string
    """
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "sub": user_uuid,  # subject = user UUID
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    logger.debug(f"Created access token for user {user_uuid} (expires: {expire})")
    return token


def decode_token(token: str) -> Optional[dict]:
    """
    Decode and validate a JWT token.

    Args:
        token: The JWT token string

    Returns:
        Decoded payload dict if valid, None if invalid/expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.debug(f"Token decode failed: {e}")
        return None


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    FastAPI dependency to extract and validate user_id from JWT token.

    Usage:
        @router.post("/endpoint")
        async def endpoint(user_id: str = Depends(get_current_user_id)):
            # user_id is now available and validated
            pass

    Args:
        credentials: HTTP Authorization header credentials (injected by FastAPI)

    Returns:
        The user UUID extracted from the token

    Raises:
        HTTPException: 401 if token is missing, invalid, or expired
    """
    token = credentials.credentials
    payload = decode_token(token)

    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Token missing user information",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_id


SUPERADMIN_EMAIL = os.getenv("SUPERADMIN_EMAIL", "")


async def require_superadmin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    FastAPI dependency that requires the caller to be a superadmin.

    Extracts the email from the JWT and checks it against SUPERADMIN_EMAIL.
    Returns the user UUID if authorized.

    Raises:
        HTTPException: 401 if token invalid, 403 if not superadmin
    """
    user_id = await get_current_user_id(credentials)

    payload = decode_token(credentials.credentials)
    email = payload.get("email", "")
    if not SUPERADMIN_EMAIL or email != SUPERADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Superadmin access required")

    return user_id


# Optional dependency that doesn't require authentication
# Useful for endpoints that work with or without auth
async def get_optional_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> Optional[str]:
    """
    FastAPI dependency to optionally extract user_id from JWT token.

    Returns None if no token is provided or token is invalid.
    Useful for endpoints that should work for both authenticated and anonymous users.

    Args:
        credentials: Optional HTTP Authorization header credentials

    Returns:
        The user UUID if token is valid, None otherwise
    """
    if not credentials:
        return None

    payload = decode_token(credentials.credentials)
    if not payload:
        return None

    return payload.get("sub")


