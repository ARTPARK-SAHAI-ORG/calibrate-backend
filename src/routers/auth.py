import hashlib
import os
import logging
import secrets
from typing import Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
import bcrypt

from db import (
    get_or_create_user,
    get_user_by_email,
    create_user,
    create_user_with_password,
    get_user,
    get_user_by_token,
    list_organizations_for_user,
    mark_email_verified,
    set_user_token,
)
from auth_utils import create_access_token
from mailer import (
    send_email,
    email_enabled,
    frontend_url,
    WELCOME_TEMPLATE,
    VERIFY_EMAIL_TEMPLATE,
    RESET_PASSWORD_TEMPLATE,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


class GoogleLoginRequest(BaseModel):
    id_token: str = Field(
        description="Google Sign-In ID token from your client"
    )


class UserResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Your user ID",
    )
    first_name: str = Field(description="Your given name")
    last_name: str = Field(description="Your family name")
    email: str = Field(description="Your email address")
    created_at: str = Field(description="When your account was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When your profile was last updated (ISO 8601 UTC)")


class LoginResponse(BaseModel):
    access_token: str = Field(
        description="JWT to send as `Authorization: Bearer <token>` on later requests"
    )
    token_type: str = Field("bearer", description="Always `bearer`")
    user: UserResponse = Field(description="Your profile")
    message: str = Field(description="Status message")


RESEND_COOLDOWN_SECONDS = 60
VERIFY_LINK_SECONDS = 24 * 3600
RESET_LINK_SECONDS = 3600
# An expired link is still recognised for this long, so its page can offer a new one.
ANY_AGE_SECONDS = 10 * 365 * 24 * 3600


def _login_response(user: dict, message: str) -> LoginResponse:
    return LoginResponse(
        access_token=create_access_token(user["uuid"], user["email"]),
        token_type="bearer",
        user=UserResponse(
            uuid=user["uuid"],
            first_name=user["first_name"],
            last_name=user["last_name"],
            email=user["email"],
            created_at=user["created_at"],
            updated_at=user["updated_at"],
        ),
        message=message,
    )


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# bcrypt refuses anything longer, which was a server error instead of a message.
MAX_PASSWORD_BYTES = 72


def _hash_password(password: str) -> str:
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise HTTPException(status_code=422, detail="Password is too long")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _password_matches(password: str, password_hash: Optional[str]) -> bool:
    # No stored password can be that long, so a longer one simply does not match.
    if not password_hash or len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return False
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def _send_welcome_unless_invited(user: dict) -> None:
    """Welcome a newly confirmed account, unless an invite to someone else's workspace already emailed them."""
    orgs = list_organizations_for_user(user["uuid"])
    if all(org["created_by_user_id"] == user["uuid"] for org in orgs):
        _send_welcome_email(user["email"], user["first_name"])


def _send_token_email(
    user: dict, kind: str, next_path: Optional[str], **signup_fields
) -> None:
    """Rotate the user's verify or reset token and email its link, unless one went out under a minute ago."""
    token = secrets.token_urlsafe(32)
    if not set_user_token(
        user["uuid"], kind, _hash_token(token), RESEND_COOLDOWN_SECONDS, **signup_fields
    ):
        return
    page, template = (
        ("verify-email", VERIFY_EMAIL_TEMPLATE)
        if kind == "verify"
        else ("reset-password", RESET_PASSWORD_TEMPLATE)
    )
    url = f"{frontend_url()}/{page}?token={token}"
    # Only a path on the frontend itself; `//host` and `/\host` leave the site.
    if next_path and next_path.startswith("/") and not next_path.startswith(("//", "/\\")):
        url += "&callbackUrl=" + quote(next_path, safe="")
    name = signup_fields.get("first_name") or user["first_name"]
    send_email(
        to=user["email"],
        template=template,
        variables={"NAME": name.strip() or "there", "URL": url},
    )


def _send_welcome_email(email: str, first_name: str) -> None:
    send_email(
        to=email,
        template=WELCOME_TEMPLATE,
        variables={
            "NAME": first_name.strip() or "there",
            "URL": frontend_url(),
        },
    )


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
            # Any app's Google token passes tokeninfo, so the audience must be ours.
            if (
                not google_client_id
                or token_info.get("aud") != google_client_id
                or token_info.get("email_verified") not in ("true", True)
            ):
                logger.error("Google token has the wrong audience or an unverified email")
                raise HTTPException(status_code=401, detail="Invalid Google token")

            return token_info

        except httpx.RequestError as e:
            logger.error(f"Error verifying Google token: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to verify Google token")


@router.post("/google", response_model=LoginResponse, summary="Log in with Google")
async def google_login(request: GoogleLoginRequest):
    """Log in with Google and receive a JWT plus your profile"""
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

    existing = get_user_by_email(email)
    # A row nobody confirmed (an invite stub, or a signup someone else started
    # with this address) still means this is the person's first real sign in.
    first_sign_in = existing is None or not existing["email_verified_at"]

    user = get_or_create_user(
        email=email,
        first_name=given_name,
        last_name=family_name,
    )

    # Google has confirmed the address, so any signup still waiting on its link
    # (possibly started by someone else) must stop being usable.
    mark_email_verified(user["uuid"])

    if first_sign_in:
        _send_welcome_unless_invited(user)

    logger.info(f"User logged in: {email} (UUID: {user['uuid']})")
    return _login_response(user, "Login successful")


class SignupRequest(BaseModel):
    first_name: str = Field(..., min_length=1, description="Your given name")
    last_name: str = Field(..., min_length=1, description="Your family name")
    email: str = Field(..., min_length=3, description="Email address for your new account")
    password: str = Field(..., min_length=6, description="Account password")
    next: Optional[str] = Field(
        None,
        description="Page to return to after following the emailed link. Only a path starting with `/` is kept",
    )


class CredentialLoginRequest(BaseModel):
    email: str = Field(description="Email address for your account")
    password: str = Field(description="Your account password")
    next: Optional[str] = Field(
        None,
        description="Page to return to after following the emailed link. Only a path starting with `/` is kept",
    )


class VerificationSentResponse(BaseModel):
    status: Literal["verification_sent"] = Field(
        description="Always `verification_sent`: open the emailed link to finish signing up"
    )
    email: str = Field(description="Address the verification link was sent to")
    already_pending: bool = Field(
        description="Whether an earlier signup for this address was still waiting to be verified"
    )


class VerifyEmailRequest(BaseModel):
    token: str = Field(min_length=1, description="Token from the verification link")


class ResendVerificationRequest(BaseModel):
    token: str = Field(min_length=1, description="Token from the expired verification link")
    next: Optional[str] = Field(
        None,
        description="Page to return to after following the emailed link. Only a path starting with `/` is kept",
    )


class ResendVerificationResponse(BaseModel):
    status: Literal["sent"] = Field(description="Always `sent`: a new verification link was emailed")
    email: str = Field(description="Address the new verification link was sent to")


class ForgotPasswordRequest(BaseModel):
    email: str = Field(description="Email address for your account")
    next: Optional[str] = Field(
        None,
        description="Page to return to after following the emailed link. Only a path starting with `/` is kept",
    )


class ForgotPasswordResponse(BaseModel):
    status: Literal["sent"] = Field(
        description="Always `sent`, whether or not an account exists for the address"
    )


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=1, description="Token from the reset link")
    password: str = Field(..., min_length=6, description="New account password")


@router.post(
    "/signup",
    response_model=LoginResponse,
    responses={202: {"model": VerificationSentResponse}},
    summary="Sign up with email and password",
)
def signup(request: SignupRequest):
    """Create an account with email and password, verified through an emailed link when email is on"""
    if not email_enabled():
        # An invite stub row is hydrated in place; any other existing account
        # (password or Google) makes `create_user_with_password` raise, which is a 409.
        existing = get_user_by_email(request.email)
        try:
            user_uuid = create_user_with_password(
                first_name=request.first_name,
                last_name=request.last_name,
                email=request.email,
                password_hash=_hash_password(request.password),
            )
        except ValueError:
            raise HTTPException(status_code=409, detail="Email already taken")

        user = get_user(user_uuid)
        logger.info(f"User signed up: {request.email} (UUID: {user['uuid']})")
        # A stub row means an invite already emailed them; only a brand-new account
        # gets the welcome, matching the Google path.
        if existing is None:
            _send_welcome_email(user["email"], user["first_name"])
        return _login_response(user, "Signup successful")

    user = get_user_by_email(request.email)
    if user is not None and user["email_verified_at"]:
        raise HTTPException(status_code=409, detail="Email already taken")
    if user is None:
        user = get_user(
            create_user(request.first_name, request.last_name, request.email)
        )

    # password_hash stays empty until the link is opened, so nobody can log in
    # with a password the address owner never confirmed.
    _send_token_email(
        user,
        "verify",
        request.next,
        first_name=request.first_name,
        last_name=request.last_name,
        verify_password_hash=_hash_password(request.password),
    )
    return JSONResponse(
        status_code=202,
        content=VerificationSentResponse(
            status="verification_sent",
            email=user["email"],
            already_pending=bool(user["verify_token_hash"]),
        ).model_dump(),
    )


@router.post("/login", response_model=LoginResponse, summary="Log in with email and password")
def login(request: CredentialLoginRequest):
    """Log in with email and password and receive a JWT plus your profile"""
    user = get_user_by_email(request.email)
    if user and _password_matches(request.password, user["password_hash"]):
        logger.info(f"User logged in: {request.email} (UUID: {user['uuid']})")
        return _login_response(user, "Login successful")

    if (
        user
        and not user["email_verified_at"]
        and _password_matches(request.password, user["verify_password_hash"])
    ):
        _send_token_email(user, "verify", request.next)
        raise HTTPException(status_code=403, detail="email_not_verified")

    raise HTTPException(status_code=401, detail="Invalid email or password")


@router.post("/verify-email", response_model=LoginResponse, summary="Verify email address")
def verify_email(request: VerifyEmailRequest):
    """Finish signing up with the token from the verification email and receive a JWT plus your profile"""
    token_hash = _hash_token(request.token)
    user = get_user_by_token("verify", token_hash, VERIFY_LINK_SECONDS)
    if user is None:
        # Still the latest link, only too old: the page can offer a new one.
        if get_user_by_token("verify", token_hash, ANY_AGE_SECONDS):
            raise HTTPException(status_code=400, detail="expired_token")
        raise HTTPException(status_code=400, detail="invalid_or_expired_token")

    mark_email_verified(user["uuid"], user["verify_password_hash"])
    user = get_user(user["uuid"])

    _send_welcome_unless_invited(user)

    logger.info(f"User confirmed email: {user['email']} (UUID: {user['uuid']})")
    return _login_response(user, "Email confirmed")


@router.post(
    "/resend-verification",
    response_model=ResendVerificationResponse,
    status_code=202,
    summary="Resend verification email",
)
def resend_verification(request: ResendVerificationRequest):
    """Email a new verification link in place of an expired one"""
    user = get_user_by_token("verify", _hash_token(request.token), ANY_AGE_SECONDS)
    # Only the latest link can ask for another: verifying clears it and a newer
    # link replaces it, so a used or replaced link cannot send a new email.
    if user is None:
        raise HTTPException(status_code=400, detail="invalid_or_expired_token")
    _send_token_email(user, "verify", request.next)
    return ResendVerificationResponse(status="sent", email=user["email"])


@router.post(
    "/forgot-password",
    response_model=ForgotPasswordResponse,
    status_code=202,
    summary="Request password reset",
)
def forgot_password(request: ForgotPasswordRequest):
    """Email a link to choose a new password, without revealing whether an account exists"""
    user = get_user_by_email(request.email)
    # A bare invite stub has nobody behind it yet, so it gets no reset link.
    if email_enabled() and user and (user["email_verified_at"] or user["verify_token_hash"]):
        _send_token_email(user, "reset", request.next)
    return ForgotPasswordResponse(status="sent")


@router.post("/reset-password", response_model=LoginResponse, summary="Reset password")
def reset_password(request: ResetPasswordRequest):
    """Set a new password with the token from the reset email and receive a JWT plus your profile"""
    user = get_user_by_token("reset", _hash_token(request.token), RESET_LINK_SECONDS)
    if user is None:
        raise HTTPException(status_code=400, detail="invalid_or_expired_token")

    # ponytail: JWTs issued before the reset stay valid until they expire; add a
    # per-user token version checked in auth_utils if a reset must log others out.
    mark_email_verified(user["uuid"], _hash_password(request.password))
    logger.info(f"User reset password: {user['email']} (UUID: {user['uuid']})")
    return _login_response(get_user(user["uuid"]), "Password reset")
