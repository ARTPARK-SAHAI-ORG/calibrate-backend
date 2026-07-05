"""Organizations (workspaces) router.

The "org" terminology lives in DB/code; the UI calls them workspaces.

For now membership simply gates access — the actual switch of entity scoping
from `user_id` to `org_uuid` is a follow-up PR. Endpoints here only manage the
org graph (orgs, members, active workspace) without changing existing routers.
"""

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field
from typing import List, Optional

from auth_utils import get_current_user_id, is_superadmin_user
from db import (
    add_organization_member,
    create_organization,
    get_member_role,
    get_organization,
    list_organization_members,
    list_organizations_for_user,
    remove_organization_member,
    update_organization_name,
)

router = APIRouter(prefix="/organizations", tags=["organizations"])


class OrganizationResponse(BaseModel):
    uuid: str = Field(description="Organization (workspace) identifier (8-char UUID)")
    name: str = Field(description="Workspace display name")
    is_personal: bool = Field(
        description="`true` for the user's auto-created personal workspace, `false` for shared orgs"
    )
    created_by_user_id: str = Field(description="UUID of the user who created the org")
    member_role: Optional[str] = Field(
        None,
        description="Caller's role in this org (`owner` | `admin`); `null` when not resolved",
    )
    created_at: str = Field(description="Creation timestamp (ISO 8601 UTC)")
    updated_at: str = Field(description="Last-update timestamp (ISO 8601 UTC)")


class CreateOrganizationRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Workspace name (non-empty)")


class UpdateOrganizationRequest(BaseModel):
    name: str = Field(..., min_length=1, description="New workspace name (non-empty)")


class AddMemberRequest(BaseModel):
    email: str = Field(
        ...,
        min_length=3,
        description="Email of the user to add; a stub user is created if not yet registered",
    )


class MemberResponse(BaseModel):
    user_id: str = Field(description="Member's user UUID (8-char identifier)")
    email: str = Field(description="Member's email address")
    first_name: str = Field(description="Member's given name")
    last_name: str = Field(description="Member's family name")
    role: str = Field(description="Member's role in the org (`owner` | `admin`)")
    created_at: str = Field(description="When the member was added (ISO 8601 UTC)")


def _require_membership(org_uuid: str, user_id: str) -> str:
    """Resolve the caller's role in `org_uuid`, 404ing if not a member.

    Superadmin bypass: any existing org grants owner-level access.
    """
    role = get_member_role(org_uuid, user_id)
    if role is None:
        if is_superadmin_user(user_id) and get_organization(org_uuid) is not None:
            return "owner"
        raise HTTPException(status_code=404, detail="Organization not found")
    return role


@router.get("", response_model=List[OrganizationResponse], summary="List organizations")
async def list_orgs(user_id: str = Depends(get_current_user_id)):
    """List every org (workspace) the caller is an active member of."""
    return [OrganizationResponse(**o) for o in list_organizations_for_user(user_id)]


@router.post("", response_model=OrganizationResponse, status_code=201, summary="Create organization")
async def create_org(
    request: CreateOrganizationRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Create a new (non-personal) org (workspace) with the caller as owner."""
    org_uuid = create_organization(name=request.name, owner_user_id=user_id)
    org = get_organization(org_uuid)
    return OrganizationResponse(**org, member_role="owner")


@router.patch("/{org_uuid}", response_model=OrganizationResponse, summary="Update organization")
async def rename_org(
    org_uuid: str = Path(description="Organization UUID (8-char identifier)"),
    request: UpdateOrganizationRequest = ...,
    user_id: str = Depends(get_current_user_id),
):
    """Rename an org. The caller must be a member; returns 404 otherwise."""
    role = _require_membership(org_uuid, user_id)
    update_organization_name(org_uuid, request.name)
    org = get_organization(org_uuid)
    return OrganizationResponse(**org, member_role=role)


@router.get("/{org_uuid}/members", response_model=List[MemberResponse], summary="List members")
async def list_members(
    org_uuid: str = Path(description="Organization UUID (8-char identifier)"),
    user_id: str = Depends(get_current_user_id),
):
    """List members of an org. The caller must be a member; returns 404 otherwise."""
    _require_membership(org_uuid, user_id)
    return [MemberResponse(**m) for m in list_organization_members(org_uuid)]


@router.post(
    "/{org_uuid}/members",
    response_model=MemberResponse,
    status_code=201,
    summary="Add member",
)
async def add_member(
    org_uuid: str = Path(description="Organization UUID (8-char identifier)"),
    request: AddMemberRequest = ...,
    user_id: str = Depends(get_current_user_id),
):
    """Add a user to this org as admin. Creates a stub user if the email isn't
    yet registered — when that person signs up, the existing row is hydrated
    and they immediately see this workspace."""
    _require_membership(org_uuid, user_id)
    try:
        member = add_organization_member(
            org_uuid=org_uuid, email=request.email, role="admin"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Re-read the full member row so the response has the joined user fields.
    for m in list_organization_members(org_uuid):
        if m["user_id"] == member["user_id"]:
            return MemberResponse(**m)
    raise HTTPException(status_code=500, detail="Member not found after insert")


@router.delete("/{org_uuid}/members/{target_user_id}", status_code=204, summary="Remove member")
async def remove_member(
    org_uuid: str = Path(description="Organization UUID (8-char identifier)"),
    target_user_id: str = Path(description="UUID of the member to remove (8-char identifier)"),
    user_id: str = Depends(get_current_user_id),
):
    """Remove a member from the org. Owners cannot be removed. Admins may remove
    themselves or any other admin. Returns 404 if the member isn't found."""
    _require_membership(org_uuid, user_id)
    try:
        removed = remove_organization_member(org_uuid, target_user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not removed:
        raise HTTPException(status_code=404, detail="Member not found")
    return None
