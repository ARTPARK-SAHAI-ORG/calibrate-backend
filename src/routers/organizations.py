"""Organizations (workspaces) router.

The "org" terminology lives in DB/code; the UI calls them workspaces.

For now membership simply gates access — the actual switch of entity scoping
from `user_id` to `org_uuid` is a follow-up PR. Endpoints here only manage the
org graph (orgs, members, active workspace) without changing existing routers.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from auth_utils import get_current_user_id
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
    uuid: str
    name: str
    is_personal: bool
    created_by_user_id: str
    member_role: Optional[str] = None
    created_at: str
    updated_at: str


class CreateOrganizationRequest(BaseModel):
    name: str = Field(..., min_length=1)


class UpdateOrganizationRequest(BaseModel):
    name: str = Field(..., min_length=1)


class AddMemberRequest(BaseModel):
    email: str = Field(..., min_length=3)


class MemberResponse(BaseModel):
    user_id: str
    email: str
    first_name: str
    last_name: str
    role: str
    created_at: str


def _require_membership(org_uuid: str, user_id: str) -> str:
    """Resolve the caller's role in `org_uuid`, 404ing if not a member.

    Both owner and admin have the same permissions everywhere, so the role
    string is returned only for the few endpoints that distinguish (e.g.
    add/remove member, rename). 404 (not 403) keeps existence-leak parity
    with the rest of the codebase.
    """
    role = get_member_role(org_uuid, user_id)
    if role is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    return role


@router.get("", response_model=List[OrganizationResponse])
async def list_orgs(user_id: str = Depends(get_current_user_id)):
    """List every org the caller is an active member of."""
    return [OrganizationResponse(**o) for o in list_organizations_for_user(user_id)]


@router.post("", response_model=OrganizationResponse, status_code=201)
async def create_org(
    request: CreateOrganizationRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Create a new (non-personal) org with the caller as owner."""
    org_uuid = create_organization(name=request.name, owner_user_id=user_id)
    org = get_organization(org_uuid)
    return OrganizationResponse(**org, member_role="owner")


@router.patch("/{org_uuid}", response_model=OrganizationResponse)
async def rename_org(
    org_uuid: str,
    request: UpdateOrganizationRequest,
    user_id: str = Depends(get_current_user_id),
):
    role = _require_membership(org_uuid, user_id)
    update_organization_name(org_uuid, request.name)
    org = get_organization(org_uuid)
    return OrganizationResponse(**org, member_role=role)


@router.get("/{org_uuid}/members", response_model=List[MemberResponse])
async def list_members(
    org_uuid: str, user_id: str = Depends(get_current_user_id)
):
    _require_membership(org_uuid, user_id)
    return [MemberResponse(**m) for m in list_organization_members(org_uuid)]


@router.post(
    "/{org_uuid}/members", response_model=MemberResponse, status_code=201
)
async def add_member(
    org_uuid: str,
    request: AddMemberRequest,
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


@router.delete("/{org_uuid}/members/{target_user_id}", status_code=204)
async def remove_member(
    org_uuid: str,
    target_user_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Remove a member. Owners cannot be removed. Admins may remove themselves
    or any other admin."""
    _require_membership(org_uuid, user_id)
    try:
        removed = remove_organization_member(org_uuid, target_user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not removed:
        raise HTTPException(status_code=404, detail="Member not found")
    return None
