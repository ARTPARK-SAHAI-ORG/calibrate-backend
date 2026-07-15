"""Tests for the organizations (workspaces) router and the underlying
multi-tenant primitives in db.py.

Covers:
  - signup auto-provisions a personal org with the user as owner
  - listing/creating/renaming/switching orgs
  - inviting an existing user vs an unregistered email (stub-user flow)
  - subsequent signup hydrates the stub user and preserves memberships
  - owner cannot be removed; admins can; current_org_uuid resets on removal
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app():
    import main as main_mod

    return main_mod.app


@pytest.fixture(scope="module")
def client(app):
    with patch("main.recover_pending_jobs"):
        with TestClient(app) as c:
            yield c


def _signup(client, email_prefix: str = "org"):
    suffix = uuid.uuid4().hex[:8]
    email = f"{email_prefix}-{suffix}@example.com"
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "O",
            "last_name": "U",
            "email": email,
            "password": "passw0rd",
        },
    ).json()
    return {
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
        "user_uuid": body["user"]["uuid"],
        "email": email,
        "access_token": body["access_token"],
    }


def test_signup_creates_personal_org_with_owner_membership(client):
    auth = _signup(client)
    resp = client.get("/organizations", headers=auth["headers"])
    assert resp.status_code == 200
    orgs = resp.json()
    assert len(orgs) == 1
    personal = orgs[0]
    assert personal["is_personal"] is True
    assert personal["created_by_user_id"] == auth["user_uuid"]
    assert personal["member_role"] == "owner"


def test_new_org_is_provisioned_with_editable_default_forks(client):
    """Both org-creation paths surface an editable fork of every seeded default
    through the API. A fresh org holds only forks, which read as `is_default`
    True (grouped under "Default", still editable) and expose no slug."""
    auth = _signup(client)
    h = auth["headers"]

    # Personal org (created at signup) — only forks yet, so all read as defaults.
    items = client.get("/evaluators", headers=h).json()["items"]
    assert items, "personal org should be provisioned with default forks"
    assert all(e["is_default"] is True for e in items)
    assert all(e.get("slug") is None for e in items)
    assert any(e["name"] == "Safety" for e in items)

    # Explicit workspace (scoped via X-Org-UUID) is provisioned too.
    org = client.post("/organizations", json={"name": "Acme"}, headers=h).json()
    scoped = {**h, "X-Org-UUID": org["uuid"]}
    new_items = client.get("/evaluators", headers=scoped).json()["items"]
    assert any(e["name"] == "Safety" for e in new_items)
    assert all(e["is_default"] is True for e in new_items)


def test_create_and_rename_org(client):
    auth = _signup(client)
    h = auth["headers"]

    create = client.post(
        "/organizations", json={"name": "Acme"}, headers=h
    )
    assert create.status_code == 201
    org = create.json()
    assert org["name"] == "Acme"
    assert org["is_personal"] is False
    assert org["member_role"] == "owner"

    rename = client.patch(
        f"/organizations/{org['uuid']}",
        json={"name": "Acme Inc"},
        headers=h,
    )
    assert rename.status_code == 200
    assert rename.json()["name"] == "Acme Inc"

    # Non-member sees 404
    other = _signup(client)
    other_resp = client.patch(
        f"/organizations/{org['uuid']}",
        json={"name": "Hacked"},
        headers=other["headers"],
    )
    assert other_resp.status_code == 404


def test_personal_org_lookup_is_implicit_default(client):
    """No `current_org_uuid` is persisted; the personal org is resolved
    on-demand via `get_personal_org_for_user`. Verify it returns the auto-
    provisioned org from signup."""
    import db

    auth = _signup(client)
    personal = db.get_personal_org_for_user(auth["user_uuid"])
    assert personal is not None
    assert personal["is_personal"] is True
    assert personal["created_by_user_id"] == auth["user_uuid"]


def test_add_existing_user_as_member(client):
    owner = _signup(client)
    member_auth = _signup(client, email_prefix="invitee")

    org = client.post(
        "/organizations", json={"name": "Team A"}, headers=owner["headers"]
    ).json()

    resp = client.post(
        f"/organizations/{org['uuid']}/members",
        json={"email": member_auth["email"]},
        headers=owner["headers"],
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["role"] == "admin"

    # Invitee now sees the org in their list
    listing = client.get("/organizations", headers=member_auth["headers"]).json()
    assert any(o["uuid"] == org["uuid"] for o in listing)

    # Duplicate invite → 400
    dup = client.post(
        f"/organizations/{org['uuid']}/members",
        json={"email": member_auth["email"]},
        headers=owner["headers"],
    )
    assert dup.status_code == 400


def test_invite_email_case_insensitive_hydration(client):
    """Invite stores the email lowercased; signup must hydrate the same stub
    even if the user types a differently-cased email."""
    owner = _signup(client)
    stub_email_lower = f"mixedcase-{uuid.uuid4().hex[:8]}@example.com"
    stub_email_mixed = stub_email_lower.replace("mixedcase", "MixedCase")

    org = client.post(
        "/organizations", json={"name": "Casing"}, headers=owner["headers"]
    ).json()
    client.post(
        f"/organizations/{org['uuid']}/members",
        json={"email": stub_email_mixed},
        headers=owner["headers"],
    )

    # Stub signs up with UPPER variation — should hydrate, not create new.
    signup = client.post(
        "/auth/signup",
        json={
            "first_name": "Mixed",
            "last_name": "Case",
            "email": stub_email_mixed.upper(),
            "password": "passw0rd",
        },
    )
    assert signup.status_code == 200, signup.text
    headers = {"Authorization": f"Bearer {signup.json()['access_token']}"}
    listing = client.get("/organizations", headers=headers).json()
    # Pre-invited org appears alongside the personal one.
    assert any(o["uuid"] == org["uuid"] for o in listing)


def test_invite_unregistered_email_creates_stub_user(client):
    owner = _signup(client)
    stub_email = f"stub-{uuid.uuid4().hex[:8]}@example.com"

    org = client.post(
        "/organizations", json={"name": "Pre-invite"}, headers=owner["headers"]
    ).json()

    resp = client.post(
        f"/organizations/{org['uuid']}/members",
        json={"email": stub_email},
        headers=owner["headers"],
    )
    assert resp.status_code == 201, resp.text

    # Now stub user signs up via password — should hydrate, not 409, and on
    # login they should see the pre-invited org.
    signup = client.post(
        "/auth/signup",
        json={
            "first_name": "Stub",
            "last_name": "User",
            "email": stub_email,
            "password": "passw0rd",
        },
    )
    assert signup.status_code == 200, signup.text
    new_headers = {"Authorization": f"Bearer {signup.json()['access_token']}"}

    listing = client.get("/organizations", headers=new_headers).json()
    org_uuids = [o["uuid"] for o in listing]
    assert org["uuid"] in org_uuids
    # And they got their own personal org too.
    assert any(o["is_personal"] for o in listing)

    # Second signup with the same email after hydration → 409.
    dup_signup = client.post(
        "/auth/signup",
        json={
            "first_name": "Stub",
            "last_name": "User",
            "email": stub_email,
            "password": "passw0rd",
        },
    )
    assert dup_signup.status_code == 409


def test_remove_member_and_owner_immutable(client):
    owner = _signup(client)
    member = _signup(client, email_prefix="rm")

    org = client.post(
        "/organizations", json={"name": "Team RM"}, headers=owner["headers"]
    ).json()
    client.post(
        f"/organizations/{org['uuid']}/members",
        json={"email": member["email"]},
        headers=owner["headers"],
    )

    # Owner cannot be removed
    bad = client.delete(
        f"/organizations/{org['uuid']}/members/{owner['user_uuid']}",
        headers=owner["headers"],
    )
    assert bad.status_code == 400

    # Admin removal works
    rm = client.delete(
        f"/organizations/{org['uuid']}/members/{member['user_uuid']}",
        headers=owner["headers"],
    )
    assert rm.status_code == 204

    # Removed member can no longer see the org
    listing = client.get("/organizations", headers=member["headers"]).json()
    assert not any(o["uuid"] == org["uuid"] for o in listing)


def test_members_list_only_visible_to_members(client):
    owner = _signup(client)
    outsider = _signup(client, email_prefix="out")

    org = client.post(
        "/organizations", json={"name": "Private"}, headers=owner["headers"]
    ).json()

    ok = client.get(
        f"/organizations/{org['uuid']}/members", headers=owner["headers"]
    )
    assert ok.status_code == 200
    assert len(ok.json()) == 1
    assert ok.json()[0]["role"] == "owner"

    denied = client.get(
        f"/organizations/{org['uuid']}/members",
        headers=outsider["headers"],
    )
    assert denied.status_code == 404


def test_superadmin_can_manage_org_without_membership(client, monkeypatch):
    """Superadmin bypass: a user matching SUPERADMIN_EMAIL can list members,
    add a member, and rename any existing org without being in org_members.
    """
    import auth_utils

    owner = _signup(client)
    outsider = _signup(client, email_prefix="superadmin")
    invitee = _signup(client, email_prefix="invitee-by-admin")

    org = client.post(
        "/organizations", json={"name": "Customer Org"}, headers=owner["headers"]
    ).json()

    # Promote outsider to superadmin via env override.
    monkeypatch.setattr(auth_utils, "SUPERADMIN_EMAIL", outsider["email"])

    # Listing members works even though outsider is not a member.
    listing = client.get(
        f"/organizations/{org['uuid']}/members", headers=outsider["headers"]
    )
    assert listing.status_code == 200

    # Adding a member works.
    added = client.post(
        f"/organizations/{org['uuid']}/members",
        json={"email": invitee["email"]},
        headers=outsider["headers"],
    )
    assert added.status_code == 201

    # Renaming works.
    renamed = client.patch(
        f"/organizations/{org['uuid']}",
        json={"name": "Customer Org Renamed"},
        headers=outsider["headers"],
    )
    assert renamed.status_code == 200

    # Bypass requires the org to actually exist.
    missing = client.patch(
        "/organizations/00000000-0000-0000-0000-000000000000",
        json={"name": "ghost"},
        headers=outsider["headers"],
    )
    assert missing.status_code == 404


def test_per_org_unique_indexes_exist_after_init(client):
    """The per-org partial unique indexes (`idx_*_org_name_active`) must be
    created on a fresh DB — i.e. they're created AFTER the `ALTER TABLE ...
    ADD COLUMN org_uuid` migration. Earlier ordering had them run first;
    SQLite would silently fail on `no such column: org_uuid`."""
    import db

    expected = {
        "idx_tests_org_name_active",
        "idx_agents_org_name_active",
        "idx_tools_org_name_active",
        "idx_personas_org_name_active",
        "idx_scenarios_org_name_active",
        "idx_simulations_org_name_active",
        "idx_annotation_tasks_org_name_active",
        "idx_annotators_org_name_active",
        "idx_evaluators_org_name_active",
    }
    with db.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name LIKE 'idx_%_org_name_active'"
        )
        present = {row["name"] for row in cursor.fetchall()}
    missing = expected - present
    assert not missing, f"missing per-org unique indexes: {sorted(missing)}"


def test_email_normalization_round_trip(client):
    """Every write path (signup, Google login, invite) and every read path
    (login lookup) must agree on the same normalized email so a user
    registered with mixed-case can be invited / looked up consistently."""
    import db

    suffix = uuid.uuid4().hex[:8]
    mixed = f"User-{suffix}@Example.COM"
    lowered = mixed.strip().lower()

    # Helper is the single source of truth.
    assert db.normalize_email(mixed) == lowered
    assert db.normalize_email(f"  {mixed}  ") == lowered
    assert db.normalize_email(None) == ""

    # Signup with mixed case stores lowercased; login with any casing finds it.
    signup = client.post(
        "/auth/signup",
        json={
            "first_name": "F",
            "last_name": "L",
            "email": mixed,
            "password": "passw0rd",
        },
    )
    assert signup.status_code == 200
    stored = db.get_user_by_email(mixed.upper())
    assert stored is not None
    assert stored["email"] == lowered


def test_add_organization_member_rejects_phantom_org(client):
    """`add_organization_member` must validate that `org_uuid` references an
    existing non-deleted org. The API path is already gated by membership, but
    the helper itself should refuse phantom UUIDs so future callers (admin
    scripts, backfills, new endpoints) can't quietly create orphan rows."""
    import db

    with pytest.raises(ValueError, match="organization not found"):
        db.add_organization_member(
            org_uuid="00000000-0000-0000-0000-000000000000",
            email=f"phantom-{uuid.uuid4().hex[:8]}@example.com",
        )


def test_init_db_backfill_is_idempotent(client):
    """Re-running init_db on an already-migrated DB must not create duplicate
    personal orgs or double-tag entity rows."""
    import db

    # Snapshot org count before
    auth = _signup(client)
    before = db.list_organizations_for_user(auth["user_uuid"])
    assert len(before) == 1

    db.init_db()
    db.init_db()

    after = db.list_organizations_for_user(auth["user_uuid"])
    assert len(after) == 1
    assert after[0]["uuid"] == before[0]["uuid"]
