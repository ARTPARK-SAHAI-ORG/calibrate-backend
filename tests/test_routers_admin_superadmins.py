"""Adding and removing superadmins through /admin/superadmins."""

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import auth_utils
import db


@pytest.fixture(scope="module")
def client():
    import main

    with patch("main.recover_pending_jobs"):
        with TestClient(main.app) as c:
            yield c


def _signup(client):
    email = f"sa-{uuid.uuid4().hex[:8]}@example.com"
    body = client.post(
        "/auth/signup",
        json={"first_name": "S", "last_name": "A", "email": email, "password": "passw0rd"},
    ).json()
    return body["user"]["uuid"], email, {"Authorization": f"Bearer {body['access_token']}"}


@pytest.fixture
def root(client, monkeypatch):
    uid, email, h = _signup(client)
    monkeypatch.setattr(auth_utils, "SUPERADMIN_EMAIL", f"  {email.upper()} ")
    return uid, email, h


def test_added_superadmin_gets_every_superadmin_power(client, root):
    _, _, root_h = root
    uid, email, h = _signup(client)
    other_org = db.get_personal_org_for_user(root[0])["uuid"]

    assert client.get("/admin/analytics", headers=h).status_code == 403
    assert client.get("/admin/superadmins", headers=h).status_code == 403
    assert client.get("/agents", headers={**h, "X-Org-UUID": other_org}).status_code == 404

    added = client.post("/admin/superadmins", json={"email": f" {email.upper()} "}, headers=root_h)
    assert added.status_code == 200, added.text
    assert added.json() == {
        "uuid": uid, "email": email, "first_name": "S", "last_name": "A", "can_remove": True,
    }

    assert client.get("/admin/access", headers=h).json() == {"is_superadmin": True}
    assert client.get("/admin/analytics", headers=h).status_code == 200
    assert client.get("/agents", headers={**h, "X-Org-UUID": other_org}).status_code == 200

    listed = client.get("/admin/superadmins", headers=h).json()
    assert [(s["uuid"], s["can_remove"]) for s in listed if s["uuid"] in (uid, root[0])] == [
        (root[0], False),
        (uid, True),
    ]

    again = client.post("/admin/superadmins", json={"email": email}, headers=h)
    assert again.status_code == 200

    assert client.delete(f"/admin/superadmins/{uid}", headers=root_h).status_code == 204
    assert client.get("/admin/analytics", headers=h).status_code == 403
    assert uid not in [s["uuid"] for s in client.get("/admin/superadmins", headers=root_h).json()]


def test_add_refuses_an_email_with_no_account(client, root):
    r = client.post("/admin/superadmins", json={"email": "nobody-here@example.com"}, headers=root[2])
    assert r.status_code == 404
    assert r.json()["detail"] == "No Calibrate account uses this email"


def test_server_superadmin_cannot_be_removed(client, root):
    uid, _, h = root
    r = client.delete(f"/admin/superadmins/{uid}", headers=h)
    assert r.status_code == 400
    assert r.json()["detail"] == "This superadmin is set by the server and cannot be removed"


def test_removing_someone_who_is_not_a_superadmin_is_404(client, root):
    uid, _, _ = _signup(client)
    assert client.delete(f"/admin/superadmins/{uid}", headers=root[2]).status_code == 404
    assert client.delete(f"/admin/superadmins/{uuid.uuid4()}", headers=root[2]).status_code == 404


def test_database_refuses_an_unknown_role(client):
    uid, _, _ = _signup(client)
    with pytest.raises(Exception):
        db.set_user_role(uid, "owner")


def test_unset_server_email_lists_no_one_extra():
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users (uuid, first_name, last_name, email) VALUES (?, '', '', '')",
            (str(uuid.uuid4()),),
        )
        conn.commit()
    assert all(u["email"] for u in db.list_superadmins(""))
