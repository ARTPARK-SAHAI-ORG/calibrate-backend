"""Storage, encryption, and run-environment building for workspace provider keys."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet

import db
import provider_keys as pk


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch):
    monkeypatch.setenv(pk.ENCRYPTION_KEY_ENV, Fernet.generate_key().decode())
    yield


@pytest.fixture
def org():
    return str(uuid.uuid4())


def _store(org_uuid, provider, values):
    return db.set_org_provider_keys(
        org_uuid=org_uuid,
        owner_user_id=str(uuid.uuid4()),
        provider=provider,
        rows=pk.encrypted_rows_for(provider, values),
    )


SERVICE_ACCOUNT = json.dumps(
    {"type": "service_account", "client_email": "runner@proj.iam.gserviceaccount.com"}
)


def test_value_round_trips_through_encryption(org):
    _store(org, "sarvam", {"SARVAM_API_KEY": "sk-live-abcd1234"})
    assert pk.workspace_env_values(org) == {"SARVAM_API_KEY": "sk-live-abcd1234"}


def test_stored_value_is_not_readable_as_plain_text(org):
    rows = _store(org, "sarvam", {"SARVAM_API_KEY": "sk-live-abcd1234"})
    assert "sk-live-abcd1234" not in rows[0]["value_encrypted"]


def test_secret_display_shows_only_the_last_four(org):
    rows = _store(org, "sarvam", {"SARVAM_API_KEY": "sk-live-abcd1234"})
    assert rows[0]["display"] == "••••1234"


def test_project_id_displays_in_full_but_credentials_show_the_account(org):
    rows = _store(
        org,
        "google",
        {
            "GOOGLE_APPLICATION_CREDENTIALS": SERVICE_ACCOUNT,
            "GOOGLE_CLOUD_PROJECT_ID": "my-project",
        },
    )
    display = {r["env_var"]: r["display"] for r in rows}
    assert display["GOOGLE_CLOUD_PROJECT_ID"] == "my-project"
    assert display["GOOGLE_APPLICATION_CREDENTIALS"] == (
        "runner@proj.iam.gserviceaccount.com"
    )


def test_half_configured_google_is_rejected(org):
    with pytest.raises(pk.ProviderKeyError) as exc:
        pk.encrypted_rows_for("google", {"GOOGLE_CLOUD_PROJECT_ID": "my-project"})
    assert "GOOGLE_APPLICATION_CREDENTIALS" in str(exc.value)


def test_credentials_that_are_not_json_are_rejected():
    with pytest.raises(pk.ProviderKeyError):
        pk.encrypted_rows_for(
            "google",
            {
                "GOOGLE_APPLICATION_CREDENTIALS": "/some/path.json",
                "GOOGLE_CLOUD_PROJECT_ID": "my-project",
            },
        )


def test_unknown_provider_is_rejected():
    with pytest.raises(pk.ProviderKeyError):
        pk.encrypted_rows_for("not-a-provider", {"X": "y"})


def test_value_for_the_wrong_provider_is_rejected():
    with pytest.raises(pk.ProviderKeyError):
        pk.encrypted_rows_for("sarvam", {"OPENAI_API_KEY": "sk-1"})


def test_saving_a_provider_again_replaces_the_old_value(org):
    _store(org, "sarvam", {"SARVAM_API_KEY": "first-key"})
    _store(org, "sarvam", {"SARVAM_API_KEY": "second-key"})
    assert pk.workspace_env_values(org) == {"SARVAM_API_KEY": "second-key"}
    assert len(db.get_org_provider_keys(org)) == 1


def test_deleting_then_adding_again_works(org):
    _store(org, "sarvam", {"SARVAM_API_KEY": "first-key"})
    assert db.soft_delete_org_provider_keys(org, "sarvam") == 1
    assert pk.workspace_env_values(org) == {}
    _store(org, "sarvam", {"SARVAM_API_KEY": "third-key"})
    assert pk.workspace_env_values(org) == {"SARVAM_API_KEY": "third-key"}


def test_one_workspace_cannot_see_another(org):
    other = str(uuid.uuid4())
    _store(org, "sarvam", {"SARVAM_API_KEY": "mine"})
    assert pk.workspace_env_values(other) == {}


def test_covered_providers_needs_every_value(org):
    _store(org, "sarvam", {"SARVAM_API_KEY": "k"})
    assert pk.covered_providers(org) == {"sarvam"}
    _store(
        org,
        "google",
        {
            "GOOGLE_APPLICATION_CREDENTIALS": SERVICE_ACCOUNT,
            "GOOGLE_CLOUD_PROJECT_ID": "my-project",
        },
    )
    assert pk.covered_providers(org) == {"sarvam", "google"}


def test_no_keys_means_the_run_environment_is_untouched(org):
    assert pk.provider_env(org, "/tmp") is None


def test_workspace_value_overrides_the_server(org, monkeypatch, tmp_path):
    monkeypatch.setenv("SARVAM_API_KEY", "server-key")
    _store(org, "sarvam", {"SARVAM_API_KEY": "workspace-key"})
    env = pk.provider_env(org, tmp_path)
    assert env["SARVAM_API_KEY"] == "workspace-key"


def test_server_keys_survive_for_providers_the_workspace_did_not_set(
    org, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "server-openai")
    _store(org, "sarvam", {"SARVAM_API_KEY": "workspace-sarvam"})
    env = pk.provider_env(org, tmp_path)
    assert env["OPENAI_API_KEY"] == "server-openai"
    assert env["SARVAM_API_KEY"] == "workspace-sarvam"


def test_credentials_are_written_into_the_run_directory(org, tmp_path):
    _store(
        org,
        "google",
        {
            "GOOGLE_APPLICATION_CREDENTIALS": SERVICE_ACCOUNT,
            "GOOGLE_CLOUD_PROJECT_ID": "my-project",
        },
    )
    env = pk.provider_env(org, tmp_path)

    written = Path(env["GOOGLE_APPLICATION_CREDENTIALS"])
    assert written.parent == tmp_path
    assert json.loads(written.read_text())["client_email"] == (
        "runner@proj.iam.gserviceaccount.com"
    )
    assert oct(written.stat().st_mode)[-3:] == "600"
    assert env["GOOGLE_CLOUD_PROJECT_ID"] == "my-project"


def test_a_rotated_encryption_key_falls_back_to_the_server(org, monkeypatch):
    _store(org, "sarvam", {"SARVAM_API_KEY": "workspace-key"})
    monkeypatch.setenv(pk.ENCRYPTION_KEY_ENV, Fernet.generate_key().decode())

    assert pk.workspace_env_values(org) == {}
    assert pk.covered_providers(org) == set()
    assert pk.provider_env(org, "/tmp") is None


def test_no_encryption_key_configured_blocks_saving_and_reading(org, monkeypatch):
    _store(org, "sarvam", {"SARVAM_API_KEY": "workspace-key"})
    monkeypatch.delenv(pk.ENCRYPTION_KEY_ENV, raising=False)

    assert pk.encryption_available() is False
    assert pk.workspace_env_values(org) == {}
    with pytest.raises(pk.ProviderKeyError):
        pk.encrypt_value("anything")


def test_a_malformed_encryption_key_does_not_crash(org, monkeypatch):
    monkeypatch.setenv(pk.ENCRYPTION_KEY_ENV, "not-a-fernet-key")
    assert pk.encryption_available() is False
    assert pk.workspace_env_values(org) == {}


def test_available_providers_merges_workspace_keys_over_the_server(org, monkeypatch):
    monkeypatch.setattr(pk, "available_provider_names", lambda: ["openai"])
    _store(org, "sarvam", {"SARVAM_API_KEY": "k"})
    assert sorted(pk.available_providers_for_org(org)) == ["openai", "sarvam"]


def test_every_known_provider_has_at_least_one_variable():
    for provider in pk.PROVIDER_ENV_VARS:
        assert pk.provider_env_vars(provider)
        for env_var in pk.provider_env_vars(provider):
            assert pk.env_var_kind(env_var) in {pk.SECRET, pk.PLAIN, pk.FILE}


def test_credentials_that_are_json_but_not_an_object_are_rejected():
    with pytest.raises(pk.ProviderKeyError):
        pk.encrypted_rows_for(
            "google",
            {
                "GOOGLE_APPLICATION_CREDENTIALS": "[1, 2]",
                "GOOGLE_CLOUD_PROJECT_ID": "my-project",
            },
        )


def test_credentials_with_no_account_name_still_show_something():
    assert pk._display_for("GOOGLE_APPLICATION_CREDENTIALS", "{}") == (
        "Service account JSON saved"
    )
    assert pk._display_for("GOOGLE_APPLICATION_CREDENTIALS", "123") == (
        "Service account JSON saved"
    )


def test_a_very_short_secret_is_still_fully_masked():
    assert pk._display_for("SARVAM_API_KEY", "ab") == "••••"


def test_a_credentials_file_that_cannot_be_written_drops_the_whole_provider(
    org, monkeypatch, tmp_path
):
    """Never pair the server's service account with the workspace's project.

    Keeping the project while the credentials fall back to the server produces
    a permission failure nobody can diagnose, so the provider moves as a unit.
    """
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/server/creds.json")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT_ID", "server-project")
    _store(
        org,
        "google",
        {
            "GOOGLE_APPLICATION_CREDENTIALS": SERVICE_ACCOUNT,
            "GOOGLE_CLOUD_PROJECT_ID": "my-project",
        },
    )
    _store(org, "sarvam", {"SARVAM_API_KEY": "workspace-sarvam"})

    def _refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _refuse)
    env = pk.provider_env(org, tmp_path)

    assert env["GOOGLE_APPLICATION_CREDENTIALS"] == "/server/creds.json"
    assert env["GOOGLE_CLOUD_PROJECT_ID"] == "server-project"
    # A failure for one provider must not disturb another.
    assert env["SARVAM_API_KEY"] == "workspace-sarvam"


def test_an_unreadable_value_does_not_half_apply_its_provider(org, monkeypatch):
    """Google needs both values, so one unreadable row drops both."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/server/creds.json")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT_ID", "server-project")
    _store(
        org,
        "google",
        {
            "GOOGLE_APPLICATION_CREDENTIALS": SERVICE_ACCOUNT,
            "GOOGLE_CLOUD_PROJECT_ID": "my-project",
        },
    )
    _store(org, "sarvam", {"SARVAM_API_KEY": "workspace-sarvam"})

    readable = [
        r
        for r in db.get_org_provider_keys(org)
        if r["env_var"] != "GOOGLE_APPLICATION_CREDENTIALS"
    ]
    with patch("provider_keys.get_org_provider_keys", return_value=readable):
        env = pk.provider_env(org, "/tmp")

    assert env["GOOGLE_CLOUD_PROJECT_ID"] == "server-project"
    assert env["GOOGLE_APPLICATION_CREDENTIALS"] == "/server/creds.json"
    assert env["SARVAM_API_KEY"] == "workspace-sarvam"


def test_reporting_an_unreadable_key_does_not_block_every_row(org, monkeypatch):
    """Sentry's flush blocks, and this runs per row on request paths."""
    import provider_keys

    for provider in ("sarvam", "openai", "deepgram"):
        _store(
            org, provider, {pk.provider_env_vars(provider)[0]: f"k-{provider}-1234"}
        )
    monkeypatch.setenv(pk.ENCRYPTION_KEY_ENV, Fernet.generate_key().decode())
    monkeypatch.setattr(provider_keys, "_decrypt_failure_reported", False)

    calls = []
    monkeypatch.setattr(
        provider_keys, "capture_exception_to_sentry", lambda exc: calls.append(exc)
    )
    pk.workspace_env_values(org)
    pk.workspace_env_values(org)

    assert len(calls) == 1


def test_a_service_account_with_an_odd_account_field_still_saves():
    """A non-text client_email must not reach the database as an object."""
    rows = pk.encrypted_rows_for(
        "google",
        {
            "GOOGLE_APPLICATION_CREDENTIALS": json.dumps({"client_email": {}}),
            "GOOGLE_CLOUD_PROJECT_ID": "my-project",
        },
    )
    display = {r["env_var"]: r["display"] for r in rows}
    assert display["GOOGLE_APPLICATION_CREDENTIALS"] == "Service account JSON saved"
    assert all(isinstance(r["display"], str) for r in rows)


def test_a_key_pasted_into_the_project_field_is_refused():
    with pytest.raises(pk.ProviderKeyError) as exc:
        pk.encrypted_rows_for(
            "google",
            {
                "GOOGLE_APPLICATION_CREDENTIALS": SERVICE_ACCOUNT,
                "GOOGLE_CLOUD_PROJECT_ID": "-----BEGIN PRIVATE KEY-----\nMIIE",
            },
        )
    assert "project id" in str(exc.value)


def test_real_project_ids_are_accepted():
    for project_id in ("my-project", "my-project-123", "abc123", "a" + "b" * 28 + "c"):
        rows = pk.encrypted_rows_for(
            "google",
            {
                "GOOGLE_APPLICATION_CREDENTIALS": SERVICE_ACCOUNT,
                "GOOGLE_CLOUD_PROJECT_ID": project_id,
            },
        )
        assert len(rows) == 2


def test_a_value_too_long_for_the_environment_is_refused():
    with pytest.raises(pk.ProviderKeyError) as exc:
        pk.encrypted_rows_for(
            "sarvam", {"SARVAM_API_KEY": "x" * (pk.MAX_VALUE_BYTES + 1)}
        )
    assert "too long" in str(exc.value)
