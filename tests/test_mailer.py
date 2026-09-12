"""Tests for the Resend email seam. httpx.post is always patched — no network."""

import threading

import httpx
import pytest

import mailer


class _InlineSender:
    """Runs the work immediately on submit so the send is deterministic in tests."""

    def submit(self, fn):
        fn()


class _FakeResponse:
    def __init__(self):
        self.raised = False

    def raise_for_status(self):
        self.raised = True


@pytest.fixture
def inline_thread(monkeypatch):
    monkeypatch.setattr(mailer, "_SENDER", _InlineSender())


def test_send_email_posts_template_to_resend(monkeypatch, inline_thread):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv("EMAIL_FROM", "Calibrate <hello@example.com>")
    calls = []
    response = _FakeResponse()

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(httpx, "post", fake_post)

    mailer.send_email(
        "dev@example.com",
        mailer.WELCOME_TEMPLATE,
        {"NAME": "Ada", "URL": "https://app.example.com"},
    )

    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == "https://api.resend.com/emails"
    assert kwargs["headers"] == {"Authorization": "Bearer re_test_key"}
    assert kwargs["json"] == {
        "from": "Calibrate <hello@example.com>",
        "to": ["dev@example.com"],
        "template": {
            "id": "calibrate-welcome",
            "variables": {"NAME": "Ada", "URL": "https://app.example.com"},
        },
    }
    assert "html" not in kwargs["json"]
    assert kwargs["timeout"] == 10
    assert response.raised


def test_send_email_strips_angle_brackets_from_values(monkeypatch, inline_thread):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    calls = []
    monkeypatch.setattr(
        httpx, "post", lambda url, **kwargs: calls.append(kwargs) or _FakeResponse()
    )

    mailer.send_email(
        "dev@example.com",
        mailer.WORKSPACE_INVITE_TEMPLATE,
        {
            "INVITER": "<b>Ada</b>",
            "WORKSPACE": "R&D + Ops",
            "URL": "https://app.example.com/join?a=1&b=2",
        },
    )

    assert calls[0]["json"]["template"]["variables"] == {
        "INVITER": "bAda/b",
        "WORKSPACE": "R&D + Ops",
        "URL": "https://app.example.com/join?a=1&b=2",
    }


def test_template_aliases():
    assert mailer.WELCOME_TEMPLATE == "calibrate-welcome"
    assert mailer.WORKSPACE_INVITE_TEMPLATE == "calibrate-workspace-invite"


def test_send_email_without_api_key_sends_nothing(monkeypatch, inline_thread):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: pytest.fail("should not post without a key")
    )

    mailer.send_email("dev@example.com", mailer.WELCOME_TEMPLATE, {"NAME": "Ada"})


def test_send_email_reports_failure_to_sentry(monkeypatch, inline_thread):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.delenv("EMAIL_FROM", raising=False)
    captured = []

    def boom(*args, **kwargs):
        raise httpx.ConnectError("provider down")

    monkeypatch.setattr(httpx, "post", boom)
    monkeypatch.setattr(mailer, "capture_exception_to_sentry", captured.append)

    mailer.send_email("dev@example.com", mailer.WORKSPACE_INVITE_TEMPLATE, {"URL": "u"})

    assert len(captured) == 1
    assert isinstance(captured[0], httpx.ConnectError)


def test_send_email_runs_off_the_calling_thread(monkeypatch):
    """The real worker-pool path: the send happens on a mailer worker, not the caller."""
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    sending_threads = []
    sent = threading.Event()

    def record(*args, **kwargs):
        sending_threads.append(threading.current_thread().name)
        sent.set()
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", record)

    mailer.send_email("dev@example.com", mailer.WORKSPACE_INVITE_TEMPLATE, {"URL": "u"})

    assert sent.wait(timeout=5)
    assert sending_threads[0].startswith("mailer")
    assert sending_threads[0] != threading.current_thread().name


def test_sender_pool_is_bounded():
    """An unauthenticated endpoint cannot spawn threads without limit."""
    assert mailer._SENDER._max_workers == 4


def test_frontend_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("FRONTEND_URL", "https://app.example.com/")
    assert mailer.frontend_url() == "https://app.example.com"


def test_frontend_url_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("FRONTEND_URL", raising=False)
    assert mailer.frontend_url() == "http://localhost:3000"
