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


def test_send_email_posts_to_resend(monkeypatch, inline_thread):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv("EMAIL_FROM", "Calibrate <hello@example.com>")
    calls = []
    response = _FakeResponse()

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(httpx, "post", fake_post)

    mailer.send_email("dev@example.com", "Hi", "<p>Hi</p>")

    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == "https://api.resend.com/emails"
    assert kwargs["headers"] == {"Authorization": "Bearer re_test_key"}
    assert kwargs["json"] == {
        "from": "Calibrate <hello@example.com>",
        "to": ["dev@example.com"],
        "subject": "Hi",
        "html": "<p>Hi</p>",
    }
    assert kwargs["timeout"] == 10
    assert response.raised


def test_send_email_without_api_key_sends_nothing(monkeypatch, inline_thread):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: pytest.fail("should not post without a key")
    )

    mailer.send_email("dev@example.com", "Hi", "<p>Hi</p>")


def test_send_email_reports_failure_to_sentry(monkeypatch, inline_thread):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.delenv("EMAIL_FROM", raising=False)
    captured = []

    def boom(*args, **kwargs):
        raise httpx.ConnectError("provider down")

    monkeypatch.setattr(httpx, "post", boom)
    monkeypatch.setattr(mailer, "capture_exception_to_sentry", captured.append)

    mailer.send_email("dev@example.com", "Hi", "<p>Hi</p>")

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

    mailer.send_email("dev@example.com", "Hi", "<p>Hi</p>")

    assert sent.wait(timeout=5)
    assert sending_threads[0].startswith("mailer")
    assert sending_threads[0] != threading.current_thread().name


def test_sender_pool_is_bounded(monkeypatch):
    """An unauthenticated endpoint cannot spawn threads without limit."""
    assert mailer._SENDER._max_workers == 4


def test_frontend_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("FRONTEND_URL", "https://app.example.com/")
    assert mailer.frontend_url() == "https://app.example.com"


def test_frontend_url_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("FRONTEND_URL", raising=False)
    assert mailer.frontend_url() == "http://localhost:3000"
