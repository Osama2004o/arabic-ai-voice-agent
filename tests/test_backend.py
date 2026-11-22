"""Offline tests for the FastAPI backend, Email/Telegram integrations, and agent wiring.

No network or real credentials are used: HTTP calls go through httpx.MockTransport
and SMTP delivery is replaced with a capture function.
"""

from __future__ import annotations

import asyncio
import json
import smtplib
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from app.agent import ACTION_LOG, clear_action_log, register_email_handler, register_telegram_handler
from app.agent.graph import confirmation_gate_node
from app.agent.tools import send_email as agent_send_email
from app.core.config import Settings, override_settings
from app.core.security import notify_limiter
from app.integrations import email_client, register_agent_integrations, telegram_client
from app.main import app

API_KEY = "test-key"
TOKEN = "123456:SECRET-TOKEN"

BASE_SETTINGS = Settings(
    app_env="development",
    api_key=API_KEY,
    notify_rate_limit_per_min=50,
    email_provider="smtp",
    email_from="bot@example.com",
    smtp_host="smtp.example.com",
    smtp_port=587,
    smtp_username="bot@example.com",
    smtp_password="pw",
    telegram_bot_token=TOKEN,
    telegram_chat_id="999",
)

AUTH = {"X-API-Key": API_KEY}


@pytest.fixture(autouse=True)
def _isolate():
    override_settings(BASE_SETTINGS)
    notify_limiter.reset()
    clear_action_log()
    yield
    override_settings(None)
    register_email_handler(None)
    register_telegram_handler(None)
    telegram_client._http_transport = None
    email_client._http_transport = None


@pytest.fixture
def client():
    # Not used as a context manager: skips the lifespan (TTS warm-up needs network).
    return TestClient(app)


@pytest.fixture
def sent_mail(monkeypatch):
    captured = []
    monkeypatch.setattr(email_client, "_smtp_send", lambda msg, settings: captured.append(msg))
    return captured


def _telegram_transport(captured: list, status: int = 200, body: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status, json=body or {"ok": True, "result": {"message_id": 42}})

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------- health & chat

def test_health_reports_integrations(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["integrations"] == {"email": True, "telegram": True}


def test_health_reports_disabled_integrations(client):
    override_settings(Settings())
    assert client.get("/health").json()["integrations"] == {"email": False, "telegram": False}


def test_chat_text_rejects_empty(client):
    assert client.post("/chat/text", json={"text": ""}).status_code == 422
    assert client.post("/chat/text", json={"text": "   "}).status_code == 400


def test_chat_voice_rejects_unknown_voice(client):
    files = {"audio": ("a.webm", b"x" * 3000, "audio/webm")}
    res = client.post("/chat/voice", files=files, data={"voice": "robot"})
    assert res.status_code == 422


# ---------------------------------------------------------------- security

def test_notify_requires_api_key(client):
    body = {"message": "اختبار"}
    assert client.post("/send-telegram", json=body).status_code == 401
    assert client.post("/send-telegram", json=body, headers={"X-API-Key": "wrong"}).status_code == 401


def test_production_without_api_key_refuses(client):
    override_settings(replace(BASE_SETTINGS, app_env="production", api_key=""))
    assert client.post("/send-telegram", json={"message": "x"}).status_code == 503


def test_rate_limit(client):
    override_settings(replace(BASE_SETTINGS, notify_rate_limit_per_min=2))
    telegram_client._http_transport = _telegram_transport([])
    codes = [client.post("/send-telegram", json={"message": "x"}, headers=AUTH).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


# ---------------------------------------------------------------- telegram

def test_send_telegram_success(client):
    captured: list = []
    telegram_client._http_transport = _telegram_transport(captured)
    res = client.post("/send-telegram", json={"message": "شكوى جديدة من عميل"}, headers=AUTH)
    assert res.status_code == 200
    assert res.json() == {"status": "sent", "provider": "telegram", "message_id": 42}
    sent = json.loads(captured[0].content)
    assert sent == {"chat_id": "999", "text": "شكوى جديدة من عميل"}
    assert captured[0].url.path == f"/bot{TOKEN}/sendMessage"


def test_send_telegram_api_error_hides_token(client):
    telegram_client._http_transport = _telegram_transport(
        [], status=400, body={"ok": False, "description": "Bad Request: chat not found"}
    )
    res = client.post("/send-telegram", json={"message": "x"}, headers=AUTH)
    assert res.status_code == 502
    assert "chat not found" in res.json()["detail"]
    assert TOKEN not in res.text


def test_send_telegram_network_error_hides_token(client):
    def boom(request):
        raise httpx.ConnectError(f"cannot reach {request.url}")

    telegram_client._http_transport = httpx.MockTransport(boom)
    res = client.post("/send-telegram", json={"message": "x"}, headers=AUTH)
    assert res.status_code == 502
    assert TOKEN not in res.text


def test_send_telegram_not_configured(client):
    override_settings(replace(BASE_SETTINGS, telegram_bot_token=""))
    assert client.post("/send-telegram", json={"message": "x"}, headers=AUTH).status_code == 503


# ---------------------------------------------------------------- email

def test_send_email_smtp_success(client, sent_mail):
    payload = {"to": "customer@gmail.com", "subject": "ملخص الشكوى", "body": "تم تسجيل شكواك رقم 123."}
    res = client.post("/send-email", json=payload, headers=AUTH)
    assert res.status_code == 200
    assert res.json()["provider"] == "smtp"
    msg = sent_mail[0]
    assert msg["To"] == "customer@gmail.com"
    assert msg["Subject"] == "ملخص الشكوى"
    assert "تم تسجيل شكواك" in msg.get_body(("plain",)).get_content()
    assert 'dir="rtl"' in msg.get_body(("html",)).get_content()


def test_send_email_invalid_address(client, sent_mail):
    res = client.post("/send-email", json={"to": "not-an-email", "body": "x"}, headers=AUTH)
    assert res.status_code == 422
    assert sent_mail == []


def test_send_email_domain_allowlist(client, sent_mail):
    override_settings(replace(BASE_SETTINGS, email_allowed_domains=["gmail.com"]))
    assert client.post("/send-email", json={"to": "a@evil.com", "body": "x"}, headers=AUTH).status_code == 403
    assert client.post("/send-email", json={"to": "a@gmail.com", "body": "x"}, headers=AUTH).status_code == 200


def test_send_email_smtp_auth_failure(client, monkeypatch):
    def fail(msg, settings):
        raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

    monkeypatch.setattr(email_client, "_smtp_send", fail)
    res = client.post("/send-email", json={"to": "a@gmail.com", "body": "x"}, headers=AUTH)
    assert res.status_code == 502
    assert "authentication" in res.json()["detail"].lower()


def test_send_email_via_resend(client):
    override_settings(replace(BASE_SETTINGS, email_provider="resend", resend_api_key="re_test"))
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json={"id": "email_1"})

    email_client._http_transport = httpx.MockTransport(handler)
    res = client.post("/send-email", json={"to": "a@gmail.com", "body": "مرحبا"}, headers=AUTH)
    assert res.status_code == 200
    assert res.json() == {"status": "sent", "provider": "resend", "message_id": "email_1"}
    assert captured[0].headers["Authorization"] == "Bearer re_test"
    assert json.loads(captured[0].content)["to"] == ["a@gmail.com"]


# ---------------------------------------------------------------- agent wiring

def test_register_agent_integrations_uses_live_senders(sent_mail):
    captured: list = []
    telegram_client._http_transport = _telegram_transport(captured)
    assert register_agent_integrations() == {"email": True, "telegram": True}

    email_result = asyncio.run(agent_send_email("a@gmail.com", "موضوع", "نص"))
    assert email_result["mode"] == "live"
    assert email_result["result"]["status"] == "sent"
    assert len(sent_mail) == 1


def test_register_agent_integrations_falls_back_to_simulated():
    override_settings(Settings())
    assert register_agent_integrations() == {"email": False, "telegram": False}
    result = asyncio.run(agent_send_email("a@gmail.com", "s", "b"))
    assert result["mode"] == "simulated"


def test_confirmed_send_failure_is_reported_to_customer():
    telegram_client._http_transport = _telegram_transport([], status=401, body={"ok": False, "description": "Unauthorized"})
    register_agent_integrations()
    state = {
        "messages": [HumanMessage(content="نعم")],
        "pending_action": {"action": "send_telegram", "args": {"message": "ملخص"}, "preview_ar": "؟"},
    }
    out = asyncio.run(confirmation_gate_node(state))
    assert "مشكلة" in out["final_response"]
    assert out["action_taken"] is None
    assert ACTION_LOG[-1]["mode"] == "failed"


def test_confirmed_send_success_is_reported_to_customer():
    telegram_client._http_transport = _telegram_transport([])
    register_agent_integrations()
    state = {
        "messages": [HumanMessage(content="نعم")],
        "pending_action": {"action": "send_telegram", "args": {"message": "ملخص"}, "preview_ar": "؟"},
    }
    out = asyncio.run(confirmation_gate_node(state))
    assert "بنجاح" in out["final_response"]
    assert out["action_taken"] == "send_telegram"
