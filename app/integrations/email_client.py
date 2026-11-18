"""Email delivery via SMTP (e.g. Gmail app password) or the Resend HTTP API.

Many free hosting tiers block outbound SMTP ports, so Resend (HTTPS) is supported
as an alternative. Pick one with EMAIL_PROVIDER=smtp|resend.
"""

from __future__ import annotations

import asyncio
import logging
import re
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from typing import Any, Dict, Optional

import httpx

from app.core.config import Settings, get_settings, mask

from .errors import IntegrationError

logger = logging.getLogger("integrations.email")

RESEND_URL = "https://api.resend.com/emails"
EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

# Overridable in tests (httpx.MockTransport).
_http_transport: Optional[httpx.AsyncBaseTransport] = None


def validate_recipient(to: str, settings: Settings) -> str:
    to = (to or "").strip()
    if not EMAIL_REGEX.match(to):
        raise IntegrationError("Invalid recipient email address.", status_code=422)
    domain = to.rsplit("@", 1)[1].lower()
    if settings.email_allowed_domains and domain not in settings.email_allowed_domains:
        raise IntegrationError("Recipient domain is not allowed.", status_code=403)
    return to


def _build_message(to: str, subject: str, body: str, settings: Settings) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = formataddr((settings.email_from_name, settings.email_from))
    msg["To"] = to
    msg["Subject"] = subject
    msg["Message-ID"] = make_msgid()
    msg.set_content(body, charset="utf-8")
    # Simple RTL HTML alternative so Arabic renders right-to-left in mail clients.
    html_body = "<br>".join(_escape_html(line) for line in body.splitlines())
    msg.add_alternative(
        f'<div dir="rtl" style="font-family:Tahoma,Arial,sans-serif;font-size:15px">{html_body}</div>',
        subtype="html",
        charset="utf-8",
    )
    return msg


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _smtp_send(msg: EmailMessage, settings: Settings) -> None:
    context = ssl.create_default_context()
    if settings.smtp_use_ssl:
        server: smtplib.SMTP = smtplib.SMTP_SSL(
            settings.smtp_host, settings.smtp_port, timeout=settings.smtp_timeout_sec, context=context
        )
    else:
        server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=settings.smtp_timeout_sec)
    with server:
        if not settings.smtp_use_ssl:
            server.ehlo()
            if server.has_extn("starttls"):
                server.starttls(context=context)
                server.ehlo()
        if settings.smtp_username:
            server.login(settings.smtp_username, settings.smtp_password)
        server.send_message(msg)


async def _send_via_smtp(to: str, subject: str, body: str, settings: Settings) -> Dict[str, Any]:
    msg = _build_message(to, subject, body, settings)
    try:
        await asyncio.to_thread(_smtp_send, msg, settings)
    except smtplib.SMTPAuthenticationError as exc:
        raise IntegrationError("SMTP authentication failed. Check SMTP_USERNAME / SMTP_PASSWORD.") from exc
    except smtplib.SMTPRecipientsRefused as exc:
        raise IntegrationError("The mail server refused the recipient.", status_code=422) from exc
    except (smtplib.SMTPException, OSError) as exc:
        raise IntegrationError(f"SMTP delivery failed ({type(exc).__name__}).") from exc
    return {"status": "sent", "provider": "smtp", "message_id": msg["Message-ID"]}


async def _send_via_resend(to: str, subject: str, body: str, settings: Settings) -> Dict[str, Any]:
    payload = {
        "from": formataddr((settings.email_from_name, settings.email_from)),
        "to": [to],
        "subject": subject,
        "text": body,
    }
    headers = {"Authorization": f"Bearer {settings.resend_api_key}"}
    try:
        async with httpx.AsyncClient(transport=_http_transport, timeout=15.0) as client:
            resp = await client.post(RESEND_URL, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise IntegrationError(f"Email API request failed ({type(exc).__name__}).") from exc
    if resp.status_code >= 400:
        raise IntegrationError(f"Email API rejected the request (HTTP {resp.status_code}).")
    return {"status": "sent", "provider": "resend", "message_id": resp.json().get("id")}


async def send_email(to: str, subject: str, body: str, settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Send a plain-text (plus RTL HTML) email. Raises IntegrationError on failure."""
    settings = settings or get_settings()
    if not settings.email_enabled:
        raise IntegrationError("Email integration is not configured.", status_code=503)

    to = validate_recipient(to, settings)
    subject = (subject or "").strip() or "ملخص محادثة خدمة العملاء"
    body = (body or "").strip()
    if not body:
        raise IntegrationError("Email body cannot be empty.", status_code=422)

    if settings.email_provider == "resend":
        result = await _send_via_resend(to, subject, body, settings)
    else:
        result = await _send_via_smtp(to, subject, body, settings)

    logger.info("Email sent via %s to %s", result["provider"], mask(to))
    return result
