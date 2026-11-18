"""Telegram Bot API integration (sendMessage).

Setup: create a bot with @BotFather to get TELEGRAM_BOT_TOKEN, send the bot any message,
then open https://api.telegram.org/bot<TOKEN>/getUpdates to read your TELEGRAM_CHAT_ID.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from app.core.config import Settings, get_settings

from .errors import IntegrationError

logger = logging.getLogger("integrations.telegram")

TELEGRAM_API = "https://api.telegram.org"
MAX_MESSAGE_LEN = 4096

# Overridable in tests (httpx.MockTransport).
_http_transport: Optional[httpx.AsyncBaseTransport] = None


async def send_telegram(message: str, chat_id: Optional[str] = None, settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Send a text message to the configured chat. Raises IntegrationError on failure."""
    settings = settings or get_settings()
    if not settings.telegram_enabled:
        raise IntegrationError("Telegram integration is not configured.", status_code=503)

    text = (message or "").strip()
    if not text:
        raise IntegrationError("Telegram message cannot be empty.", status_code=422)
    if len(text) > MAX_MESSAGE_LEN:
        text = text[: MAX_MESSAGE_LEN - 1] + "…"

    url = f"{TELEGRAM_API}/bot{settings.telegram_bot_token}/sendMessage"
    payload = {"chat_id": chat_id or settings.telegram_chat_id, "text": text}

    # NOTE: never include str(exc) in errors/logs — httpx messages contain the URL, which holds the bot token.
    try:
        async with httpx.AsyncClient(transport=_http_transport, timeout=10.0) as client:
            resp = await client.post(url, json=payload)
    except httpx.HTTPError as exc:
        raise IntegrationError(f"Telegram request failed ({type(exc).__name__}).") from None

    try:
        data = resp.json()
    except ValueError:
        data = {}

    if resp.status_code >= 400 or not data.get("ok"):
        description = str(data.get("description") or "unknown error")[:200]
        raise IntegrationError(f"Telegram API error (HTTP {resp.status_code}): {description}")

    message_id = (data.get("result") or {}).get("message_id")
    logger.info("Telegram message sent (message_id=%s)", message_id)
    return {"status": "sent", "provider": "telegram", "message_id": message_id}
