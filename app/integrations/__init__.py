"""External integrations (Email, Telegram) and their wiring into the AI agent's tools."""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.core.config import get_settings
from app.core.security import notify_limiter

from .email_client import send_email
from .errors import IntegrationError
from .telegram_client import send_telegram

logger = logging.getLogger("integrations")


async def _agent_email_handler(to: str, subject: str, body: str) -> Dict[str, Any]:
    if not notify_limiter.allow("agent:email", get_settings().notify_rate_limit_per_min):
        raise IntegrationError("Rate limit reached for emails.", status_code=429)
    return await send_email(to, subject, body)


async def _agent_telegram_handler(message: str) -> Dict[str, Any]:
    if not notify_limiter.allow("agent:telegram", get_settings().notify_rate_limit_per_min):
        raise IntegrationError("Rate limit reached for Telegram messages.", status_code=429)
    return await send_telegram(message)


def register_agent_integrations() -> Dict[str, bool]:
    """Attach real senders to the agent's confirmed-action tools.

    A channel that is not configured stays in the agent's "simulated" mode (logged only),
    so local development works without credentials.
    """
    from app.agent import register_email_handler, register_telegram_handler

    settings = get_settings()
    register_email_handler(_agent_email_handler if settings.email_enabled else None)
    register_telegram_handler(_agent_telegram_handler if settings.telegram_enabled else None)

    status = {"email": settings.email_enabled, "telegram": settings.telegram_enabled}
    logger.info(
        "Integrations: email=%s (%s), telegram=%s",
        "live" if status["email"] else "simulated",
        settings.email_provider,
        "live" if status["telegram"] else "simulated",
    )
    return status


__all__ = ["IntegrationError", "register_agent_integrations", "send_email", "send_telegram"]
