"""Backend settings loaded from environment variables (.env in development)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _get_list(name: str) -> List[str]:
    return [item.strip().lower() for item in _get(name).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # --- App ---
    app_env: str = "development"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    # Shared secret required in the `X-API-Key` header for /send-email and /send-telegram.
    # Empty = endpoints are open (development only).
    api_key: str = ""
    # Max outbound notifications per client per minute (endpoints) and per channel (agent tools).
    notify_rate_limit_per_min: int = 5

    # --- Email ---
    email_provider: str = "none"  # smtp | resend | none
    email_from: str = ""
    email_from_name: str = "خدمة العملاء"
    email_allowed_domains: List[str] = field(default_factory=list)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_ssl: bool = False
    smtp_timeout_sec: int = 15
    resend_api_key: str = ""

    # --- Telegram ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @property
    def email_enabled(self) -> bool:
        if self.email_provider == "smtp":
            return bool(self.smtp_host and self.email_from)
        if self.email_provider == "resend":
            return bool(self.resend_api_key and self.email_from)
        return False

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


def load_settings() -> Settings:
    smtp_host = _get("SMTP_HOST")
    resend_key = _get("RESEND_API_KEY")
    provider = _get("EMAIL_PROVIDER").lower()
    if not provider:
        provider = "smtp" if smtp_host else "resend" if resend_key else "none"

    smtp_port = _get_int("SMTP_PORT", 587)
    smtp_username = _get("SMTP_USERNAME")
    cors = _get_list("CORS_ORIGINS") or ["*"]

    return Settings(
        app_env=_get("APP_ENV", "development").lower(),
        cors_origins=cors,
        api_key=_get("API_KEY"),
        notify_rate_limit_per_min=max(1, _get_int("NOTIFY_RATE_LIMIT_PER_MIN", 5)),
        email_provider=provider,
        email_from=_get("EMAIL_FROM") or smtp_username,
        email_from_name=_get("EMAIL_FROM_NAME", "خدمة العملاء"),
        email_allowed_domains=_get_list("EMAIL_ALLOWED_DOMAINS"),
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_username=smtp_username,
        smtp_password=_get("SMTP_PASSWORD"),
        smtp_use_ssl=_get_bool("SMTP_USE_SSL", smtp_port == 465),
        smtp_timeout_sec=_get_int("SMTP_TIMEOUT_SEC", 15),
        resend_api_key=resend_key,
        telegram_bot_token=_get("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_get("TELEGRAM_CHAT_ID"),
    )


_override: Optional[Settings] = None


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return load_settings()


def get_settings() -> Settings:
    return _override if _override is not None else _cached_settings()


def override_settings(settings: Optional[Settings]) -> None:
    """Force specific settings (tests). Pass None to go back to environment-based settings."""
    global _override
    _override = settings


def reload_settings() -> Settings:
    """Re-read environment variables."""
    _cached_settings.cache_clear()
    return get_settings()


def mask(value: Optional[str], keep: int = 3) -> str:
    """Mask a secret or address for logs, e.g. 'abc***'."""
    if not value:
        return ""
    return value[:keep] + "***"
