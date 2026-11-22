"""Health check used by Docker and the hosting platform."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.core.config import get_settings
from app.models.schemas import HealthResponse
from app.speech.config import STT_PROVIDER, TTS_PROVIDER

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    settings = get_settings()
    return {
        "status": "healthy",
        "service": "arabic-ai-voice-agent",
        "version": __version__,
        "environment": settings.app_env,
        "integrations": {"email": settings.email_enabled, "telegram": settings.telegram_enabled},
        "stt_provider": STT_PROVIDER,
        "tts_provider": TTS_PROVIDER,
    }
