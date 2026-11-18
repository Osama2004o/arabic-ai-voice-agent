"""Request/response models for the public API."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class ChatTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Customer message in Arabic.")
    session_id: str = Field("default_session", min_length=1, max_length=128)


class ChatTextResponse(BaseModel):
    response: str
    action: Optional[str] = None
    needs_confirmation: bool = False
    needs_human: bool = False
    session_id: str
    customer_name: Optional[str] = None
    order_id: Optional[str] = None
    latency_ms: float


class VoiceTurnResponse(BaseModel):
    transcript: str
    reply_text: str
    audio_base64: str
    audio_mime: str = "audio/mpeg"
    timings_ms: Dict[str, float]


class SendEmailRequest(BaseModel):
    to: str = Field(..., max_length=254, examples=["customer@example.com"])
    subject: str = Field("ملخص محادثة خدمة العملاء", max_length=200)
    body: str = Field(..., min_length=1, max_length=10_000)


class SendTelegramRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)


class NotificationResponse(BaseModel):
    status: str
    provider: str
    message_id: Optional[str | int] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    integrations: Dict[str, bool]
    stt_provider: str
    tts_provider: str
    latency_target_sec: float = 3.0
