"""Conversation endpoints: text and full voice turns."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.agent import agent, arun_agent
from app.models.schemas import ChatTextRequest, ChatTextResponse, VoiceTurnResponse
from app.speech.config import DEFAULT_VOICE, MAX_AUDIO_BYTES, OPTIONAL_MSA_VOICES, VOICES
from app.speech.pipeline import run_voice_turn

router = APIRouter(prefix="/chat", tags=["chat"])

SUPPORTED_VOICES = {**VOICES, **OPTIONAL_MSA_VOICES}


@router.post("/text", response_model=ChatTextResponse)
async def chat_text(req: ChatTextRequest):
    """Text turn against the LangGraph customer-service agent."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    return await arun_agent(session_id=req.session_id, text=req.text)


@router.post("/voice", response_model=VoiceTurnResponse)
async def chat_voice(
    audio: UploadFile = File(..., description="Recorded speech (webm/ogg/wav/mp3)."),
    voice: str = Form(DEFAULT_VOICE),
    session_id: str = Form("default_session", max_length=128),
):
    """Full voice turn: STT -> agent -> TTS, with per-stage latency."""
    if voice not in SUPPORTED_VOICES:
        raise HTTPException(status_code=422, detail=f"Unknown voice. Use one of: {', '.join(SUPPORTED_VOICES)}.")
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds size limit.")
    return await run_voice_turn(
        audio_bytes=content,
        mime_type=audio.content_type or "audio/webm",
        voice=voice,
        session_id=session_id,
        agent_fn=agent.respond,
    )
