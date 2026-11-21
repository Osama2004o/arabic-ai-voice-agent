import logging
from typing import Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Response
from pydantic import BaseModel

from .config import (
    VOICES,
    OPTIONAL_MSA_VOICES,
    DEFAULT_VOICE,
    TTS_PROVIDER,
    ELEVENLABS_VOICES,
    MIN_AUDIO_BYTES,
    MAX_AUDIO_BYTES,
)
from .stt import transcribe
from .tts import synthesize
from .pipeline import run_voice_turn
from .stub_agent import stub_agent

logger = logging.getLogger("speech.router")

router = APIRouter(prefix="/speech", tags=["speech"])


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = DEFAULT_VOICE
    provider: Optional[str] = None


@router.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Transcribes uploaded audio file to Arabic text using Groq Whisper / local fallback.
    """
    if not audio:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")

    content = await audio.read()
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio payload too large ({len(content)} bytes exceeds {MAX_AUDIO_BYTES} limit)."
        )

    mime_type = audio.content_type or "audio/webm"
    result = await transcribe(content, mime_type=mime_type, language="ar")

    return {
        "text": result.text,
        "language": result.language,
        "stt_ms": result.stt_ms,
        "provider": result.provider,
        "no_speech_prob": result.no_speech_prob,
    }


@router.post("/tts")
async def text_to_speech(req: TTSRequest):
    """
    Synthesizes Arabic text to MP3 audio using Edge-TTS.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty.")

    audio_bytes = await synthesize(req.text, voice=req.voice or DEFAULT_VOICE, provider=req.provider)
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="TTS synthesis failed to produce audio.")

    media_type = "audio/wav" if audio_bytes.startswith(b"RIFF") else "audio/mpeg"
    return Response(content=audio_bytes, media_type=media_type)


@router.get("/voices")
async def get_available_voices(provider: Optional[str] = None):
    """
    Returns available Arabic voices for male and female selection.
    """
    active_provider = (provider or TTS_PROVIDER).lower().strip()
    is_eleven = active_provider == "elevenlabs"
    is_hf = active_provider in ("huggingface", "hf")

    if is_hf:
        voices = [
            {
                "id": "female",
                "label": "أنثوي — فصحى (TinyVoca)",
                "voice_id": "tiny-voca-arabic-female",
                "dialect": "Modern Standard Arabic",
                "gender": "Female",
            },
            {
                "id": "male",
                "label": "ذكوري — فصحى (TinyVoca)",
                "voice_id": "tiny-voca-arabic-male",
                "dialect": "Modern Standard Arabic",
                "gender": "Male",
            },
        ]
    elif is_eleven:
        voices = [
            {
                "id": "female",
                "label": "أنثوي — سارة (إيلفن لابس)",
                "voice_id": ELEVENLABS_VOICES["female"],
                "dialect": "Arabic",
                "gender": "Female",
            },
            {
                "id": "male",
                "label": "ذكوري — آدم (إيلفن لابس)",
                "voice_id": ELEVENLABS_VOICES["male"],
                "dialect": "Arabic",
                "gender": "Male",
            },
            {
                "id": "female_msa",
                "label": "أنثوي — سارة (فصحى)",
                "voice_id": ELEVENLABS_VOICES["female_msa"],
                "dialect": "Modern Standard Arabic",
                "gender": "Female",
            },
            {
                "id": "male_msa",
                "label": "ذكوري — آدم (فصحى)",
                "voice_id": ELEVENLABS_VOICES["male_msa"],
                "dialect": "Modern Standard Arabic",
                "gender": "Male",
            },
        ]
    else:
        voices = [
            {
                "id": "female",
                "label": "أنثوي — سلمى (لهجة مصرية)",
                "voice_id": VOICES["female"],
                "dialect": "Egyptian Arabic",
                "gender": "Female",
            },
            {
                "id": "male",
                "label": "ذكوري — شاكر (لهجة مصرية)",
                "voice_id": VOICES["male"],
                "dialect": "Egyptian Arabic",
                "gender": "Male",
            },
            {
                "id": "female_msa",
                "label": "أنثوي — زارية (فصحى)",
                "voice_id": OPTIONAL_MSA_VOICES["female_msa"],
                "dialect": "Modern Standard Arabic",
                "gender": "Female",
            },
            {
                "id": "male_msa",
                "label": "ذكوري — حامد (فصحى)",
                "voice_id": OPTIONAL_MSA_VOICES["male_msa"],
                "dialect": "Modern Standard Arabic",
                "gender": "Male",
            },
        ]

    return {
        "provider": active_provider,
        "default": DEFAULT_VOICE,
        "providers": ["elevenlabs", "edge", "huggingface"],
        "voices": voices,
    }


@router.post("/demo-turn")
async def demo_voice_turn(
    audio: UploadFile = File(...),
    voice: str = Form(DEFAULT_VOICE),
    session_id: str = Form("demo_session"),
    tts_provider: Optional[str] = Form(None),
):
    """
    Executes a complete end-to-end voice turn with the stub LLM agent.
    Returns transcript, reply text, base64 audio, and per-stage latency timings.
    """
    if not audio:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    content = await audio.read()
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds size limit.")

    mime_type = audio.content_type or "audio/webm"

    turn_result = await run_voice_turn(
        audio_bytes=content,
        mime_type=mime_type,
        voice=voice,
        session_id=session_id,
        agent_fn=stub_agent,
        tts_provider=tts_provider,
    )

    return turn_result
