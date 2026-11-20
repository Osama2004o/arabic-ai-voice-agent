import base64
import logging
from typing import Awaitable, Callable, Dict, Any

from .config import UNCLEAR_SPEECH_REPLY, DEFAULT_VOICE
from .latency import LatencyTracker
from .stt import transcribe
from .tts import synthesize

logger = logging.getLogger("speech.pipeline")

# Signature agreed upon with Member 2 and Member 3
AgentFn = Callable[[str, str], Awaitable[str]]


async def run_voice_turn(
    audio_bytes: bytes,
    mime_type: str = "audio/webm",
    voice: str = DEFAULT_VOICE,
    session_id: str = "default_session",
    agent_fn: AgentFn = None,
    tts_provider: str = None,
) -> Dict[str, Any]:
    """
    Executes a complete end-to-end voice turn:
    1. STT: audio_bytes -> Arabic transcript (with hallucination/silence guard)
    2. LLM: transcript + session_id -> Arabic customer service reply
    3. TTS: reply -> Egyptian Arabic MP3 audio bytes
    4. Tracks fine-grained latency for each stage.
    """
    tracker = LatencyTracker()

    # Stage 1: STT
    with tracker.stage("stt"):
        tr = await transcribe(audio_bytes, mime_type=mime_type, language="ar")

    transcript = tr.text.strip() if tr.text else ""

    # Stage 2: Agent Response
    reply_text = ""
    if not transcript:
        # User spoke unintelligibly or was silent; use safe polite retry prompt
        tracker.set_stage("llm", 0.0)
        reply_text = UNCLEAR_SPEECH_REPLY
        logger.info("Empty or unintelligible speech detected. Bypassing LLM with retry prompt.")
    else:
        with tracker.stage("llm"):
            if agent_fn is not None:
                reply_text = await agent_fn(transcript, session_id)
            else:
                from .stub_agent import stub_agent
                reply_text = await stub_agent(transcript, session_id)

    # Stage 3: TTS Synthesis
    audio_mp3_bytes = b""
    with tracker.stage("tts"):
        audio_mp3_bytes = await synthesize(reply_text, voice=voice, provider=tts_provider)

    timings = tracker.finish()

    # Log turn summary without exposing audio or keys
    logger.info(
        f"[VOICE TURN] Trans: {len(transcript)} chars | Reply: {len(reply_text)} chars | "
        f"STT: {timings.get('stt')}ms | LLM: {timings.get('llm')}ms | TTS: {timings.get('tts')}ms | "
        f"Total: {timings.get('server_total')}ms"
    )

    audio_b64 = base64.b64encode(audio_mp3_bytes).decode("ascii") if audio_mp3_bytes else ""

    return {
        "transcript": transcript,
        "reply_text": reply_text,
        "audio_base64": audio_b64,
        "audio_mime": "audio/mpeg",
        "timings_ms": timings,
    }
