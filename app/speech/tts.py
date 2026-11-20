import asyncio
import logging
from typing import AsyncIterator, Dict, Tuple, Literal, Optional
import edge_tts
import httpx

from .config import (
    TTS_PROVIDER,
    VOICES,
    OPTIONAL_MSA_VOICES,
    DEFAULT_VOICE,
    ELEVENLABS_API_KEY,
    ELEVENLABS_MODEL_ID,
    ELEVENLABS_DEFAULT_VOICE_ID,
    ELEVENLABS_VOICES,
    UNCLEAR_SPEECH_REPLY,
    GREETING_REPLY,
)
from .text_utils import clean_text_for_tts

logger = logging.getLogger("speech.tts")

VoiceType = Literal["female", "male"]

# In-memory cache for common/fixed phrases: (provider, voice, clean_text) -> bytes
_TTS_CACHE: Dict[Tuple[str, str, str], bytes] = {}
_MAX_CACHE_ENTRIES = 128

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1/text-to-speech"


def _resolve_edge_voice_id(voice: str) -> str:
    cleaned = (voice or "").lower().strip()
    if cleaned in VOICES:
        return VOICES[cleaned]
    if cleaned in OPTIONAL_MSA_VOICES:
        return OPTIONAL_MSA_VOICES[cleaned]
    if "neural" in cleaned:
        return voice
    return VOICES.get(DEFAULT_VOICE, VOICES["female"])


def _resolve_elevenlabs_voice_id(voice: str) -> str:
    """
    Resolve requested voice alias or direct voice ID to an ElevenLabs voice ID.
    Uses EXAVITQu4vr4xnSDxMaL (Sarah) if female is selected; otherwise defaults to pNInz6obpgDQGcFmaJgB (Adam).
    """
    cleaned = (voice or "").strip()
    if len(cleaned) >= 20 and not cleaned.startswith("ar-"):
        return cleaned
    cleaned_lower = cleaned.lower()
    if "female" in cleaned_lower:
        return ELEVENLABS_VOICES.get("female", "EXAVITQu4vr4xnSDxMaL")
    return ELEVENLABS_VOICES.get(cleaned_lower, ELEVENLABS_DEFAULT_VOICE_ID)


async def _synthesize_elevenlabs(text: str, voice: str) -> Optional[bytes]:
    """
    Synthesize Arabic audio using ElevenLabs REST API.
    Returns raw MP3 bytes on success, or None on failure / quota exhaustion (enabling fallback).
    """
    if not ELEVENLABS_API_KEY:
        logger.warning("ELEVENLABS_API_KEY is not configured; cannot use ElevenLabs TTS.")
        return None

    voice_id = _resolve_elevenlabs_voice_id(voice)
    url = f"{ELEVENLABS_API_BASE}/{voice_id}"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(url, json=payload, headers=headers)

        if resp.status_code == 200 and resp.content:
            return resp.content

        logger.warning(
            "ElevenLabs TTS failed HTTP %d: %s",
            resp.status_code,
            resp.text[:200],
        )
        return None
    except Exception as exc:
        logger.warning("ElevenLabs TTS exception: %s", exc)
        return None


async def _stream_elevenlabs(text: str, voice: str) -> AsyncIterator[bytes]:
    """
    Stream MP3 audio chunks in real-time from ElevenLabs.
    """
    if not ELEVENLABS_API_KEY:
        logger.warning("ELEVENLABS_API_KEY is not configured; cannot stream from ElevenLabs.")
        return

    voice_id = _resolve_elevenlabs_voice_id(voice)
    url = f"{ELEVENLABS_API_BASE}/{voice_id}/stream"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code == 200:
                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        if chunk:
                            yield chunk
                else:
                    err_bytes = await response.aread()
                    logger.warning(
                        "ElevenLabs stream HTTP %d: %s",
                        response.status_code,
                        err_bytes.decode("utf-8", errors="ignore")[:200],
                    )
    except Exception as exc:
        logger.warning("ElevenLabs stream exception: %s", exc)


async def _synthesize_edge_stream(text: str, voice_id: str) -> AsyncIterator[bytes]:
    """
    Streams MP3 audio chunks generated via Edge-TTS with 1 retry on network glitch.
    """
    attempts = 2
    for attempt in range(1, attempts + 1):
        try:
            communicate = edge_tts.Communicate(text, voice_id)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
            return
        except Exception as e:
            if attempt < attempts:
                logger.warning(f"Edge-TTS stream attempt {attempt} failed ({e}). Retrying...")
                await asyncio.sleep(0.2)
            else:
                logger.error(f"Edge-TTS stream failed after {attempts} attempts: {e}")
                return


async def synthesize_stream(text: str, voice: str = "female", provider: Optional[str] = None) -> AsyncIterator[bytes]:
    """
    Streams MP3 audio chunks using the configured or explicitly requested TTS provider.
    Falls back gracefully to Edge-TTS if ElevenLabs fails, is out of quota, or has no key.
    """
    cleaned = clean_text_for_tts(text)
    if not cleaned:
        return

    active_provider = (provider or TTS_PROVIDER).lower().strip()
    voice_id = _resolve_edge_voice_id(voice)
    cache_key = (active_provider, voice, cleaned)

    if cache_key in _TTS_CACHE:
        yield _TTS_CACHE[cache_key]
        return

    if active_provider == "elevenlabs":
        got_chunks = False
        chunks = []
        try:
            async for chunk in _stream_elevenlabs(cleaned, voice):
                got_chunks = True
                chunks.append(chunk)
                yield chunk
        except Exception as e:
            logger.warning("ElevenLabs streaming error: %s", e)

        if got_chunks:
            if chunks and len(_TTS_CACHE) < _MAX_CACHE_ENTRIES:
                _TTS_CACHE[cache_key] = b"".join(chunks)
            return

        logger.warning("ElevenLabs streaming produced no audio. Falling back to Edge-TTS.")

    # Edge-TTS (primary or fallback)
    chunks = []
    async for chunk in _synthesize_edge_stream(cleaned, voice_id):
        chunks.append(chunk)
        yield chunk

    if chunks and len(_TTS_CACHE) < _MAX_CACHE_ENTRIES:
        _TTS_CACHE[cache_key] = b"".join(chunks)


async def synthesize(text: str, voice: str = "female", provider: Optional[str] = None) -> bytes:
    """
    Synthesizes complete MP3 audio bytes for text using the configured or explicitly requested TTS provider.
    Uses in-memory caching and automatic fallback to Edge-TTS on error or quota exhaustion.
    """
    cleaned = clean_text_for_tts(text)
    if not cleaned:
        return b""

    active_provider = (provider or TTS_PROVIDER).lower().strip()
    cache_key = (active_provider, voice, cleaned)

    if cache_key in _TTS_CACHE:
        return _TTS_CACHE[cache_key]

    if active_provider == "elevenlabs":
        audio = await _synthesize_elevenlabs(cleaned, voice)
        if audio:
            if len(_TTS_CACHE) < _MAX_CACHE_ENTRIES:
                _TTS_CACHE[cache_key] = audio
            return audio
        logger.warning("ElevenLabs synthesis failed or unavailable. Falling back to Edge-TTS.")

    voice_id = _resolve_edge_voice_id(voice)
    chunks = []
    async for chunk in _synthesize_edge_stream(cleaned, voice_id):
        chunks.append(chunk)

    audio_bytes = b"".join(chunks)
    if audio_bytes and len(_TTS_CACHE) < _MAX_CACHE_ENTRIES:
        _TTS_CACHE[cache_key] = audio_bytes

    return audio_bytes


async def warmup_tts_cache():
    """
    Pre-warms the cache for common fixed responses to ensure ultra-low latency.
    """
    for v in ["female", "male"]:
        for phrase in [UNCLEAR_SPEECH_REPLY, GREETING_REPLY]:
            try:
                await synthesize(phrase, voice=v)
            except Exception as e:
                logger.warning(f"Warmup synthesis failed for '{phrase[:15]}': {e}")
