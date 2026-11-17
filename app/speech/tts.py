import asyncio
import logging
from typing import AsyncIterator, Dict, Tuple, Literal
import edge_tts

from .config import VOICES, DEFAULT_VOICE, UNCLEAR_SPEECH_REPLY, GREETING_REPLY
from .text_utils import clean_text_for_tts

logger = logging.getLogger("speech.tts")

VoiceType = Literal["female", "male"]

# In-memory cache for common/fixed phrases: (voice, clean_text) -> bytes
_TTS_CACHE: Dict[Tuple[str, str], bytes] = {}
_MAX_CACHE_ENTRIES = 128


def _resolve_voice_id(voice: str) -> str:
    cleaned = (voice or "").lower().strip()
    if cleaned in VOICES:
        return VOICES[cleaned]
    # Check if a direct voice identifier was passed
    if "neural" in cleaned:
        return voice
    return VOICES.get(DEFAULT_VOICE, VOICES["female"])


async def synthesize_stream(text: str, voice: str = "female") -> AsyncIterator[bytes]:
    """
    Streams MP3 audio chunks generated via edge-tts.
    """
    cleaned = clean_text_for_tts(text)
    if not cleaned:
        return

    voice_id = _resolve_voice_id(voice)
    cache_key = (voice_id, cleaned)

    # If cached, yield the cached payload as a single chunk
    if cache_key in _TTS_CACHE:
        yield _TTS_CACHE[cache_key]
        return

    # Try streaming with 1 retry on network glitch or timeout
    attempts = 2
    for attempt in range(1, attempts + 1):
        try:
            communicate = edge_tts.Communicate(cleaned, voice_id)
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



async def synthesize(text: str, voice: str = "female") -> bytes:
    """
    Synthesizes complete MP3 audio bytes for text.
    Uses in-memory caching and 1 retry on network error.
    """
    cleaned = clean_text_for_tts(text)
    if not cleaned:
        return b""

    voice_id = _resolve_voice_id(voice)
    cache_key = (voice_id, cleaned)

    if cache_key in _TTS_CACHE:
        return _TTS_CACHE[cache_key]

    chunks = []
    async for chunk in synthesize_stream(cleaned, voice):
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
