import asyncio
import io
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from groq import AsyncGroq

from .config import (
    STT_PROVIDER,
    GROQ_STT_MODEL,
    LOCAL_WHISPER_MODEL,
    STT_PROMPT,
    MIN_AUDIO_BYTES,
    MAX_NO_SPEECH_PROB,
    KNOWN_HALLUCINATIONS,
)
from .text_utils import normalize_arabic

logger = logging.getLogger("speech.stt")

# Singleton client instances
_groq_client: Optional[AsyncGroq] = None
_local_whisper_model = None


@dataclass
class TranscriptResult:
    text: str
    language: str
    stt_ms: float
    provider: str
    no_speech_prob: Optional[float] = None


def _get_groq_client() -> Optional[AsyncGroq]:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            _groq_client = AsyncGroq(api_key=api_key)
        else:
            logger.warning("GROQ_API_KEY is not set. Groq STT will not be available directly.")
    return _groq_client


def _get_local_model():
    global _local_whisper_model
    if _local_whisper_model is None:
        logger.info(f"Loading local faster-whisper model '{LOCAL_WHISPER_MODEL}' on CPU (int8)...")
        from faster_whisper import WhisperModel
        _local_whisper_model = WhisperModel(
            LOCAL_WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )
        logger.info("Local faster-whisper model loaded successfully.")
    return _local_whisper_model


def _ext_from_mime(mime: str) -> str:
    cleaned_mime = mime.split(";")[0].strip().lower()
    mapping = {
        "audio/webm": "webm",
        "audio/ogg": "ogg",
        "audio/mp4": "mp4",
        "audio/m4a": "m4a",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
    }
    return mapping.get(cleaned_mime, "webm")


def _is_silence_or_hallucination(text: str, no_speech_prob: Optional[float]) -> bool:
    if not text:
        return True

    # Check no-speech probability threshold
    if no_speech_prob is not None and no_speech_prob >= MAX_NO_SPEECH_PROB:
        logger.info(f"Rejected transcript due to high no_speech_prob: {no_speech_prob:.2f}")
        return True

    # Check if text is only punctuation / empty after normalization
    norm = normalize_arabic(text)
    if not norm or len(norm.strip()) < 2:
        return True

    # Check against known YouTube/credit hallucinations on silent Arabic audio
    for hallu in KNOWN_HALLUCINATIONS:
        if hallu in text or hallu in norm:
            logger.info(f"Detected and suppressed known hallucination: '{text}'")
            return True

    return False


async def _transcribe_groq(
    audio_bytes: bytes,
    mime_type: str,
    language: str,
    model: str = GROQ_STT_MODEL,
) -> Tuple[str, Optional[float]]:
    client = _get_groq_client()
    if client is None:
        raise RuntimeError("Groq client not initialized (missing API key).")

    ext = _ext_from_mime(mime_type)
    filename = f"audio.{ext}"

    resp = await client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=model,
        language=language,
        prompt=STT_PROMPT,
        temperature=0.0,
        response_format="verbose_json",
    )

    text = resp.text.strip() if hasattr(resp, "text") and resp.text else ""

    # Extract max no_speech_prob from segments if present
    no_speech_prob = None
    segments = getattr(resp, "segments", None) or []
    if segments:
        probs = []
        for s in segments:
            val = s.get("no_speech_prob") if isinstance(s, dict) else getattr(s, "no_speech_prob", None)
            if val is not None:
                probs.append(float(val))
        if probs:
            no_speech_prob = max(probs)

    return text, no_speech_prob


def _transcribe_local_sync(
    audio_bytes: bytes,
    language: str,
) -> Tuple[str, Optional[float]]:
    model = _get_local_model()
    segments, info = model.transcribe(
        io.BytesIO(audio_bytes),
        language=language,
        beam_size=1,
        vad_filter=True,
        initial_prompt=STT_PROMPT,
        temperature=0.0,
    )
    seg_list = list(segments)
    text = " ".join(s.text.strip() for s in seg_list).strip()
    no_speech_prob = getattr(info, "all_language_probs", None)
    return text, None


async def transcribe(
    audio_bytes: bytes,
    mime_type: str = "audio/webm",
    language: str = "ar",
    force_provider: Optional[str] = None,
) -> TranscriptResult:
    """
    Transcribes audio bytes to Arabic text.
    Uses Groq Whisper-large-v3-turbo as the high-speed primary path.
    Falls back gracefully to local CPU faster-whisper if Groq fails or is requested.
    Applies silence and hallucination filtering.
    """
    start_time = time.perf_counter()

    # Pre-validation: reject tiny/empty recordings
    if len(audio_bytes) < MIN_AUDIO_BYTES:
        logger.info(f"Audio payload too small ({len(audio_bytes)} bytes < {MIN_AUDIO_BYTES}). Rejecting.")
        return TranscriptResult(
            text="",
            language=language,
            stt_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
            provider="none (audio too short)",
            no_speech_prob=1.0,
        )

    provider = force_provider or STT_PROVIDER
    transcript = ""
    nsp = None
    used_provider = provider

    if provider == "groq":
        try:
            transcript, nsp = await _transcribe_groq(audio_bytes, mime_type, language)
            used_provider = f"groq:{GROQ_STT_MODEL}"
        except Exception as e:
            logger.warning(f"Groq STT failed ({e}). Falling back to local faster-whisper CPU model...")
            try:
                transcript, nsp = await asyncio.to_thread(_transcribe_local_sync, audio_bytes, language)
                used_provider = f"local:{LOCAL_WHISPER_MODEL}"
            except Exception as local_err:
                logger.error(f"Local STT fallback also failed: {local_err}")
                return TranscriptResult(
                    text="",
                    language=language,
                    stt_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                    provider="error",
                    no_speech_prob=None,
                )
    else:
        # Local direct
        transcript, nsp = await asyncio.to_thread(_transcribe_local_sync, audio_bytes, language)
        used_provider = f"local:{LOCAL_WHISPER_MODEL}"

    # Apply silence and hallucination guard
    if _is_silence_or_hallucination(transcript, nsp):
        logger.info(f"Silence/hallucination filter triggered. Returning empty transcript.")
        transcript = ""

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    return TranscriptResult(
        text=transcript,
        language=language,
        stt_ms=elapsed_ms,
        provider=used_provider,
        no_speech_prob=nsp,
    )
