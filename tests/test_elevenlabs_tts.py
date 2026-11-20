"""Unit tests for ElevenLabs TTS integration, voice resolution, streaming, and Edge-TTS fallback."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.speech.config import ELEVENLABS_DEFAULT_VOICE_ID
from app.speech.tts import (
    _resolve_elevenlabs_voice_id,
    _synthesize_elevenlabs,
    _stream_elevenlabs,
    synthesize,
    synthesize_stream,
)


def test_elevenlabs_voice_resolution():
    """Verify female resolves to EXAVITQu4vr4xnSDxMaL (Sarah) and other voices resolve to pNInz6obpgDQGcFmaJgB (Adam)."""
    assert _resolve_elevenlabs_voice_id("female") == "EXAVITQu4vr4xnSDxMaL"
    assert _resolve_elevenlabs_voice_id("female_msa") == "EXAVITQu4vr4xnSDxMaL"
    assert _resolve_elevenlabs_voice_id("male") == "pNInz6obpgDQGcFmaJgB"
    assert _resolve_elevenlabs_voice_id("male_msa") == "pNInz6obpgDQGcFmaJgB"
    assert _resolve_elevenlabs_voice_id("unknown_alias") == "pNInz6obpgDQGcFmaJgB"

    # Direct 20+ character voice ID
    custom_voice = "21m00Tcm4TlvDq8ikWAM"
    assert _resolve_elevenlabs_voice_id(custom_voice) == custom_voice


@pytest.mark.asyncio
async def test_elevenlabs_missing_api_key_returns_none():
    """Verify _synthesize_elevenlabs returns None safely when ELEVENLABS_API_KEY is empty."""
    with patch("app.speech.tts.ELEVENLABS_API_KEY", ""):
        result = await _synthesize_elevenlabs("مرحبا بك", "female")
        assert result is None


@pytest.mark.asyncio
async def test_elevenlabs_mock_success():
    """Verify _synthesize_elevenlabs returns binary audio content on 200 OK."""
    fake_audio = b"\xff\xfb\x90\x00ELEVENLABS_MP3_DATA"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = fake_audio

    with patch("app.speech.tts.ELEVENLABS_API_KEY", "test_eleven_key"):
        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            result = await _synthesize_elevenlabs("مرحبا بك", "female")
            assert result == fake_audio


@pytest.mark.asyncio
async def test_elevenlabs_fallback_to_edge_on_quota_error():
    """Verify synthesize() falls back to Edge-TTS when ElevenLabs returns 429 Quota Exceeded."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = '{"detail": {"status": "quota_exceeded"}}'

    fake_edge_audio = b"\xff\xfb\x90\x00EDGE_FALLBACK_AUDIO"

    async def fake_edge_stream(text, voice_id):
        yield fake_edge_audio

    with patch("app.speech.tts.TTS_PROVIDER", "elevenlabs"):
        with patch("app.speech.tts.ELEVENLABS_API_KEY", "test_eleven_key"):
            with patch("httpx.AsyncClient.post", return_value=mock_resp):
                with patch("app.speech.tts._synthesize_edge_stream", side_effect=fake_edge_stream):
                    audio = await synthesize("تجربة استرجاع", voice="female")
                    assert audio == fake_edge_audio


@pytest.mark.asyncio
async def test_elevenlabs_streaming_fallback_to_edge_on_error():
    """Verify synthesize_stream() falls back to Edge-TTS when ElevenLabs streaming fails."""
    fake_edge_audio = b"\xff\xfb\x90\x00EDGE_FALLBACK_STREAM"

    async def fake_edge_stream(text, voice_id):
        yield fake_edge_audio

    # When API key is missing or invalid, streaming should fall back to Edge-TTS
    with patch("app.speech.tts.TTS_PROVIDER", "elevenlabs"):
        with patch("app.speech.tts.ELEVENLABS_API_KEY", ""):
            with patch("app.speech.tts._synthesize_edge_stream", side_effect=fake_edge_stream):
                chunks = []
                async for chunk in synthesize_stream("تجربة فحص", voice="female"):
                    chunks.append(chunk)
                assert chunks == [fake_edge_audio]
