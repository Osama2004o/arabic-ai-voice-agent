import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.speech.hf_tts import synthesize_hf
from app.speech.tts import synthesize


@pytest.fixture
def client():
    return TestClient(app)


def test_hf_synthesize_wav_format():
    """
    Verify TinyVoca produces valid RIFF WAV audio bytes.
    """
    wav = synthesize_hf("أهلاً بك", voice="female")
    assert wav is not None
    assert len(wav) > 1000
    assert wav.startswith(b"RIFF")


def test_hf_synthesize_both_voices():
    """
    Verify both male and female voice parameters generate valid audio.
    """
    wav_female = synthesize_hf("مرحبا بكم", voice="female")
    wav_male = synthesize_hf("مرحبا بكم", voice="male")

    assert wav_female is not None and wav_female.startswith(b"RIFF")
    assert wav_male is not None and wav_male.startswith(b"RIFF")


@pytest.mark.asyncio
async def test_hf_synthesize_router_integration():
    """
    Verify speech.tts.synthesize routes to Hugging Face when provider is huggingface.
    """
    wav = await synthesize("تجربة فحص الصوت", voice="female", provider="huggingface")
    assert wav is not None
    assert wav.startswith(b"RIFF")


@pytest.mark.asyncio
async def test_hf_fallback_to_edge_on_error():
    """
    Verify fallback to Edge-TTS if Hugging Face synthesis raises an exception.
    """
    with patch("app.speech.tts.synthesize_hf", side_effect=RuntimeError("Simulated HF failure")):
        # Edge-TTS fallback produces MP3 audio (does not start with RIFF)
        audio = await synthesize("تجربة السقوط الاحتياطي", voice="female", provider="huggingface")
        assert audio is not None
        assert len(audio) > 0


def test_voices_endpoint_huggingface(client):
    """
    Verify GET /speech/voices?provider=huggingface returns TinyVoca voice options.
    """
    response = client.get("/speech/voices?provider=huggingface")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "huggingface"
    assert "huggingface" in data["providers"]
    voices = data["voices"]
    assert any("TinyVoca" in v["label"] for v in voices)
