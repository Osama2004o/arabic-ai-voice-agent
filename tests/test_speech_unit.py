import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.speech.text_utils import clean_text_for_tts, normalize_arabic
from app.speech.latency import LatencyTracker, summarize
from app.speech.stt import _is_silence_or_hallucination
from app.speech.config import MIN_AUDIO_BYTES, MAX_AUDIO_BYTES

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Text Utility Tests
# ---------------------------------------------------------------------------

def test_clean_text_for_tts_removes_markdown():
    raw = "**مرحباً** بك في _خدمة_ العملاء! للمزيد زر [موقعنا](https://example.com) `كود 12`"
    cleaned = clean_text_for_tts(raw)
    assert "**" not in cleaned
    assert "_" not in cleaned
    assert "`" not in cleaned
    assert "https://example.com" not in cleaned
    assert "موقعنا" in cleaned
    assert "مرحباً بك في خدمة العملاء" in cleaned


def test_clean_text_for_tts_removes_emojis():
    raw = "أهلاً وسهلاً بكم 😃🎉 نتمنى لكم يوماً سعيداً ❤️"
    cleaned = clean_text_for_tts(raw)
    assert "😃" not in cleaned
    assert "🎉" not in cleaned
    assert "❤️" not in cleaned
    assert "أهلاً وسهلاً بكم نتمنى لكم يوماً سعيداً" in cleaned


def test_clean_text_for_tts_spaces_long_numbers():
    raw = "رقم الطلب الخاص بك هو 94821 وسيصل قريباً"
    cleaned = clean_text_for_tts(raw)
    assert "9 4 8 2 1" in cleaned


def test_clean_text_for_tts_caps_length():
    long_text = "كلمة " * 300
    cleaned = clean_text_for_tts(long_text, max_chars=100)
    assert len(cleaned) <= 102
    assert cleaned.endswith(".")


def test_normalize_arabic_tashkeel_and_tatweel():
    raw = "الْكِتَابُ جَمِيلٌ جدّاً ومُــفِيد"
    norm = normalize_arabic(raw)
    assert norm == "الكتاب جميل جدا ومفيد"


def test_normalize_arabic_unification():
    raw = "إبراهيم أحمد إلى القاهرة و فاطمة"
    norm = normalize_arabic(raw)
    # أ/إ/آ -> ا, ى -> ي, ة -> ه
    assert norm == "ابراهيم احمد الي القاهره و فاطمه"


def test_normalize_arabic_digits_and_punctuation():
    raw = "السعر: ٥٠ جنيه، ورقم الهاتف 01012345678؟!"
    norm = normalize_arabic(raw)
    assert "50" in norm
    assert "01012345678" in norm
    assert ":" not in norm
    assert "،" not in norm
    assert "؟" not in norm
    assert "!" not in norm


# ---------------------------------------------------------------------------
# 2. Latency Tracker & Summarize Tests
# ---------------------------------------------------------------------------

def test_latency_tracker_stages():
    tracker = LatencyTracker()
    with tracker.stage("stt"):
        pass  # instantaneous
    with tracker.stage("tts"):
        pass

    timings = tracker.finish()
    assert "stt" in timings
    assert "tts" in timings
    assert "server_total" in timings
    assert timings["server_total"] >= 0.0


def test_summarize_metrics():
    latencies = [1200.0, 1500.0, 1800.0, 2100.0, 3000.0]
    stats = summarize(latencies)
    assert stats["n"] == 5
    assert stats["min"] == 1200.0
    assert stats["max"] == 3000.0
    assert stats["avg"] == 1920.0
    # int(0.95 * 4) = 3 -> latencies[3] = 2100.0
    assert stats["p95"] == 2100.0



def test_summarize_empty():
    stats = summarize([])
    assert stats["n"] == 0
    assert stats["avg"] == 0.0


# ---------------------------------------------------------------------------
# 3. Silence and Hallucination Filter Tests
# ---------------------------------------------------------------------------

def test_silence_guard_empty_or_punctuation():
    assert _is_silence_or_hallucination("", 0.1) is True
    assert _is_silence_or_hallucination("   ... ؟؟  ", 0.1) is True


def test_silence_guard_high_no_speech_prob():
    assert _is_silence_or_hallucination("أهلاً وسهلاً", 0.75) is True
    assert _is_silence_or_hallucination("أهلاً وسهلاً", 0.20) is False


def test_silence_guard_known_hallucination():
    assert _is_silence_or_hallucination("شكرا لكم على المشاهدة", 0.1) is True
    assert _is_silence_or_hallucination("اشترك في القناة وتفعيل الجرس", 0.1) is True


# ---------------------------------------------------------------------------
# 4. API Router Tests
# ---------------------------------------------------------------------------

def test_get_voices_endpoint():
    response = client.get("/speech/voices")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    ids = [v["id"] for v in data["voices"]]
    assert "female" in ids
    assert "male" in ids


def test_speech_stt_rejects_empty():
    # Empty audio file (0 bytes) should be rejected or return empty
    fake_audio = io.BytesIO(b"")
    response = client.post(
        "/speech/stt",
        files={"audio": ("empty.webm", fake_audio, "audio/webm")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == ""


def test_speech_stt_rejects_oversized():
    # File larger than MAX_AUDIO_BYTES
    huge_bytes = b"0" * (MAX_AUDIO_BYTES + 1024)
    huge_audio = io.BytesIO(huge_bytes)
    response = client.post(
        "/speech/stt",
        files={"audio": ("huge.webm", huge_audio, "audio/webm")},
    )
    assert response.status_code == 413


def test_speech_tts_validation():
    response = client.post("/speech/tts", json={"text": ""})
    assert response.status_code == 400
