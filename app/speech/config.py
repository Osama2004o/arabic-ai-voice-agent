import os
from dotenv import load_dotenv

load_dotenv()

# STT Configuration
STT_PROVIDER = os.getenv("STT_PROVIDER", "groq").lower()
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
GROQ_ACCURACY_MODEL = "whisper-large-v3"
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "base")

# TTS Configuration
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge").lower()
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "female").lower()

VOICES = {
    "female": "ar-EG-SalmaNeural",
    "male": "ar-EG-ShakirNeural",
}

OPTIONAL_MSA_VOICES = {
    "female_msa": "ar-SA-ZariyahNeural",
    "male_msa": "ar-SA-HamedNeural",
}

# ElevenLabs Configuration (Free Tier supported: eleven_multilingual_v2, eleven_flash_v2_5)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
ELEVENLABS_FEMALE_VOICE_ID = os.getenv("ELEVENLABS_VOICE_FEMALE", "EXAVITQu4vr4xnSDxMaL")  # Sarah (Free tier pre-made)
ELEVENLABS_DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_VOICE_MALE") or os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Adam (Free tier pre-made)

ELEVENLABS_VOICES = {
    "female": ELEVENLABS_FEMALE_VOICE_ID,
    "female_msa": os.getenv("ELEVENLABS_VOICE_FEMALE_MSA", ELEVENLABS_FEMALE_VOICE_ID),
    "male": ELEVENLABS_DEFAULT_VOICE_ID,
    "male_msa": os.getenv("ELEVENLABS_VOICE_MALE_MSA", ELEVENLABS_DEFAULT_VOICE_ID),
}

# STT Egyptian-flavored hint to bias Whisper
STT_PROMPT = "ازيك، عايز أسأل عن الطلب بتاعي، الشكوى، الدفع، الشحن، الاسترجاع."

# Hallucination phrases commonly output by Whisper on silence / noise
KNOWN_HALLUCINATIONS = {
    "شكرا لكم على المشاهدة",
    "شكراً للمشاهدة",
    "شكرا على المشاهدة",
    "اشترك في القناة",
    "اشتركوا في القناة",
    "اشتركوا بالقناة",
    "لاتنسوا الاشتراك في القناة",
    "لا تنسوا الاشتراك في القناة",
    "تفعيل جرس التنبيهات",
    "تفعيل زر التنبيهات",
    "مع السلامة",
    "مرحباً بكم",
    "ترجمة:",
    "Subtitle by",
    "Subtitles by",
}

# Audio Constraints
MIN_AUDIO_BYTES = 2000          # reject tiny/empty recordings
MAX_AUDIO_BYTES = 10 * 1024**2  # 10 MB limit
MIN_AUDIO_DURATION_SEC = 0.4
MAX_NO_SPEECH_PROB = 0.60

# Fixed responses
UNCLEAR_SPEECH_REPLY = "لم أسمعك جيدًا، هل يمكنك إعادة المحاولة؟"
GREETING_REPLY = "أهلاً بك! كيف يمكنني مساعدتك اليوم؟"

# LLM Stub configuration (reads OpenRouter model from .env: LLM_MODEL or LLM_STUB_MODEL)
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("LLM_STUB_MODEL", "openrouter/free")
LLM_STUB_MODEL = LLM_MODEL
