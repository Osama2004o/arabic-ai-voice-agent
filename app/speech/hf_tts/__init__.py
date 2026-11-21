import logging
from typing import Optional
from .engine import TinyVocaEngine

logger = logging.getLogger("speech.hf_tts")


def synthesize_hf(text: str, voice: str = "female") -> Optional[bytes]:
    """
    Synthesize Arabic speech using local CPU TinyVoca model.
    Returns standard 24 kHz mono WAV bytes.
    """
    try:
        engine = TinyVocaEngine.get_instance()
        return engine.synthesize_wav(text, voice=voice)
    except Exception as exc:
        logger.error("Hugging Face TinyVoca synthesis failed: %s", exc)
        return None


def warmup_hf():
    """
    Pre-loads the ONNX model and diacritizer into memory to eliminate first-turn latency.
    """
    try:
        engine = TinyVocaEngine.get_instance()
        engine.load()
    except Exception as exc:
        logger.warning("Hugging Face TinyVoca warmup failed: %s", exc)
