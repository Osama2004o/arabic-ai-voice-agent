import io
import json
import logging
import os
import re
import threading
import wave
from typing import Optional, List

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from .numbers import expand_numbers
from .text import phonemize, NothingToPronounce

logger = logging.getLogger("speech.hf_tts")

DEFAULT_MODEL_ID = "oddadmix/tiny-voca-arabic"
SENTENCE_END = re.compile(r"(?<=[.!?؟…])\s+(?=\S)")
CLAUSE_END = re.compile(r"(?<=[،,;؛:])\s+")
MAX_CHARS = 3000
SAMPLE_RATE = 24000


class TinyVocaEngine:
    """
    Lightweight, CPU-only Arabic Text-to-Speech engine powered by oddadmix/tiny-voca-arabic.
    Generates 24 kHz mono 16-bit WAV audio in ~200-500ms on CPU with zero external API calls.
    """

    _instance: Optional["TinyVocaEngine"] = None
    _lock = threading.Lock()

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        self._session: Optional[ort.InferenceSession] = None
        self._init_lock = threading.Lock()
        self.sample_rate = SAMPLE_RATE
        self.max_tokens = 500

    @classmethod
    def get_instance(cls, model_id: str = DEFAULT_MODEL_ID) -> "TinyVocaEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(model_id=model_id)
            return cls._instance

    def load(self):
        """
        Ensures the ONNX model and diacritizer are downloaded and loaded in memory.
        """
        if self._session is not None:
            return

        with self._init_lock:
            if self._session is not None:
                return

            logger.info("Loading TinyVoca Arabic ONNX model from %s...", self.model_id)
            onnx_path = hf_hub_download(repo_id=self.model_id, filename="onnx/model.onnx")

            optimal_threads = int(os.environ.get("THREADS", str(min(6, max(2, (os.cpu_count() or 4) // 2)))))
            os.environ.setdefault("CATT_THREADS", str(optimal_threads))
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = optimal_threads
            sess_opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(onnx_path, sess_opts, providers=["CPUExecutionProvider"])

            # Warm up diacritizer
            try:
                from .arabic_frontend import diacritize
                diacritize(["مرحبا"])
            except Exception as e:
                logger.warning("CATT diacritizer warm-up warning: %s", e)

            logger.info("TinyVoca Arabic ONNX engine loaded successfully on CPU.")

    def _run_inference(self, phones, speed: float, pitch_shift: float, pitch_range: float) -> np.ndarray:
        one = lambda v: np.array([v], dtype=np.float32)
        audio, _ = self._session.run(
            None,
            {
                "tokens": np.array([phones.ids], dtype=np.int64),
                "stress": np.array([phones.stress], dtype=np.int64),
                "voice": np.array([0], dtype=np.int64),
                "speed": one(speed),
                "pitch_shift": one(pitch_shift * 0.058),  # 0.058 ≈ 1 semitone
                "pitch_range": one(pitch_range),
            },
        )
        return audio[0]

    def _split_long_clause(self, text: str) -> List[str]:
        parts = [p.strip() for p in CLAUSE_END.split(text) if p.strip()]
        if len(parts) > 1:
            return parts
        words = text.split()
        if len(words) < 2:
            return []
        half = len(words) // 2
        return [" ".join(words[:half]), " ".join(words[half:])]

    def synthesize_wav(
        self,
        text: str,
        voice: str = "female",
        speed: float = 1.0,
    ) -> bytes:
        """
        Synthesizes text into 24 kHz mono 16-bit WAV bytes.
        """
        cleaned = (text or "").strip()[:MAX_CHARS]
        if not cleaned:
            return b""

        self.load()

        # Adjust pitch depending on requested voice
        voice_lower = (voice or "").lower().strip()
        if "female" in voice_lower:
            pitch_shift = 2.5
            pitch_range = 1.1
        else:
            pitch_shift = -11.0
            pitch_range = 0.55

        if cleaned[-1] not in ".!?؟…":
            cleaned += "."

        waves: List[np.ndarray] = []
        gap = np.zeros(int(0.12 * self.sample_rate), dtype=np.float32)

        def read_piece(piece: str):
            try:
                phones = phonemize(piece)
            except NothingToPronounce:
                return
            except Exception as exc:
                logger.warning("Phonemize error on piece '%s': %s", piece[:30], exc)
                return

            if len(phones) <= self.max_tokens:
                audio_piece = self._run_inference(phones, speed, pitch_shift, pitch_range)
                waves.extend([audio_piece, gap])
                return

            for subpart in self._split_long_clause(piece):
                read_piece(subpart)

        expanded_text = expand_numbers(cleaned)
        for sentence in SENTENCE_END.split(expanded_text):
            sentence = sentence.strip()
            if sentence:
                read_piece(sentence)

        if not waves:
            logger.warning("TinyVoca produced no audio waves for text: %s", text[:40])
            return b""

        # Combine and normalize
        audio = np.concatenate(waves[:-1]) if len(waves) > 1 else waves[0]
        peak = float(np.abs(audio).max())
        if peak > 0.99:
            audio = audio * (0.99 / peak)

        # Convert float32 [-1.0, 1.0] to int16 PCM
        pcm_data = (audio * 32767).astype(np.int16).tobytes()

        # Wrap in standard RIFF WAV container
        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)      # Mono
                wf.setsampwidth(2)      # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm_data)
            return buf.getvalue()
