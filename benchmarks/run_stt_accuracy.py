import asyncio
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
import jiwer

# Add repo root to sys.path so app modules import cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.speech.text_utils import normalize_arabic
from app.speech.stt import _transcribe_groq, _transcribe_local_sync, _is_silence_or_hallucination
from app.speech.config import LOCAL_WHISPER_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark.stt")

AUDIO_DIR = ROOT_DIR / "benchmarks" / "test_audio"
RESULTS_DIR = ROOT_DIR / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_wer(reference: str, hypothesis: str) -> float:
    ref_norm = normalize_arabic(reference)
    hyp_norm = normalize_arabic(hypothesis)

    if not ref_norm and not hyp_norm:
        return 0.0
    if not ref_norm and hyp_norm:
        return 1.0  # Hallucinated on silence
    if ref_norm and not hyp_norm:
        return 1.0  # Missed speech entirely

    try:
        return round(float(jiwer.wer(ref_norm, hyp_norm)), 4)
    except Exception as e:
        logger.warning(f"WER calculation error: {e}")
        return 1.0


async def evaluate_clip(clip_entry: dict, model_name: str) -> dict:
    clip_file = AUDIO_DIR / clip_entry["file"]
    reference_text = clip_entry["text"]
    category = clip_entry["category"]

    with open(clip_file, "rb") as f:
        audio_bytes = f.read()

    ext = clip_file.suffix.lstrip(".").lower()
    mime = "audio/wav" if ext == "wav" else "audio/mpeg"

    start = time.perf_counter()
    hypothesis = ""
    nsp = None

    if model_name.startswith("groq:"):
        groq_model = model_name.split(":", 1)[1]
        try:
            raw_hyp, nsp = await _transcribe_groq(audio_bytes, mime, language="ar", model=groq_model)
            if not _is_silence_or_hallucination(raw_hyp, nsp):
                hypothesis = raw_hyp
            else:
                hypothesis = ""
        except Exception as e:
            logger.error(f"Groq ({groq_model}) transcription failed on {clip_file.name}: {e}")
            hypothesis = ""
    elif model_name.startswith("local:"):
        try:
            raw_hyp, nsp = await asyncio.to_thread(_transcribe_local_sync, audio_bytes, "ar")
            if not _is_silence_or_hallucination(raw_hyp, nsp):
                hypothesis = raw_hyp
            else:
                hypothesis = ""
        except Exception as e:
            logger.error(f"Local transcription failed on {clip_file.name}: {e}")
            hypothesis = ""

    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    wer = calculate_wer(reference_text, hypothesis)

    return {
        "file": clip_file.name,
        "category": category,
        "model": model_name,
        "wer": wer,
        "stt_ms": elapsed_ms,
        "reference": reference_text,
        "hypothesis": hypothesis,
    }


async def main():
    manifest_path = AUDIO_DIR / "references.json"
    if not manifest_path.exists():
        logger.error(f"References manifest {manifest_path} not found. Run generate_synthetic_audio.py first.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        clips = json.load(f)

    models_to_test = [
        "groq:whisper-large-v3-turbo",
        "groq:whisper-large-v3",
        f"local:faster-whisper-{LOCAL_WHISPER_MODEL}",
    ]

    all_records = []
    csv_file = RESULTS_DIR / "stt_accuracy.csv"

    print("=" * 80)
    print("STARTING STT ACCURACY BENCHMARK (WER & Latency)")
    print(f"Total test clips: {len(clips)} | Models: {models_to_test}")
    print("=" * 80)

    for model_name in models_to_test:
        print(f"\n--- Evaluating Model: {model_name} ---")
        for i, clip in enumerate(clips, start=1):
            print(f"[{i:02d}/{len(clips):02d}] {clip['file']} ({clip['category']})...", end=" ", flush=True)
            res = await evaluate_clip(clip, model_name)
            all_records.append(res)
            print(f"WER={res['wer']:.2f} | Time={res['stt_ms']:.0f}ms")

            # Pause to respect Groq free tier rate limits
            if model_name.startswith("groq:"):
                await asyncio.sleep(1.2)

    # Save to CSV
    fieldnames = ["file", "category", "model", "wer", "stt_ms", "reference", "hypothesis"]
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + "=" * 80)
    print(f"RESULTS SAVED TO {csv_file}")
    print("=" * 80)

    # Print Summary Table
    print("\n=== SUMMARY TABLE: MEAN WER & LATENCY BY MODEL & CATEGORY ===")
    from collections import defaultdict
    category_model_wer = defaultdict(lambda: defaultdict(list))
    model_time = defaultdict(list)
    model_overall_wer = defaultdict(list)

    for r in all_records:
        m = r["model"]
        c = r["category"]
        w = r["wer"]
        t = r["stt_ms"]
        category_model_wer[c][m].append(w)
        model_time[m].append(t)
        model_overall_wer[m].append(w)

    categories = sorted(list(category_model_wer.keys()))
    header = f"{'Category':<35} | " + " | ".join([f"{m[:18]:<18}" for m in models_to_test])
    print(header)
    print("-" * len(header))

    for cat in categories:
        row_str = f"{cat:<35} | "
        cols = []
        for m in models_to_test:
            vals = category_model_wer[cat][m]
            mean_wer = sum(vals) / len(vals) if vals else 0.0
            cols.append(f"{mean_wer * 100:6.1f}%{'':<11}")
        row_str += " | ".join(cols)
        print(row_str)

    print("-" * len(header))
    overall_row = f"{'OVERALL MEAN WER':<35} | "
    overall_row += " | ".join([f"{sum(model_overall_wer[m])/len(model_overall_wer[m])*100:6.1f}%{'':<11}" for m in models_to_test])
    print(overall_row)

    time_row = f"{'MEAN STT LATENCY (ms)':<35} | "
    time_row += " | ".join([f"{sum(model_time[m])/len(model_time[m]):6.0f} ms{'':<9}" for m in models_to_test])
    print(time_row)
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
