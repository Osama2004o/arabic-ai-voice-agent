import asyncio
import csv
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

# Add repo root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.speech.pipeline import run_voice_turn
from app.speech.stub_agent import stub_agent
from app.speech.tts import synthesize_stream, synthesize
from app.speech.latency import summarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark.latency")

AUDIO_DIR = ROOT_DIR / "benchmarks" / "test_audio"
RESULTS_DIR = ROOT_DIR / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def benchmark_tts_only():
    """
    Measures TTS time-to-first-chunk and total synthesis duration across short, medium, and long texts.
    """
    print("\n--- Running TTS-Only Detailed Benchmark ---")
    test_cases = [
        ("short", "أهلاً بك، كيف أساعدك؟"),
        ("medium", "تم استلام طلبك بنجاح وسيقوم مندوب الشحن بالتواصل معك هاتفياً قبل الوصول لتأكيد العنوان."),
        ("long", "شكراً لتواصلك مع خدمة العملاء. نود إحاطتك علماً بأن سياسة الاسترجاع تتيح لك استبدال المنتج خلال أربعة عشر يوماً من تاريخ الاستلام في حال وجود أي عيب مصنعي أو رغبة في استبدال المقاس، بشرط الاحتفاظ بالفاتورة والتغليف الأصلي."),
    ]
    voices = ["female", "male"]
    tts_records = []

    for length_name, text in test_cases:
        for voice in voices:
            # First Chunk Time
            t_start = time.perf_counter()
            first_chunk_ms = None
            total_bytes = 0

            async for chunk in synthesize_stream(text, voice=voice):
                if first_chunk_ms is None:
                    first_chunk_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                total_bytes += len(chunk)

            total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            word_count = len(text.split())

            tts_records.append({
                "length_category": length_name,
                "word_count": word_count,
                "voice": voice,
                "first_chunk_ms": first_chunk_ms or total_ms,
                "total_synthesis_ms": total_ms,
                "audio_bytes": total_bytes,
            })
            print(f"TTS [{voice} | {length_name} ({word_count} words)] -> First chunk: {first_chunk_ms}ms | Total: {total_ms}ms")
            await asyncio.sleep(0.3)

    tts_csv = RESULTS_DIR / "tts_latency.csv"
    with open(tts_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["length_category", "word_count", "voice", "first_chunk_ms", "total_synthesis_ms", "audio_bytes"])
        writer.writeheader()
        writer.writerows(tts_records)

    print(f"TTS latency benchmark saved to {tts_csv}")
    return tts_records


async def main():
    manifest_path = AUDIO_DIR / "references.json"
    if not manifest_path.exists():
        logger.error(f"References manifest {manifest_path} not found. Run generate_synthetic_audio.py first.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        all_clips = json.load(f)

    # Exclude pure silence clips from conversational latency runs to measure full end-to-end turns
    speech_clips = [c for c in all_clips if c["category"] != "Silence / non-speech"]

    # We need >= 30 total turns as specified in section 10.3
    turns_target = 32
    clips_to_run = (speech_clips * 2)[:turns_target]

    print("=" * 80)
    print("STARTING END-TO-END VOICE LATENCY BENCHMARK (< 3.0s TARGET)")
    print(f"Test Environment: OS={platform.system()} {platform.release()} | CPU={platform.processor()}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Target turns: {len(clips_to_run)}")
    print("=" * 80)

    turn_records = []
    csv_file = RESULTS_DIR / "latency.csv"

    for i, clip in enumerate(clips_to_run, start=1):
        clip_path = AUDIO_DIR / clip["file"]
        voice = "female" if (i % 2 == 1) else "male"

        with open(clip_path, "rb") as f:
            audio_bytes = f.read()

        mime = "audio/wav" if clip_path.suffix == ".wav" else "audio/mpeg"

        print(f"[{i:02d}/{len(clips_to_run):02d}] Turn on {clip['file']} ({voice})...", end=" ", flush=True)

        try:
            turn_result = await run_voice_turn(
                audio_bytes=audio_bytes,
                mime_type=mime,
                voice=voice,
                session_id=f"bench_sess_{i}",
                agent_fn=stub_agent,
            )

            timings = turn_result["timings_ms"]
            server_total = timings.get("server_total", 0.0)
            stt_t = timings.get("stt", 0.0)
            llm_t = timings.get("llm", 0.0)
            tts_t = timings.get("tts", 0.0)
            transcript = turn_result.get("transcript", "")
            reply = turn_result.get("reply_text", "")

            status_str = "PASS (<3s)" if server_total < 3000.0 else "OVER (>3s)"
            print(f"STT: {stt_t:.0f}ms | LLM: {llm_t:.0f}ms | TTS: {tts_t:.0f}ms | Total: {server_total:.0f}ms [{status_str}]")

            turn_records.append({
                "turn_id": i,
                "file": clip["file"],
                "category": clip["category"],
                "voice": voice,
                "stt_ms": stt_t,
                "llm_ms": llm_t,
                "tts_ms": tts_t,
                "server_total_ms": server_total,
                "under_3s": server_total < 3000.0,
                "transcript": transcript,
                "reply_text": reply,
            })
        except Exception as e:
            logger.error(f"Turn {i} failed: {e}")

        # Sleep between requests to respect OpenRouter & Groq rate limits
        await asyncio.sleep(1.5)

    # Save to CSV
    fieldnames = [
        "turn_id", "file", "category", "voice",
        "stt_ms", "llm_ms", "tts_ms", "server_total_ms",
        "under_3s", "transcript", "reply_text"
    ]
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(turn_records)

    print("\n" + "=" * 80)
    print(f"RESULTS SAVED TO {csv_file}")
    print("=" * 80)

    # Compute Stage & Total Statistics
    stt_vals = [r["stt_ms"] for r in turn_records]
    llm_vals = [r["llm_ms"] for r in turn_records]
    tts_vals = [r["tts_ms"] for r in turn_records]
    tot_vals = [r["server_total_ms"] for r in turn_records]

    stt_stat = summarize(stt_vals)
    llm_stat = summarize(llm_vals)
    tts_stat = summarize(tts_vals)
    tot_stat = summarize(tot_vals)

    under_3s_count = sum(1 for r in turn_records if r["under_3s"])
    pct_under_3s = (under_3s_count / len(turn_records) * 100) if turn_records else 0.0

    print("\n=== LATENCY BENCHMARK SUMMARY (N = %d turns) ===" % len(turn_records))
    print(f"{'Stage':<20} | {'Budget':<10} | {'Avg (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'P95 (ms)':<10}")
    print("-" * 80)
    print(f"{'STT (Groq Turbo)':<20} | {'<= 1000':<10} | {stt_stat['avg']:<10.1f} | {stt_stat['min']:<10.1f} | {stt_stat['max']:<10.1f} | {stt_stat['p95']:<10.1f}")
    print(f"{'LLM (Agent Stub)':<20} | {'<= 1000':<10} | {llm_stat['avg']:<10.1f} | {llm_stat['min']:<10.1f} | {llm_stat['max']:<10.1f} | {llm_stat['p95']:<10.1f}")
    print(f"{'TTS (Edge-TTS)':<20} | {'<= 800':<10} | {tts_stat['avg']:<10.1f} | {tts_stat['min']:<10.1f} | {tts_stat['max']:<10.1f} | {tts_stat['p95']:<10.1f}")
    print("-" * 80)
    print(f"{'TOTAL SERVER TIME':<20} | {'< 3000':<10} | {tot_stat['avg']:<10.1f} | {tot_stat['min']:<10.1f} | {tot_stat['max']:<10.1f} | {tot_stat['p95']:<10.1f}")
    print("=" * 80)
    print(f">> Percentage of Turns Meeting < 3.0s Target: {pct_under_3s:.1f}% ({under_3s_count}/{len(turn_records)})")
    print("=" * 80)

    # Run TTS detailed benchmark
    await benchmark_tts_only()


if __name__ == "__main__":
    asyncio.run(main())
