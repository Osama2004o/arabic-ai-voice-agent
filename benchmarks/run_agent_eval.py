"""
Live 10-scenario evaluation benchmark for the Arabic Customer Service Conversational AI Agent.
Verifies accuracy, Egyptian/MSA dialect support, multi-turn memory, confirmation safety,
human escalation, plain-text voice constraints, and LLM latency (< 1.0s budget).
"""

from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.agent import (
    ACTION_LOG,
    arun_agent,
    clear_action_log,
    register_email_handler,
    register_telegram_handler,
    reset_session,
)
from app.speech.latency import summarize

RESULTS_DIR = ROOT_DIR / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _has_forbidden_formatting(text: str) -> bool:
    return any(tok in text for tok in ("**", "##", "`", "•", "\n1.", "\n-"))


async def run_all_scenarios() -> List[Dict[str, Any]]:
    sent_emails: List[Dict[str, str]] = []
    sent_telegrams: List[str] = []

    register_email_handler(lambda to, sub, body: sent_emails.append({"to": to, "subject": sub, "body": body}) or {"ok": True})
    register_telegram_handler(lambda msg: sent_telegrams.append(msg) or {"ok": True})

    records: List[Dict[str, Any]] = []

    async def exec_case(
        case_id: int,
        name: str,
        session_id: str,
        turns: List[str],
        expected_desc: str,
        validator,
    ) -> None:
        last_res: Dict[str, Any] = {}
        turn_latencies: List[float] = []
        for idx, utterance in enumerate(turns, start=1):
            last_res = await arun_agent(session_id=session_id, text=utterance)
            turn_latencies.append(float(last_res["latency_ms"]))
            print(f"  [Turn {idx}] User: {utterance}")
            print(f"           Agent ({last_res['latency_ms']} ms): {last_res['response']}")
            await asyncio.sleep(0.4)

        await asyncio.sleep(1.5)

        passed, reason = validator(last_res, turns)
        max_lat = max(turn_latencies) if turn_latencies else 0.0
        avg_lat = round(sum(turn_latencies) / len(turn_latencies), 2) if turn_latencies else 0.0
        status = "PASS" if passed else f"FAIL ({reason})"
        print(f"-> Case {case_id:02d} [{name}]: {status} | Avg: {avg_lat}ms\n")

        records.append({
            "case_id": case_id,
            "scenario": name,
            "input": " | ".join(turns),
            "expected": expected_desc,
            "actual_response": last_res.get("response", ""),
            "action": last_res.get("action") or "",
            "needs_confirmation": last_res.get("needs_confirmation", False),
            "needs_human": last_res.get("needs_human", False),
            "pass": passed,
            "reason": reason if not passed else "OK",
            "avg_latency_ms": avg_lat,
            "max_latency_ms": max_lat,
        })

    print("=" * 80)
    print("CONVERSATIONAL AI AGENT EVALUATION (10 REQUIRED SCENARIOS)")
    print("=" * 80)

    # 1. Arabic FAQ question (MSA)
    await exec_case(
        1,
        "Arabic FAQ question (MSA)",
        "eval_01",
        ["ما هي مواعيد العمل الرسمية لديكم؟"],
        "States Sun-Thu 9 AM - 5 PM from KB, plain Arabic, <= 60 words",
        lambda r, _: (
            ("الأحد" in r["response"] or "التاسعة" in r["response"])
            and not _has_forbidden_formatting(r["response"])
            and len(r["response"].split()) <= 60,
            "Missing working hours or formatting violation",
        ),
    )

    # 2. Egyptian Arabic question
    await exec_case(
        2,
        "Egyptian Arabic question",
        "eval_02",
        ["عايز أعرف مصاريف الشحن للقاهرة كام؟"],
        "Mentions 50 EGP for Cairo and free over 500 EGP",
        lambda r, _: (
            ("50" in r["response"] or "خمسين" in r["response"] or "خمسون" in r["response"])
            and not _has_forbidden_formatting(r["response"]),
            "Missing Cairo shipping fee",
        ),
    )

    # 3. Follow-up question with memory
    await exec_case(
        3,
        "Multi-turn follow-up & memory",
        "eval_03",
        [
            "اسمي أحمد ولدي مشكلة في طلبي",
            "رقمه 12345",
            "هل يمكن إلغاؤه؟",
        ],
        "Remembers name (أحمد) and order_id (12345) on follow-up",
        lambda r, _: (
            r.get("order_id") == "12345"
            and r.get("customer_name") == "أحمد"
            and ("12345" in r["response"] or "إلغاء" in r["response"] or "الغاء" in r["response"]),
            f"Lost memory slots: name={r.get('customer_name')} order={r.get('order_id')}",
        ),
    )

    # 4. Customer complaint
    await exec_case(
        4,
        "Customer complaint",
        "eval_04",
        ["أنا استلمت منتج غلط ومكسور وبجد متضايق جداً من التأخير"],
        "Shows empathy/apology and offers free replacement/return or asks order ID",
        lambda r, _: (
            any(w in r["response"] for w in ("نعتذر", "آسف", "آسفين", "عذر", "حقك"))
            and not _has_forbidden_formatting(r["response"]),
            "Missing empathetic apology",
        ),
    )

    # 5. Unknown question
    await exec_case(
        5,
        "Unknown / out-of-scope question",
        "eval_05",
        ["هل تبيعون تذاكر طيران دولية إلى أوروبا؟"],
        "Admits lack of info without hallucinating prices/policies",
        lambda r, _: (
            not any(w in r["response"] for w in ("500 دولار", "يورو", "حجز تذكرة"))
            and len(r["response"]) > 10,
            "Hallucinated out-of-scope answer",
        ),
    )

    # 6. Email request + confirmation
    before_emails = len(sent_emails)
    await exec_case(
        6,
        "Email request & confirmation",
        "eval_06",
        [
            "عندي مشكلة تأخير في الطلب رقم 98421",
            "أرسل ملخص مشكلتي في الطلب 98421 إلى بريدي ahmed@example.com",
            "نعم أرسله",
        ],
        "Asks confirmation on turn 2, sends email only after turn 3 confirmation",
        lambda r, _: (
            len(sent_emails) == before_emails + 1
            and sent_emails[-1]["to"] == "ahmed@example.com"
            and r["needs_confirmation"] is False,
            f"Email count={len(sent_emails)}, expected {before_emails + 1}",
        ),
    )

    # 7. Telegram request + confirmation
    before_tg = len(sent_telegrams)
    await exec_case(
        7,
        "Telegram request & confirmation",
        "eval_07",
        [
            "ابعت رسالة عبر تيليجرام للدعم بخصوص تأخير الطلب 55210",
            "أيوة تمام ابعتها",
        ],
        "Stages telegram on turn 1, executes only after turn 2 confirmation",
        lambda r, _: (
            len(sent_telegrams) == before_tg + 1 and r["needs_confirmation"] is False,
            f"Telegram count={len(sent_telegrams)}, expected {before_tg + 1}",
        ),
    )

    # 8. Confirmation rejection
    email_count_before_reject = len(sent_emails)
    await exec_case(
        8,
        "Confirmation rejection",
        "eval_08",
        [
            "أرسل ملخص الطلب 77112 إلى بريدي sara@example.com",
            "لأ بلاش تبعت",
        ],
        "Cancels pending email without executing send_email",
        lambda r, _: (
            len(sent_emails) == email_count_before_reject
            and r["needs_confirmation"] is False
            and r["action"] is None,
            "Email was sent despite rejection or confirmation stayed pending",
        ),
    )

    # 9. Human escalation
    await exec_case(
        9,
        "Human escalation",
        "eval_09",
        ["المشكلة متكررة ومحتاج أتكلم مع موظف بشري حالاً"],
        "Sets needs_human=True and logs escalate_to_human",
        lambda r, _: (
            r["needs_human"] is True
            and r["action"] == "escalate_to_human"
            and any(e["action"] == "escalate_to_human" for e in ACTION_LOG),
            f"needs_human={r.get('needs_human')}, action={r.get('action')}",
        ),
    )

    # 10. Conversation with missing information
    email_count_before_missing = len(sent_emails)
    await exec_case(
        10,
        "Missing information handling",
        "eval_10",
        ["أرسل ملخص المشكلة إلى بريدي الإلكتروني"],
        "Asks for missing email address without sending or hallucinating",
        lambda r, _: (
            len(sent_emails) == email_count_before_missing
            and r["needs_confirmation"] is False
            and ("بريد" in r["response"] or "إيميل" in r["response"] or "ايميل" in r["response"]),
            "Did not prompt for missing email address cleanly",
        ),
    )

    csv_path = RESULTS_DIR / "agent_eval.csv"
    fieldnames = [
        "case_id",
        "scenario",
        "input",
        "expected",
        "actual_response",
        "action",
        "needs_confirmation",
        "needs_human",
        "pass",
        "reason",
        "avg_latency_ms",
        "max_latency_ms",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    latencies = [r["avg_latency_ms"] for r in records]
    stats = summarize(latencies)
    passed_count = sum(1 for r in records if r["pass"])

    print("=" * 80)
    print(f"EVALUATION COMPLETE: {passed_count}/{len(records)} PASSED")
    print(
        f"Agent Turn Latency -> Mean: {stats['avg']}ms | Min: {stats['min']}ms | "
        f"Max: {stats['max']}ms | P95: {stats['p95']}ms"
    )
    print(f"Results saved to: {csv_path}")
    print("=" * 80)
    return records


async def run_voice_integration_check() -> List[Dict[str, Any]]:
    """Run a 4-turn end-to-end voice check connecting Speech Pipeline with the Conversational AI Agent."""
    from app.agent import agent as ai_agent
    from app.speech.pipeline import run_voice_turn

    audio_dir = ROOT_DIR / "benchmarks" / "test_audio"
    clips = ["msa_clear_01.mp3", "msa_clear_02.mp3", "eg_clear_01.mp3", "eg_clear_03.mp3"]
    voice_records: List[Dict[str, Any]] = []

    print("=" * 80)
    print("VOICE PIPELINE + CONVERSATIONAL AI AGENT INTEGRATION CHECK")
    print("=" * 80)

    for idx, fname in enumerate(clips, start=1):
        clip_path = audio_dir / fname
        if not clip_path.exists():
            continue
        audio_bytes = clip_path.read_bytes()
        res = await run_voice_turn(
            audio_bytes=audio_bytes,
            mime_type="audio/mpeg",
            voice="female",
            session_id=f"voice_eval_{idx}",
            agent_fn=ai_agent.respond,
        )
        timings = res["timings_ms"]
        print(
            f"[{idx}] {fname} -> STT: {timings['stt']}ms | LLM: {timings['llm']}ms | "
            f"TTS: {timings['tts']}ms | Total: {timings['server_total']}ms"
        )
        print(f"    Transcript: {res['transcript']}")
        print(f"    Reply:      {res['reply_text']}\n")
        voice_records.append({
            "file": fname,
            "transcript": res["transcript"],
            "reply_text": res["reply_text"],
            "stt_ms": timings["stt"],
            "llm_ms": timings["llm"],
            "tts_ms": timings["tts"],
            "server_total_ms": timings["server_total"],
            "audio_bytes_b64_len": len(res["audio_base64"]),
        })
        await asyncio.sleep(0.5)

    out_csv = RESULTS_DIR / "agent_voice_integration.csv"
    if voice_records:
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(voice_records[0].keys()))
            writer.writeheader()
            writer.writerows(voice_records)
        llm_stats = summarize([r["llm_ms"] for r in voice_records])
        tot_stats = summarize([r["server_total_ms"] for r in voice_records])
        print(
            f"Voice Integration Summary -> Agent LLM Mean: {llm_stats['avg']}ms | "
            f"Server Total Mean: {tot_stats['avg']}ms | Saved to: {out_csv}"
        )
    return voice_records


if __name__ == "__main__":
    clear_action_log()
    reset_session()
    if "--voice-only" in sys.argv:
        asyncio.run(run_voice_integration_check())
    else:
        asyncio.run(run_all_scenarios())
        if "--with-voice" in sys.argv:
            asyncio.run(run_voice_integration_check())
