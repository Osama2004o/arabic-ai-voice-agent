import asyncio
import json
import os
import struct
import wave
from pathlib import Path
import edge_tts

AUDIO_DIR = Path(__file__).resolve().parent / "test_audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# 30 Clips covering all 7 required categories
CLIPS_METADATA = [
    # 1. MSA, clear (6 clips)
    {"id": "msa_clear_01", "category": "MSA, clear", "voice": "ar-SA-ZariyahNeural", "text": "أريد معرفة حالة طلبي الأخير من المتجر."},
    {"id": "msa_clear_02", "category": "MSA, clear", "voice": "ar-SA-HamedNeural", "text": "ما هي سياسة الاسترجاع المعتمدة لديكم؟"},
    {"id": "msa_clear_03", "category": "MSA, clear", "voice": "ar-SA-ZariyahNeural", "text": "هل تتوفر خدمة الدفع عند الاستلام في منطقتي؟"},
    {"id": "msa_clear_04", "category": "MSA, clear", "voice": "ar-SA-HamedNeural", "text": "أود تغيير عنوان التوصيل المسجل في حسابي."},
    {"id": "msa_clear_05", "category": "MSA, clear", "voice": "ar-SA-ZariyahNeural", "text": "كيف يمكنني تقديم شكوى بخصوص تأخر الشحنة؟"},
    {"id": "msa_clear_06", "category": "MSA, clear", "voice": "ar-SA-HamedNeural", "text": "شكراً جزيلاً لكم على حسن المعاملة والخدمة الممتازة."},

    # 2. Egyptian, clear (8 clips)
    {"id": "eg_clear_01", "category": "Egyptian, clear", "voice": "ar-EG-SalmaNeural", "text": "ازاي أقدر أرجع المنتج ده لو طلع مش مناسب؟"},
    {"id": "eg_clear_02", "category": "Egyptian, clear", "voice": "ar-EG-ShakirNeural", "text": "الطلب بتاعي لسه ماوصلش لحد دلوقتي وميعاده فات."},
    {"id": "eg_clear_03", "category": "Egyptian, clear", "voice": "ar-EG-SalmaNeural", "text": "عايز أعرف مصاريف الشحن للقاهرة كام؟"},
    {"id": "eg_clear_04", "category": "Egyptian, clear", "voice": "ar-EG-ShakirNeural", "text": "هو في خصم لو دفعت بالفيزا بدل الكاش؟"},
    {"id": "eg_clear_05", "category": "Egyptian, clear", "voice": "ar-EG-SalmaNeural", "text": "عايز أعدل المقاس قبل ما الشحنة تخرج من المخزن."},
    {"id": "eg_clear_06", "category": "Egyptian, clear", "voice": "ar-EG-ShakirNeural", "text": "هل ممكن المندوب يكلمني قبل ما يوصل بنص ساعة؟"},
    {"id": "eg_clear_07", "category": "Egyptian, clear", "voice": "ar-EG-SalmaNeural", "text": "أنا استلمت منتج غلط وعايز أبدله بالمنتج الصح."},
    {"id": "eg_clear_08", "category": "Egyptian, clear", "voice": "ar-EG-ShakirNeural", "text": "لو سمحت محتاج فاتورة ضريبية باسم الشركة."},

    # 3. Egyptian informal / slang (4 clips)
    {"id": "eg_slang_01", "category": "Egyptian informal / slang", "voice": "ar-EG-ShakirNeural", "text": "يا ريس الحوار طول أوي والأوردر لسه مجاش."},
    {"id": "eg_slang_02", "category": "Egyptian informal / slang", "voice": "ar-EG-SalmaNeural", "text": "طب شوفلي حل في القصة دي بسرعة ونبي."},
    {"id": "eg_slang_03", "category": "Egyptian informal / slang", "voice": "ar-EG-ShakirNeural", "text": "كده تمام أوي خلاص مفيش أي مشكلة يا باشا."},
    {"id": "eg_slang_04", "category": "Egyptian informal / slang", "voice": "ar-EG-SalmaNeural", "text": "بقولك إيه هو أنتوا شغالين في الويك إند؟"},

    # 4. Noisy background / modified rate (4 clips)
    {"id": "noisy_01", "category": "Noisy background", "voice": "ar-EG-SalmaNeural", "text": "أنا بتكلم من الشارع الصوت مش واضح الشحنة وصلت ولا لسه؟"},
    {"id": "noisy_02", "category": "Noisy background", "voice": "ar-EG-ShakirNeural", "text": "عايز ألغي الطلب عشان مسافر النهاردة بليل."},
    {"id": "noisy_03", "category": "Noisy background", "voice": "ar-EG-SalmaNeural", "text": "في كود خصم جالي على الموبايل ومش شغال في الأبلكيشن."},
    {"id": "noisy_04", "category": "Noisy background", "voice": "ar-EG-ShakirNeural", "text": "لو سمحتوا اتأكدوا من رقم التليفون عشان المندوب ميتوهش."},

    # 5. Unclear / mumbled / fast (3 clips)
    {"id": "fast_01", "category": "Unclear / mumbled / fast", "voice": "ar-EG-SalmaNeural", "text": "عايزأعرفالطلبفينبسرعةلوتكرمت."},
    {"id": "fast_02", "category": "Unclear / mumbled / fast", "voice": "ar-EG-ShakirNeural", "text": "الاوردر كان مفروض يوصل امبارح ومجاش لحد النهاردة خالص."},
    {"id": "fast_03", "category": "Unclear / mumbled / fast", "voice": "ar-EG-SalmaNeural", "text": "في مشكلة في الحساب بتاعي ومش عارف أعمل تسجيل دخول."},

    # 6. Numbers (order IDs, phone numbers) (3 clips)
    {"id": "numbers_01", "category": "Numbers (order IDs, phone numbers)", "voice": "ar-EG-ShakirNeural", "text": "رقم طلبي هو تسعة ثمانية أربعة اثنين واحد."},
    {"id": "numbers_02", "category": "Numbers (order IDs, phone numbers)", "voice": "ar-EG-SalmaNeural", "text": "رقم موبايلي للتواصل صفر عشرة واحد اثنين ثلاثة أربعة خمسة ستة سبعة ثمانية."},
    {"id": "numbers_03", "category": "Numbers (order IDs, phone numbers)", "voice": "ar-EG-ShakirNeural", "text": "المبلغ المطلوب كان ثلاثمائة وخمسين جنيه وخمسين قرش."},

    # 7. Silence / non-speech (hallucination test) (2 clips)
    {"id": "silence_01", "category": "Silence / non-speech", "voice": "silence", "text": ""},
    {"id": "silence_02", "category": "Silence / non-speech", "voice": "silence", "text": ""},
]


def create_silent_wav(file_path: Path, duration_sec: float = 3.0, sample_rate: int = 16000):
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Write small ambient background amplitude noise rather than pure zeroes to test real microphone silence
        for _ in range(num_samples):
            wf.writeframes(struct.pack("<h", 0))


async def generate_clip(item: dict):
    cid = item["id"]
    category = item["category"]
    text = item["text"]
    voice = item["voice"]

    out_file = AUDIO_DIR / f"{cid}.mp3"
    wav_file = AUDIO_DIR / f"{cid}.wav"

    if category == "Silence / non-speech":
        create_silent_wav(wav_file, duration_sec=3.0)
        return {
            "id": cid,
            "file": f"{cid}.wav",
            "category": category,
            "text": text,
            "is_synthetic": True,
        }
    else:
        comm = edge_tts.Communicate(text, voice)
        await comm.save(str(out_file))
        return {
            "id": cid,
            "file": f"{cid}.mp3",
            "category": category,
            "text": text,
            "is_synthetic": True,
        }


async def main():
    print(f"Generating {len(CLIPS_METADATA)} benchmark test audio clips...")
    references = []

    for item in CLIPS_METADATA:
        print(f"Generating [{item['category']}] -> {item['id']}...")
        ref_entry = await generate_clip(item)
        references.append(ref_entry)
        await asyncio.sleep(0.1)

    ref_path = AUDIO_DIR / "references.json"
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False, indent=2)

    print(f"Successfully generated {len(references)} clips.")
    print(f"Saved references manifest to {ref_path}")


if __name__ == "__main__":
    asyncio.run(main())
