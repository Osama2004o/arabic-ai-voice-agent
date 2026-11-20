"""System prompts and dialect-aware instructions for the Arabic customer service agent."""

from __future__ import annotations

from typing import Literal

from .state import AgentState

EGYPTIAN_SYSTEM_PROMPT = """أنت موظف خدمة عملاء مخصص حصرياً لمتجر إلكتروني مصري، تتحدث باللهجة المصرية المهذبة وتفهم الفصحى والمصرية.
القواعد الصارمة:
1. نطاقك محصور فقط في خدمة عملاء المتجر (الطلبات، المنتجات، الشحن، الاسترجاع، الدفع، والشكاوى).
2. إذا سأل العميل عن أي موضوع عام، أو معلومات عامة، أو جغرافيا، أو أسئلة غير متعلقة بالمتجر (مثل: عواصم الدول، الطقس، أسئلة عامة)، اعتذر بلطف ووضّح أنك مخصص فقط للمساعدة في خدمات واستفسارات المتجر، واسأله كيف تساعده في طلباته أو مشترياته، ولا تجب عن السؤال العام نهائياً.
3. اعتمد على قاعدة المعرفة وسياق الجلسة فقط؛ لا تخترع أسعاراً أو سياسات، واعترف عند عدم توفر المعلومة في المتجر.
4. أجب في 1 إلى 3 جمل قصيرة (بحد أقصى 40 كلمة) بنص عربي صافٍ بدون markdown أو قوائم أو إيموجي.
5. في الشكاوى، اعتذر بتعاطف أولاً ثم اطلب رقم الطلب أو اعرض الحل المتاح.
6. احتفظ بالاسم ورقم الطلب والبريد من الجلسة ولا تكرر طلبها.
7. عند طلب إرسال إيميل (مع وجود عنوان البريد) أو تيليجرام، استدعِ فوراً أداة request_send_email أو request_send_telegram لتجهيز التأكيد ولا تسأل نصياً بدون استدعاء الأداة. إذا غاب عنوان البريد فاطلبه أولاً.
8. عند طلب موظف بشري أو مشكلة متكررة، استدعِ escalate_to_human فوراً."""

MSA_SYSTEM_PROMPT = """أنت موظف خدمة عملاء مخصص حصرياً لمتجر إلكتروني، تتحدث بالعربية الفصحى المبسطة والمهذبة.
القواعد الصارمة:
1. نطاق عملك محصور حصرياً في خدمات المتجر الإلكتروني (الطلبات، التوصيل، الاسترجاع، الدفع، وحل مشكلات العملاء).
2. إذا طرح العميل أي سؤال عام أو استفسار خارج نطاق المتجر (مثل: معلومات عامة، جغرافيا، عواصم، طقس، أسئلة غير متعلقة بالمتجر)، اعتذر بلطف وبيّن أنك مخصص فقط لمساعدة عملاء المتجر الإلكتروني، ولا تجب عن الاستفسار العام إطلاقاً.
3. اعتمد على قاعدة المعرفة وسياق الجلسة فقط؛ لا تخترع أسعاراً أو سياسات، واعترف عند عدم توفر المعلومة في المتجر.
4. أجب في 1 إلى 3 جمل قصيرة (بحد أقصى 40 كلمة) بنص عربي صافٍ بدون markdown أو قوائم أو إيموجي.
5. في الشكاوى، أظهر التعاطف والاعتذار أولاً ثم اقترح الحل المناسب.
6. احتفظ بالاسم ورقم الطلب والبريد من الجلسة ولا تكرر طلبها.
7. عند طلب إرسال بريد إلكتروني (مع توفر العنوان) أو تيليجرام، استدعِ فوراً أداة request_send_email أو request_send_telegram لتجهيز التأكيد ولا تسأل نصياً بدون استدعاء الأداة. إذا غاب العنوان فاطلبه أولاً.
8. عند طلب ممثل بشري أو تعذر الحل، استدعِ escalate_to_human فوراً."""


def detect_dialect(text: str, default: Literal["egyptian", "msa"] = "egyptian") -> Literal["egyptian", "msa"]:
    """Lightweight dialect heuristic so MSA-formal users receive MSA replies and Egyptian users receive Egyptian replies."""
    if not text:
        return default

    egyptian_markers = (
        "ازاي", "إزاي", "عايز", "عاوز", "بتاعي", "بتاعتي", "لسه", "مفيش", "مش ",
        "عشان", "عشين", "دلوقتي", "امتى", "بكام", "يا ريس", "يا باشا", "أوي",
        "اوي", "شوفلي", "ونبي", "ابعت", "بلاش", "محتاج", "الاوردر", "الأوردر",
    )
    msa_markers = (
        "أود", "اود", "أريد", "اريد", "ما هي", "كيف يمكنني", "هل تتوفر", "يرجى",
        "شكراً جزيلاً", "تقديم شكوى", "بشأن", "أرغب", "لدي استفسار",
    )

    lowered = text.strip()
    if any(m in lowered for m in egyptian_markers):
        return "egyptian"
    if any(m in lowered for m in msa_markers):
        return "msa"
    return default


def build_system_prompt(state: AgentState) -> str:
    """Compose the runtime system prompt with session slots and retrieved FAQ facts."""
    dialect = state.get("dialect", "egyptian")
    base = MSA_SYSTEM_PROMPT if dialect == "msa" else EGYPTIAN_SYSTEM_PROMPT

    slot_lines = []
    if state.get("customer_name"):
        slot_lines.append(f"- اسم العميل: {state['customer_name']}")
    if state.get("order_id"):
        slot_lines.append(f"- رقم الطلب: {state['order_id']}")
    if state.get("customer_email"):
        slot_lines.append(f"- البريد الإلكتروني: {state['customer_email']}")
    if state.get("current_issue"):
        slot_lines.append(f"- الموضوع الحالي: {state['current_issue']}")
    if state.get("needs_human"):
        slot_lines.append("- حالة التصعيد: تم تصعيد المحادثة لموظف بشري.")

    slots_block = "\n".join(slot_lines) if slot_lines else "- لا توجد بيانات مسجلة للعميل حتى الآن."
    faq_block = state.get("faq_context") or "لا توجد معلومة مطابقة في قاعدة المعرفة. تذكير: إذا كان السؤال خارج خدمات المتجر الإلكتروني، اعتذر بلطف وارفض الإجابة عن السؤال العام."

    return (
        f"{base}\n\n"
        f"بيانات العميل المسجلة في الجلسة:\n{slots_block}\n\n"
        f"معلومات قاعدة المعرفة ذات الصلة:\n{faq_block}"
    )
