import logging
import os
from typing import Optional
from langchain_openai import ChatOpenAI
from groq import AsyncGroq

from .config import LLM_MODEL, LLM_STUB_MODEL

logger = logging.getLogger("speech.stub_agent")

SYSTEM_PROMPT = (
    "أنت موظف خدمة عملاء مهذب ومحترف لمتجر إلكتروني. "
    "أجب باللغة العربية باختصار شديد ومباشر في جملة واحدة إلى جملتين قصيرتين فقط (بحد أقصى 15 كلمة). "
    "لا تستخدم أي رموز تنسيق أو نقاط أو إيموجي."
)

_llm: Optional[ChatOpenAI] = None
_groq_llm: Optional[AsyncGroq] = None


def _get_groq_llm() -> Optional[AsyncGroq]:
    global _groq_llm
    if _groq_llm is None:
        key = os.getenv("GROQ_API_KEY")
        if key:
            _groq_llm = AsyncGroq(api_key=key)
    return _groq_llm


def _get_llm_client() -> Optional[ChatOpenAI]:
    global _llm
    if _llm is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = LLM_MODEL
        if api_key and model:
            _llm = ChatOpenAI(
                model=model,
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                max_tokens=200,
                temperature=0.3,
                timeout=7.0,
            )
    return _llm




def _mock_customer_service_reply(text: str) -> str:
    """
    Deterministic realistic customer service fallback if online LLMs are unreachable or rate limited.
    """
    t = text.lower()
    if any(w in t for w in ["أوردر", "طلب", "طلبي", "order"]):
        return "أهلاً بك! يمكنك متابعة حالة طلبك من خلال رقم الشحنة المرسل لك في رسالة التأكيد، وسيصلك خلال يومين إلى ثلاثة أيام عمل."
    elif any(w in t for w in ["استرجاع", "ارجع", "تبديل", "رجع"]):
        return "يسعدنا مساعدتك، الاسترجاع متاح مجاناً خلال أربعة عشر يوماً من استلام الطلب بشرط وجود المنتج بحالته الأصلية."
    elif any(w in t for w in ["شحن", "توصيل", "مصاريف"]):
        return "مدة التوصيل عادة تستغرق من يومين إلى أربعة أيام، والشحن مجاني للطلبات التي تزيد عن خمس مئة جنيه."
    elif any(w in t for w in ["شكوى", "مشكلة", "مش راضي"]):
        return "نعتذر جداً عن أي إزعاج واجهته. تم تسجيل شكواك وسيقوم فريق الدعم الفني بالتواصل معك خلال أربع وعشرين ساعة لحلها."
    elif any(w in t for w in ["سلام", "مرحبا", "ازيك", "صباح", "مساء"]):
        return "أهلاً وسهلاً بحضرتك في خدمة العملاء! كيف أستطيع مساعدتك اليوم؟"
    else:
        return "شكراً لتواصلك معنا. استفسارك قيد المتابعة وسأكون سعيداً بمساعدتك في أي استفسار آخر بخصوص منتجاتنا وطلباتك."


async def stub_agent(text: str, session_id: str = "default_session") -> str:
    """
    Stub agent simulating Member 2's LangGraph customer service agent.
    Prioritizes OpenRouter, falls back to Groq Allam Arabic LLM, then to mock customer service answers.
    """
    if not text or not text.strip():
        return "لم أسمع استفسارك جيداً، هل يمكنك التوضيح؟"

    # Strategy 1: Try OpenRouter LLM
    llm = _get_llm_client()
    if llm is not None:
        try:
            msg = await llm.ainvoke([("system", SYSTEM_PROMPT), ("human", text)])
            content = msg.content
            if isinstance(content, list):
                content = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
            if content and content.strip():
                return content.strip()
        except Exception as e:
            logger.info(f"OpenRouter stub unavailable ({e}). Falling back to Groq Arabic LLM...")

    # Strategy 2: Fast Groq Arabic LLM (allam-2-7b)
    groq_client = _get_groq_llm()
    if groq_client is not None:
        try:
            resp = await groq_client.chat.completions.create(
                model="allam-2-7b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=40,
                temperature=0.3,
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                return content.strip()
        except Exception as groq_err:
            logger.warning(f"Groq Allam LLM failed: {groq_err}. Using mock reply.")

    # Strategy 3: Deterministic mock fallback
    return _mock_customer_service_reply(text)

