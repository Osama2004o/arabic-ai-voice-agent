"""Session memory, LangGraph checkpointer, and Arabic slot extraction."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from .text_utils import ARABIC_INDIC_DIGITS

# Shared LangGraph in-memory checkpointer keyed by thread_id=session_id
CHECKPOINTER = MemorySaver()

# Lightweight per-session metadata cache for fast inspection/reset
_SESSION_SLOTS: Dict[str, Dict[str, Any]] = {}

_EMAIL_FINDER = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_NAME_PATTERNS = [
    re.compile(r"(?:انا\s+)?اسمي\s+([\u0600-\u06FFa-zA-Z]+(?:\s+[\u0600-\u06FFa-zA-Z]+)?)"),
    re.compile(r"معاك(?:م)?\s+([\u0600-\u06FFa-zA-Z]+)"),
    re.compile(r"أنا\s+([\u0600-\u06FFa-zA-Z]+)\s+(?:و|عندي|لدي|محتاج|عايز)"),
]

_NAME_CLAUSE_BREAKS = frozenset({
    "ولدي", "وعندي", "ومعايا", "وبخصوص", "ومشكلتي", "وعايز", "وعايزة",
    "واريد", "وأريد", "واود", "وأود", "ومحتاج", "ومحتاجة", "وبسأل",
})

_NAME_STOP_WORDS = frozenset({
    "و", "عندي", "لدي", "محتاج", "عايز", "عايزة", "اريد", "أريد", "بخصوص",
    "في", "من", "على", "عن", "مشكلة", "طلب", "طلبي", "الطلب",
})

_ORDER_PATTERNS = [
    re.compile(r"(?:رقم\s*(?:الطلب|الأوردر|الاوردر|الشحنة)|رقمه|طلب\s*رقم|اوردر\s*رقم|أوردر\s*رقم|طلب|أوردر|اوردر)\s*[:#]?\s*(\d{3,12})"),
    re.compile(r"#(\d{3,12})"),
]
_STANDALONE_ORDER_RE = re.compile(r"^\s*(\d{4,10})\s*$")

_ISSUE_KEYWORDS = [
    ("تأخير أو تتبع طلب", ("متابعة", "حالة الطلب", "اتأخر", "ماوصلش", "مجاش", "فين الطلب", "الطلب فين")),
    ("إلغاء طلب", ("إلغاء", "الغاء", "الغي", "ألغي", "كنسل")),
    ("استرجاع أو استبدال منتج", ("استرجاع", "ارجع", "أرجع", "تبديل", "استبدال", "منتج غلط", "خاطئ", "مكسور", "تالف")),
    ("تعديل بيانات الطلب", ("تغيير العنوان", "عنوان التوصيل", "تعديل المقاس", "رقم تليفون")),
    ("استفسار عن الشحن والتوصيل", ("الشحن", "مصاريف", "التوصيل", "المندوب")),
    ("فاتورة ضريبية", ("فاتورة ضريبية", "فاتورة")),
    ("شكوى خدمة عملاء", ("شكوى", "مش راضي", "مشكلة كبيرة", "حسبي الله")),
]


def _normalize_digits(text: str) -> str:
    return text.translate(ARABIC_INDIC_DIGITS)


def extract_slots(text: str, existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract customer_name, order_id, customer_email, and current_issue from an Arabic utterance.
    Preserves previously known slots when not overwritten in the current turn.
    """
    prior = existing or {}
    normalized_digits = _normalize_digits(text or "")
    updates: Dict[str, Any] = {}

    # 1. Customer Name
    for pattern in _NAME_PATTERNS:
        m = pattern.search(normalized_digits)
        if m:
            parts = [p for p in m.group(1).split() if p not in _NAME_STOP_WORDS]
            # Stop at words starting with conjunction 'و' followed by issue verbs (e.g. 'ولدي')
            cleaned_parts: List[str] = []
            for p in parts:
                if p in _NAME_CLAUSE_BREAKS:
                    break
                cleaned_parts.append(p)
            if cleaned_parts:
                updates["customer_name"] = " ".join(cleaned_parts[:2])
                break

    # 2. Order ID
    order_id = None
    for pattern in _ORDER_PATTERNS:
        m = pattern.search(normalized_digits)
        if m:
            order_id = m.group(1)
            break
    if not order_id:
        m_standalone = _STANDALONE_ORDER_RE.match(normalized_digits)
        if m_standalone:
            order_id = m_standalone.group(1)
    if order_id:
        updates["order_id"] = order_id

    # 3. Email
    email_match = _EMAIL_FINDER.search(normalized_digits)
    if email_match:
        updates["customer_email"] = email_match.group(0)

    # 4. Issue / Topic
    for issue_label, kws in _ISSUE_KEYWORDS:
        if any(kw in normalized_digits for kw in kws):
            updates["current_issue"] = issue_label
            break

    # Merge without overwriting existing values with None
    merged = {
        "customer_name": updates.get("customer_name") or prior.get("customer_name"),
        "order_id": updates.get("order_id") or prior.get("order_id"),
        "customer_email": updates.get("customer_email") or prior.get("customer_email"),
        "current_issue": updates.get("current_issue") or prior.get("current_issue"),
    }
    return merged


def get_session_slots(session_id: str) -> Dict[str, Any]:
    return dict(_SESSION_SLOTS.get(session_id, {}))


def update_session_slots(session_id: str, slots: Dict[str, Any]) -> None:
    current = _SESSION_SLOTS.get(session_id, {})
    current.update({k: v for k, v in slots.items() if v is not None})
    _SESSION_SLOTS[session_id] = current


def clear_session_slots(session_id: Optional[str] = None) -> None:
    if session_id is None:
        _SESSION_SLOTS.clear()
    else:
        _SESSION_SLOTS.pop(session_id, None)


def trim_messages_for_llm(messages: Sequence[AnyMessage], max_messages: int = 8) -> List[AnyMessage]:
    """
    Keep only the most recent conversational messages (starting on a HumanMessage when possible)
    to keep LLM input tokens minimal and protect the < 1.0s LLM latency budget.
    """
    if len(messages) <= max_messages:
        return list(messages)

    sliced = list(messages[-max_messages:])
    while sliced and not isinstance(sliced[0], HumanMessage):
        sliced.pop(0)
    return sliced or list(messages[-2:])
