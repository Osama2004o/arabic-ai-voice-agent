"""Tool definitions and pluggable dispatchers for Email, Telegram, and Human Escalation."""

from __future__ import annotations

import inspect
import logging
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger("agent.tools")

EmailHandler = Callable[[str, str, str], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]
TelegramHandler = Callable[[str], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]

_EMAIL_HANDLER: Optional[EmailHandler] = None
_TELEGRAM_HANDLER: Optional[TelegramHandler] = None

# In-memory audit log of executed actions (useful for verification, tests, and Backend diagnostics)
ACTION_LOG: List[Dict[str, Any]] = []

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def register_email_handler(handler: Optional[EmailHandler]) -> None:
    """Register an SMTP or email API implementation from the Backend Integration layer."""
    global _EMAIL_HANDLER
    _EMAIL_HANDLER = handler


def register_telegram_handler(handler: Optional[TelegramHandler]) -> None:
    """Register a Telegram Bot API implementation from the Backend Integration layer."""
    global _TELEGRAM_HANDLER
    _TELEGRAM_HANDLER = handler


def clear_action_log() -> None:
    """Clear recorded action audit entries (used in unit tests)."""
    ACTION_LOG.clear()


def action_succeeded(payload: Dict[str, Any]) -> bool:
    """True unless the dispatcher reported a delivery failure."""
    return payload.get("mode") != "failed"


def is_valid_email(email: Optional[str]) -> bool:
    return bool(email and EMAIL_REGEX.match(email.strip()))


async def send_email(to: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Execute an already-confirmed email send.
    Delegates to the registered email handler when available; otherwise records a simulated dispatch.
    """
    payload = {
        "action": "send_email",
        "to": to.strip(),
        "subject": subject.strip() or "ملخص محادثة خدمة العملاء",
        "body": body.strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if _EMAIL_HANDLER is not None:
        try:
            result = _EMAIL_HANDLER(payload["to"], payload["subject"], payload["body"])
            if inspect.isawaitable(result):
                result = await result
            payload["mode"] = "live"
            payload["result"] = result
        except Exception as exc:  # never let a delivery failure crash the conversation
            payload["mode"] = "failed"
            payload["result"] = {"status": "failed", "error": getattr(exc, "message", type(exc).__name__)}
            logger.warning("[TOOL FAILED] %s -> %s", payload["action"], payload["result"]["error"])
    else:
        payload["mode"] = "simulated"
        payload["result"] = {"status": "queued", "message": "Email handler not attached; logged locally."}

    ACTION_LOG.append(payload)
    logger.info("[TOOL EXECUTED] send_email -> to=%s mode=%s", payload["to"], payload["mode"])
    return payload


async def send_telegram(message: str) -> Dict[str, Any]:
    """
    Execute an already-confirmed Telegram message send.
    Delegates to the registered Telegram handler when available; otherwise records a simulated dispatch.
    """
    payload = {
        "action": "send_telegram",
        "message": message.strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if _TELEGRAM_HANDLER is not None:
        try:
            result = _TELEGRAM_HANDLER(payload["message"])
            if inspect.isawaitable(result):
                result = await result
            payload["mode"] = "live"
            payload["result"] = result
        except Exception as exc:  # never let a delivery failure crash the conversation
            payload["mode"] = "failed"
            payload["result"] = {"status": "failed", "error": getattr(exc, "message", type(exc).__name__)}
            logger.warning("[TOOL FAILED] %s -> %s", payload["action"], payload["result"]["error"])
    else:
        payload["mode"] = "simulated"
        payload["result"] = {"status": "queued", "message": "Telegram handler not attached; logged locally."}

    ACTION_LOG.append(payload)
    logger.info("[TOOL EXECUTED] send_telegram -> len=%d mode=%s", len(payload["message"]), payload["mode"])
    return payload


def record_escalation(reason: str, customer_name: Optional[str] = None, order_id: Optional[str] = None) -> Dict[str, Any]:
    """Record a human-handoff escalation ticket."""
    payload = {
        "action": "escalate_to_human",
        "reason": reason.strip() or "طلب العميل التحدث مع موظف بشري",
        "customer_name": customer_name,
        "order_id": order_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    ACTION_LOG.append(payload)
    logger.info("[ESCALATION] reason=%s order_id=%s", payload["reason"], order_id)
    return payload


# ---------------------------------------------------------------------------
# LangChain Tool Schemas bound to the LLM (never execute side effects directly)
# ---------------------------------------------------------------------------

class EmailToolInput(BaseModel):
    to: str = Field(description="عنوان البريد الإلكتروني الصحيح الذي ذكره العميل في المحادثة فقط.")
    subject: str = Field(description="موضوع البريد الإلكتروني باللغة العربية.")
    body: str = Field(description="ملخص مختصر ومهذب للمشكلة أو الطلب لإرساله في البريد.")


class TelegramToolInput(BaseModel):
    message: str = Field(description="نص الرسالة المختصر المراد إرساله عبر تيليجرام.")


class EscalateToolInput(BaseModel):
    reason: str = Field(description="سبب تصعيد المحادثة إلى موظف خدمة العملاء البشري.")
    customer_name: str = Field(default="", description="اسم العميل إن كان معروفاً.")
    order_id: str = Field(default="", description="رقم الطلب إن كان معروفاً.")


@tool("request_send_email", args_schema=EmailToolInput)
def request_send_email(to: str, subject: str, body: str) -> str:
    """تجهيز طلب إرسال بريد إلكتروني للعميل لأخذ موافقته وتأكيده قبل الإرسال الفعلي."""
    return f"PENDING_EMAIL:{to}|{subject}|{body}"


@tool("request_send_telegram", args_schema=TelegramToolInput)
def request_send_telegram(message: str) -> str:
    """تجهيز طلب إرسال إشعار أو ملخص عبر تيليجرام لأخذ موافقة العميل قبل الإرسال الفعلي."""
    return f"PENDING_TELEGRAM:{message}"


@tool("escalate_to_human", args_schema=EscalateToolInput)
def escalate_to_human(reason: str, customer_name: str = "", order_id: str = "") -> str:
    """تصعيد المشكلة إلى ممثل خدمة عملاء بشري عند طلب العميل أو تعذر الحل الآلي."""
    return f"ESCALATE:{reason}"


AGENT_TOOLS = [request_send_email, request_send_telegram, escalate_to_human]
