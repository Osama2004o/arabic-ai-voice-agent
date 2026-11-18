"""LangGraph conversational customer-service workflow for Arabic & Egyptian Arabic."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .kb import format_faq_context, retrieve_faq
from .memory import CHECKPOINTER, extract_slots, trim_messages_for_llm, update_session_slots
from .prompts import build_system_prompt, detect_dialect
from .state import AgentResult, AgentState, PendingAction
from .text_utils import EMOJI_PATTERN, MARKDOWN_CHARS_PATTERN, MARKDOWN_LINK_PATTERN, MULTIPLE_SPACES, URL_PATTERN
from .tools import (
    AGENT_TOOLS,
    is_valid_email,
    record_escalation,
    send_email,
    send_telegram,
)

load_dotenv()

logger = logging.getLogger("agent.graph")

DEFAULT_GROQ_MODEL = os.getenv("AGENT_LLM_MODEL", "qwen/qwen3.8-27b")
DEFAULT_OPENROUTER_MODEL = os.getenv("AGENT_FALLBACK_MODEL") or os.getenv("LLM_MODEL", "openrouter/free")
DEFAULT_DIALECT = os.getenv("AGENT_DEFAULT_DIALECT", "egyptian").lower()

_CONFIRM_TOKENS = frozenset({
    "نعم", "ايوة", "أيوة", "ايوه", "أيوه", "تمام", "موافق", "موافقة",
    "اكيد", "أكيد", "ابعت", "ابعته", "ابعتها", "ارسل", "أرسل", "أرسله",
    "تأكيد", "أكد", "حاضر", "ماشي", "اوكي", "أوكي", "yes", "ok", "confirm",
})

_CONFIRM_PHRASES = ("تمام ابعت", "موافق ابعت", "أكيد ابعت", "نعم أرسل", "ايوه ابعت")

_CANCEL_TOKENS = frozenset({
    "لا", "لأ", "لاء", "الغاء", "إلغاء", "الغي", "ألغي", "بلاش",
    "توقف", "كنسل", "no", "cancel",
})

_CANCEL_PHRASES = ("مش عايز", "مش عايزة", "متبعتش", "لا ترسل", "لا تبعت", "الغي الارسال", "إلغاء الإرسال")

_NUMBERED_LIST_RE = re.compile(r"(?:^|\s)\d+[.)\-]\s+")
_BULLET_RE = re.compile(r"(?:^|\n)\s*[-•*]\s+")
_EMAIL_IN_TEXT_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_PRIMARY_LLM: Optional[ChatOpenAI] = None
_FALLBACK_LLM: Optional[ChatOpenAI] = None


def sanitize_plain_arabic(text: str, max_words: int = 55) -> str:
    """
    Guarantee pure plain Arabic text for TTS pipeline:
    strips markdown, links, bullet/numbered prefixes, and emojis while keeping email addresses intact.
    """
    if not text:
        return ""

    cleaned = MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    cleaned = URL_PATTERN.sub("", cleaned)
    cleaned = MARKDOWN_CHARS_PATTERN.sub("", cleaned)
    cleaned = EMOJI_PATTERN.sub("", cleaned)
    cleaned = _BULLET_RE.sub(" ", cleaned)
    cleaned = _NUMBERED_LIST_RE.sub(" ", cleaned)
    cleaned = MULTIPLE_SPACES.sub(" ", cleaned).strip()

    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]).rstrip("،, ") + "."
    return cleaned


def _get_primary_llm() -> Optional[ChatOpenAI]:
    global _PRIMARY_LLM
    if _PRIMARY_LLM is not None:
        return _PRIMARY_LLM

    provider = os.getenv("AGENT_LLM_PROVIDER", "groq").lower()
    if provider == "groq" and os.getenv("GROQ_API_KEY"):
        _PRIMARY_LLM = ChatOpenAI(
            model=DEFAULT_GROQ_MODEL,
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2,
            max_tokens=140,
            timeout=5.0,
        ).bind_tools(AGENT_TOOLS)
    elif os.getenv("OPENROUTER_API_KEY"):
        _PRIMARY_LLM = ChatOpenAI(
            model=DEFAULT_OPENROUTER_MODEL,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.2,
            max_tokens=140,
            timeout=6.0,
        ).bind_tools(AGENT_TOOLS)
    return _PRIMARY_LLM


def _get_fallback_llm() -> Optional[ChatOpenAI]:
    global _FALLBACK_LLM
    if _FALLBACK_LLM is not None:
        return _FALLBACK_LLM

    provider = os.getenv("AGENT_LLM_PROVIDER", "groq").lower()
    if provider == "groq" and os.getenv("OPENROUTER_API_KEY"):
        _FALLBACK_LLM = ChatOpenAI(
            model=DEFAULT_OPENROUTER_MODEL,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.2,
            max_tokens=140,
            timeout=6.0,
        ).bind_tools(AGENT_TOOLS)
    elif provider != "groq" and os.getenv("GROQ_API_KEY"):
        _FALLBACK_LLM = ChatOpenAI(
            model=DEFAULT_GROQ_MODEL,
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2,
            max_tokens=140,
            timeout=5.0,
        ).bind_tools(AGENT_TOOLS)
    return _FALLBACK_LLM


def _classify_confirmation(text: str) -> Literal["confirm", "cancel", "other"]:
    cleaned = re.sub(r"[؟?!.,،؛:]", " ", (text or "").strip().lower())
    tokens = cleaned.split()
    if not tokens:
        return "other"

    token_set = set(tokens)
    # Cancellation takes priority if explicit negation token/phrase appears
    if token_set & _CANCEL_TOKENS or any(p in cleaned for p in _CANCEL_PHRASES):
        return "cancel"
    if token_set & _CONFIRM_TOKENS or any(p in cleaned for p in _CONFIRM_PHRASES):
        return "confirm"
    return "other"


def _last_user_text(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _build_conversation_summary(state: AgentState) -> str:
    parts: List[str] = []
    if state.get("customer_name"):
        parts.append(f"اسم العميل: {state['customer_name']}")
    if state.get("order_id"):
        parts.append(f"رقم الطلب: {state['order_id']}")
    if state.get("current_issue"):
        parts.append(f"الموضوع: {state['current_issue']}")
    if not parts:
        last_txt = _last_user_text(state)
        parts.append(f"تفاصيل المحادثة: {last_txt[:180]}")
    return " | ".join(parts)


def _deterministic_fallback(state: AgentState, user_text: str) -> Dict[str, Any]:
    """
    Zero-network / rate-limit-resilient fallback using FAQ matching + intent heuristics.
    Ensures tests and degraded network scenarios always return accurate, polite Arabic responses.
    """
    text_lower = user_text.strip()
    dialect = state.get("dialect", "egyptian")
    matches = retrieve_faq(text_lower, top_k=2, min_score=0.35)

    # 1. Explicit escalation
    if any(k in text_lower for k in ("موظف", "بشري", "اكلم حد", "تحدث مع شخص", "تحويل لموظف", "مدير")):
        record_escalation(
            reason="طلب العميل التحدث مع موظف خدمة عملاء بشري",
            customer_name=state.get("customer_name"),
            order_id=state.get("order_id"),
        )
        return {
            "final_response": "حاضر، تم تحويل محادثتك إلى أحد ممثلي خدمة العملاء البشريين وسيتواصل معك في أقرب وقت.",
            "needs_human": True,
            "escalation_reason": "طلب العميل التحدث مع موظف بشري",
            "action_taken": "escalate_to_human",
        }

    # 2. Email request
    if any(k in text_lower for k in ("بريد", "ايميل", "إيميل", "email", "ميل")):
        email_match = _EMAIL_IN_TEXT_RE.search(text_lower)
        target_email = email_match.group(0) if email_match else state.get("customer_email")
        if not is_valid_email(target_email):
            return {
                "final_response": "بالتأكيد، من فضلك اذكر لي عنوان البريد الإلكتروني الذي تريد أن أرسل إليه الملخص.",
                "needs_confirmation": False,
            }
        summary = _build_conversation_summary(state)
        pending: PendingAction = {
            "action": "send_email",
            "args": {
                "to": target_email,
                "subject": "ملخص طلب خدمة العملاء",
                "body": summary,
            },
            "preview_ar": f"هل تريد مني إرسال ملخص المحادثة إلى {target_email}؟",
        }
        return {
            "pending_action": pending,
            "needs_confirmation": True,
            "final_response": pending["preview_ar"],
        }

    # 3. Telegram request
    if any(k in text_lower for k in ("تيليجرام", "تليجرام", "telegram")):
        summary = _build_conversation_summary(state)
        pending = {
            "action": "send_telegram",
            "args": {"message": summary},
            "preview_ar": "بالتأكيد، هل تؤكد إرسال ملخص الطلب عبر تيليجرام الآن؟",
        }
        return {
            "pending_action": pending,
            "needs_confirmation": True,
            "final_response": pending["preview_ar"],
        }

    # 4. Greeting / unclear short input
    if len(text_lower.split()) <= 2 and any(k in text_lower for k in ("سلام", "اهلا", "أهلا", "مرحبا", "ازيك", "صباح", "مساء")):
        name_part = f" يا {state['customer_name']}" if state.get("customer_name") else ""
        return {"final_response": f"أهلاً بك{name_part} في خدمة العملاء، كيف يمكنني مساعدتك اليوم؟"}

    if len(text_lower) < 4 or text_lower in {"مش فاهم", "ايه", "طب", "ها"}:
        return {"final_response": "عذراً لم أفهم طلبك بوضوح، هل يمكنك توضيح سؤالك أو ذكر رقم الطلب لمساعدتك؟"}

    # 5. Follow-up referencing existing order_id
    if state.get("order_id") and any(k in text_lower for k in ("الغاء", "إلغاء", "إلغاؤ", "الغاؤ", "الغي", "ألغي", "ألغ", "الغ", "كنسل", "حالة", "فين", "وصل")):
        oid = state["order_id"]
        name_part = f" يا {state['customer_name']}" if state.get("customer_name") else ""
        if any(k in text_lower for k in ("الغ", "إلغ", "ألغ", "كنسل")):
            return {"final_response": f"تحت أمرك{name_part}، بخصوص الطلب رقم {oid} يمكن إلغاؤه إذا لم يخرج من المخزن بعد، وسأتحقق من حالته مع الفريق."}
        return {"final_response": f"بخصوص طلبك رقم {oid}{name_part}، التوصيل يستغرق من يومين إلى أربعة أيام عمل، ويمكنني متابعته أو تحويلك لموظف إذا تأخر."}

    # 6. FAQ match
    if matches:
        return {"final_response": matches[0][0].answer}

    # 7. Complaint without FAQ hit
    if any(k in text_lower for k in ("شكوى", "مش راضي", "سيء", "زعلان", "تأخير")):
        return {"final_response": "نعتذر جداً عن أي إزعاج واجهته. من فضلك أخبرني برقم الطلب وتفاصيل المشكلة وسنساعدك فوراً أو نحولك لموظف مختص."}

    # 8. Unknown / out-of-scope question
    if dialect == "msa":
        return {"final_response": "عذراً، لا تتوفر لدي معلومة مؤكدة حول هذا الاستفسار في قاعدة المعرفة، هل تود أن أحولك إلى ممثل خدمة العملاء؟"}
    return {"final_response": "آسف جداً، معنديش معلومة مؤكدة عن النقطة دي في قاعدة البيانات، تحب أحولك لموظف خدمة عملاء يساعدك؟"}


# ---------------------------------------------------------------------------
# LangGraph Nodes
# ---------------------------------------------------------------------------

async def prepare_turn_node(state: AgentState) -> Dict[str, Any]:
    user_text = _last_user_text(state)
    slots = extract_slots(user_text, state)
    default_dialect: Literal["egyptian", "msa"] = "msa" if DEFAULT_DIALECT == "msa" else "egyptian"
    dialect = detect_dialect(user_text, default=default_dialect)

    matches = retrieve_faq(user_text, top_k=3)
    faq_context = format_faq_context(matches)
    top_score = matches[0][1] if matches else 0.0

    session_id = state.get("session_id", "default_session")
    update_session_slots(session_id, slots)

    return {
        **slots,
        "dialect": dialect,
        "faq_context": faq_context,
        "faq_top_score": top_score,
        "needs_confirmation": bool(state.get("pending_action")),
        "action_taken": None,
        "final_response": "",
    }


async def confirmation_gate_node(state: AgentState) -> Dict[str, Any]:
    pending = state.get("pending_action")
    if not pending:
        return {}

    user_text = _last_user_text(state)
    decision = _classify_confirmation(user_text)

    if decision == "confirm":
        action = pending["action"]
        args = pending["args"]
        if action == "send_email":
            await send_email(args["to"], args.get("subject", ""), args.get("body", ""))
            reply = f"تم إرسال البريد الإلكتروني بنجاح إلى {args['to']}."
        else:
            await send_telegram(args["message"])
            reply = "تم إرسال الرسالة عبر تيليجرام بنجاح."

        return {
            "pending_action": None,
            "needs_confirmation": False,
            "last_action": action,
            "action_taken": action,
            "final_response": reply,
            "messages": [AIMessage(content=reply)],
        }

    if decision == "cancel":
        reply = "حاضر، تم إلغاء الإرسال ولن يتم إرسال أي رسالة. هل تحتاج مساعدة في شيء آخر؟"
        return {
            "pending_action": None,
            "needs_confirmation": False,
            "last_action": None,
            "action_taken": None,
            "final_response": reply,
            "messages": [AIMessage(content=reply)],
        }

    # User responded with something else while confirmation was pending (e.g. provided a new email or changed topic)
    new_email_match = _EMAIL_IN_TEXT_RE.search(user_text)
    if new_email_match and pending["action"] == "send_email":
        new_email = new_email_match.group(0)
        updated_pending: PendingAction = {
            "action": "send_email",
            "args": {**pending["args"], "to": new_email},
            "preview_ar": f"هل تريد مني إرسال البريد الإلكتروني إلى {new_email}؟",
        }
        return {
            "pending_action": updated_pending,
            "customer_email": new_email,
            "needs_confirmation": True,
            "final_response": updated_pending["preview_ar"],
            "messages": [AIMessage(content=updated_pending["preview_ar"])],
        }

    # Otherwise clear stale pending_action and let the normal agent handle the new request
    return {
        "pending_action": None,
        "needs_confirmation": False,
    }


def _route_after_confirmation(state: AgentState) -> Literal["finalize", "agent_llm"]:
    if state.get("final_response"):
        return "finalize"
    return "agent_llm"


async def _invoke_llm_with_fallback(messages: List[BaseMessage]) -> Optional[AIMessage]:
    for getter, name in ((_get_primary_llm, "primary"), (_get_fallback_llm, "fallback")):
        llm = getter()
        if llm is None:
            continue
        try:
            resp = await llm.ainvoke(messages)
            if isinstance(resp, AIMessage):
                return resp
        except Exception as exc:
            logger.warning("Agent %s LLM call failed: %s", name, exc)
    return None


async def agent_llm_node(state: AgentState) -> Dict[str, Any]:
    user_text = _last_user_text(state)
    system_prompt = build_system_prompt(state)
    trimmed = trim_messages_for_llm(state.get("messages", []), max_messages=8)

    llm_messages: List[BaseMessage] = [SystemMessage(content=system_prompt), *trimmed]
    ai_msg = await _invoke_llm_with_fallback(llm_messages)

    if ai_msg is None:
        fb = _deterministic_fallback(state, user_text)
        reply = fb["final_response"]
        return {
            **fb,
            "messages": [AIMessage(content=reply)],
        }

    # Handle tool calls emitted by the LLM
    if ai_msg.tool_calls:
        tc = ai_msg.tool_calls[0]
        tc_name = tc.get("name", "")
        tc_args = tc.get("args", {}) or {}

        if tc_name == "request_send_email":
            candidate_to = (tc_args.get("to") or "").strip()
            # Guard against LLM fabricating placeholder emails not provided by the user
            known_emails = set(_EMAIL_IN_TEXT_RE.findall(user_text))
            if state.get("customer_email"):
                known_emails.add(state["customer_email"])

            if not is_valid_email(candidate_to) or (known_emails and candidate_to not in known_emails) or not known_emails:
                reply = "بالتأكيد، من فضلك أخبرني بعنوان البريد الإلكتروني الصحيح الذي تريد الإرسال إليه."
                return {
                    "pending_action": None,
                    "needs_confirmation": False,
                    "final_response": reply,
                    "messages": [AIMessage(content=reply)],
                }

            subject = (tc_args.get("subject") or "ملخص طلب خدمة العملاء").strip()
            body = (tc_args.get("body") or _build_conversation_summary(state)).strip()
            preview = f"هل تريد مني إرسال البريد الإلكتروني إلى {candidate_to} بخصوص {subject}؟"
            pending: PendingAction = {
                "action": "send_email",
                "args": {"to": candidate_to, "subject": subject, "body": body},
                "preview_ar": preview,
            }
            return {
                "pending_action": pending,
                "needs_confirmation": True,
                "final_response": preview,
                "messages": [AIMessage(content=preview)],
            }

        if tc_name == "request_send_telegram":
            msg_body = (tc_args.get("message") or _build_conversation_summary(state)).strip()
            preview = "بالتأكيد، هل تؤكد إرسال هذه الرسالة عبر تيليجرام الآن؟"
            pending = {
                "action": "send_telegram",
                "args": {"message": msg_body},
                "preview_ar": preview,
            }
            return {
                "pending_action": pending,
                "needs_confirmation": True,
                "final_response": preview,
                "messages": [AIMessage(content=preview)],
            }

        if tc_name == "escalate_to_human":
            reason = (tc_args.get("reason") or "طلب العميل التحدث مع موظف بشري").strip()
            record_escalation(
                reason=reason,
                customer_name=tc_args.get("customer_name") or state.get("customer_name"),
                order_id=tc_args.get("order_id") or state.get("order_id"),
            )
            reply = "حاضر، سأقوم بتحويل محادثتك الآن إلى أحد ممثلي خدمة العملاء البشريين لمتابعة طلبك."
            return {
                "needs_human": True,
                "escalation_reason": reason,
                "action_taken": "escalate_to_human",
                "final_response": reply,
                "messages": [AIMessage(content=reply)],
            }

    # Normal text reply from LLM
    content = ai_msg.content
    if isinstance(content, list):
        content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)

    cleaned_reply = sanitize_plain_arabic(str(content or ""))
    if not cleaned_reply:
        fb = _deterministic_fallback(state, user_text)
        cleaned_reply = fb["final_response"]
        return {**fb, "messages": [AIMessage(content=cleaned_reply)]}

    return {
        "final_response": cleaned_reply,
        "messages": [AIMessage(content=cleaned_reply)],
    }


async def finalize_node(state: AgentState) -> Dict[str, Any]:
    cleaned = sanitize_plain_arabic(state.get("final_response", ""))
    return {"final_response": cleaned}


def build_agent_graph(checkpointer=CHECKPOINTER):
    """Compile the LangGraph customer service StateGraph."""
    builder = StateGraph(AgentState)
    builder.add_node("prepare_turn", prepare_turn_node)
    builder.add_node("confirmation_gate", confirmation_gate_node)
    builder.add_node("agent_llm", agent_llm_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "prepare_turn")
    builder.add_edge("prepare_turn", "confirmation_gate")
    builder.add_conditional_edges(
        "confirmation_gate",
        _route_after_confirmation,
        {"finalize": "finalize", "agent_llm": "agent_llm"},
    )
    builder.add_edge("agent_llm", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


class ArabicCustomerServiceAgent:
    """
    Unified conversational AI agent interface connecting:
    - Voice Pipeline (`app/speech/pipeline.py`): `await agent.respond(text, session_id) -> str`
    - Backend Text API (`FastAPI`): `await arun_agent(session_id, text) -> AgentResult`
    """

    def __init__(self) -> None:
        self._graph = build_agent_graph()

    async def arun(self, session_id: str, text: str) -> AgentResult:
        t0 = time.perf_counter()
        sid = (session_id or "default_session").strip()
        cleaned_input = (text or "").strip()

        if not cleaned_input:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "response": "لم أسمع استفسارك جيداً، هل يمكنك التوضيح؟",
                "action": None,
                "needs_confirmation": False,
                "needs_human": False,
                "session_id": sid,
                "customer_name": None,
                "order_id": None,
                "latency_ms": latency_ms,
            }

        config = {"configurable": {"thread_id": sid}}
        result_state = await self._graph.ainvoke(
            {
                "session_id": sid,
                "messages": [HumanMessage(content=cleaned_input)],
            },
            config=config,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        pending = result_state.get("pending_action")
        action_name = result_state.get("action_taken")
        if not action_name and pending:
            action_name = pending.get("action")

        return {
            "response": result_state.get("final_response", ""),
            "action": action_name,
            "needs_confirmation": bool(result_state.get("needs_confirmation", False)),
            "needs_human": bool(result_state.get("needs_human", False)),
            "session_id": sid,
            "customer_name": result_state.get("customer_name"),
            "order_id": result_state.get("order_id"),
            "latency_ms": latency_ms,
        }

    async def respond(self, text: str, session_id: str = "default_session") -> str:
        """Async plain-text signature consumed by the Voice Pipeline (`run_voice_turn`)."""
        result = await self.arun(session_id=session_id, text=text)
        return result["response"]
