"""Offline unit tests for the LangGraph Arabic Customer Service AI Agent."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from app.agent import (
    ACTION_LOG,
    ArabicCustomerServiceAgent,
    arun_agent,
    clear_action_log,
    get_session_slots,
    register_email_handler,
    register_telegram_handler,
    reset_session,
    run_agent,
    sanitize_plain_arabic,
)
from app.agent.kb import get_all_faqs, retrieve_faq
from app.agent.memory import extract_slots
from app.agent.prompts import detect_dialect


class _ChatTextReq(BaseModel):
    text: str
    session_id: str = "default_session"


@pytest.fixture(autouse=True)
def clean_agent_state():
    clear_action_log()
    reset_session()
    register_email_handler(None)
    register_telegram_handler(None)
    yield
    clear_action_log()
    reset_session()
    register_email_handler(None)
    register_telegram_handler(None)


# ---------------------------------------------------------------------------
# 1. Knowledge Base & Retrieval Tests
# ---------------------------------------------------------------------------

def test_knowledge_base_size_and_fields():
    faqs = get_all_faqs()
    assert len(faqs) >= 15
    for item in faqs:
        assert item.id
        assert item.question
        assert item.answer
        assert len(item.keywords) >= 2


def test_faq_retrieval_msa_and_egyptian():
    msa_matches = retrieve_faq("ما هي مواعيد العمل لديكم؟")
    assert msa_matches
    assert msa_matches[0][0].id == "faq_working_hours"

    eg_matches = retrieve_faq("عايز أعرف مصاريف الشحن للقاهرة كام؟")
    assert eg_matches
    assert eg_matches[0][0].id == "faq_shipping_fees"

    return_matches = retrieve_faq("ازاي أقدر أرجع المنتج ده؟")
    assert return_matches
    assert return_matches[0][0].id == "faq_return_policy"


def test_faq_retrieval_unknown_topic():
    matches = retrieve_faq("كم سعر تذكرة الطيران من القاهرة إلى باريس؟", min_score=0.35)
    assert matches == []


# ---------------------------------------------------------------------------
# 2. Dialect, Slot Extraction & Plain-Arabic Sanitizer Tests
# ---------------------------------------------------------------------------

def test_detect_dialect():
    assert detect_dialect("عايز أعرف الأوردر بتاعي فين؟") == "egyptian"
    assert detect_dialect("أود معرفة سياسة الاسترجاع المعتمدة لديكم") == "msa"


def test_extract_slots_multi_turn():
    s1 = extract_slots("اسمي أحمد ولدي مشكلة في طلبي")
    assert s1["customer_name"] == "أحمد"
    assert s1["order_id"] is None

    s2 = extract_slots("رقمه ١٢٣٤٥", s1)
    assert s2["customer_name"] == "أحمد"
    assert s2["order_id"] == "12345"

    s3 = extract_slots("بريدي ahmed@example.com", s2)
    assert s3["customer_email"] == "ahmed@example.com"
    assert s3["customer_name"] == "أحمد"
    assert s3["order_id"] == "12345"


def test_sanitize_plain_arabic():
    raw = "**أهلاً بك** في [المتجر](https://example.com) 🎉\n1. الأول\n- الثاني `كود`"
    cleaned = sanitize_plain_arabic(raw)
    assert "**" not in cleaned
    assert "`" not in cleaned
    assert "🎉" not in cleaned
    assert "https://example.com" not in cleaned
    assert "أهلاً بك في المتجر" in cleaned


# ---------------------------------------------------------------------------
# 3. Graph Workflow, Confirmation Gate & Memory Tests (LLM Mocked Offline)
# ---------------------------------------------------------------------------

def test_multi_turn_memory_with_mocked_llm():
    agent = ArabicCustomerServiceAgent()
    sid = "sess_memory_01"

    with patch("app.agent.graph._invoke_llm_with_fallback", new=AsyncMock(return_value=None)):
        r1 = asyncio.run(agent.arun(sid, "اسمي أحمد ولدي مشكلة في طلبي"))
        assert r1["customer_name"] == "أحمد"

        r2 = asyncio.run(agent.arun(sid, "رقمه 12345"))
        assert r2["customer_name"] == "أحمد"
        assert r2["order_id"] == "12345"

        r3 = asyncio.run(agent.arun(sid, "هل يمكن إلغاؤه؟"))
        assert "12345" in r3["response"]
        assert r3["needs_confirmation"] is False
        assert r3["needs_human"] is False

        # Ensure a second session does not leak memory from the first
        other = asyncio.run(agent.arun("sess_other_02", "هل يمكن إلغاؤه؟"))
        assert "12345" not in other["response"]
        assert get_session_slots("sess_other_02").get("order_id") is None


def test_email_confirmation_gate_confirm_and_reject():
    sent_emails = []
    register_email_handler(lambda to, sub, body: sent_emails.append((to, sub, body)) or {"ok": True})

    agent = ArabicCustomerServiceAgent()

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "tc_1",
                "name": "request_send_email",
                "args": {
                    "to": "ahmed@example.com",
                    "subject": "ملخص المشكلة",
                    "body": "ملخص مشكلة الطلب 12345",
                },
            }
        ],
    )

    with patch("app.agent.graph._invoke_llm_with_fallback", new=AsyncMock(return_value=tool_call_msg)):
        # Step 1: Request email -> staged, NOT sent
        res1 = asyncio.run(agent.arun("sess_email_confirm", "أرسل ملخص المشكلة إلى ahmed@example.com"))
        assert res1["needs_confirmation"] is True
        assert res1["action"] == "send_email"
        assert len(sent_emails) == 0
        assert "ahmed@example.com" in res1["response"]

        # Step 2: Confirm -> executes handler once
        res2 = asyncio.run(agent.arun("sess_email_confirm", "نعم"))
        assert res2["needs_confirmation"] is False
        assert res2["action"] == "send_email"
        assert len(sent_emails) == 1
        assert sent_emails[0][0] == "ahmed@example.com"

        # Step 3: New request followed by rejection -> never executes
        res3 = asyncio.run(agent.arun("sess_email_reject", "أرسل ملخص المشكلة إلى ahmed@example.com"))
        assert res3["needs_confirmation"] is True
        assert len(sent_emails) == 1

        res4 = asyncio.run(agent.arun("sess_email_reject", "لأ بلاش"))
        assert res4["needs_confirmation"] is False
        assert res4["action"] is None
        assert len(sent_emails) == 1


def test_email_missing_address_does_not_fabricate():
    agent = ArabicCustomerServiceAgent()
    hallucinated_tool = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "tc_missing",
                "name": "request_send_email",
                "args": {
                    "to": "example@email.com",
                    "subject": "ملخص",
                    "body": "ملخص",
                },
            }
        ],
    )

    with patch("app.agent.graph._invoke_llm_with_fallback", new=AsyncMock(return_value=hallucinated_tool)):
        res = asyncio.run(agent.arun("sess_missing_email", "أرسل ملخص المشكلة إلى بريدي الإلكتروني"))
        assert res["needs_confirmation"] is False
        assert "example@email.com" not in res["response"]
        assert len(ACTION_LOG) == 0


def test_telegram_confirmation_and_execution():
    sent_tg = []
    register_telegram_handler(lambda msg: sent_tg.append(msg) or {"ok": True})

    agent = ArabicCustomerServiceAgent()
    with patch("app.agent.graph._invoke_llm_with_fallback", new=AsyncMock(return_value=None)):
        r1 = asyncio.run(agent.arun("sess_tg", "أرسل رسالة عبر تيليجرام بملخص الشكوى"))
        assert r1["needs_confirmation"] is True
        assert r1["action"] == "send_telegram"
        assert len(sent_tg) == 0

        r2 = asyncio.run(agent.arun("sess_tg", "أيوة تمام"))
        assert r2["needs_confirmation"] is False
        assert r2["action"] == "send_telegram"
        assert len(sent_tg) == 1


def test_human_escalation():
    agent = ArabicCustomerServiceAgent()
    with patch("app.agent.graph._invoke_llm_with_fallback", new=AsyncMock(return_value=None)):
        res = asyncio.run(agent.arun("sess_esc", "أريد التحدث مع موظف بشري فوراً"))
        assert res["needs_human"] is True
        assert res["action"] == "escalate_to_human"
        assert any(entry["action"] == "escalate_to_human" for entry in ACTION_LOG)


def test_public_contracts_and_fastapi_chat_text():
    with patch("app.agent.graph._invoke_llm_with_fallback", new=AsyncMock(return_value=None)):
        sync_res = run_agent(session_id="sync_user", text="ما هي مواعيد العمل؟")
        assert isinstance(sync_res, dict)
        assert "التاسعة" in sync_res["response"]
        assert sync_res["needs_confirmation"] is False

        agent = ArabicCustomerServiceAgent()
        voice_str = asyncio.run(agent.respond("كم مصاريف الشحن للقاهرة؟", "voice_user"))
        assert isinstance(voice_str, str)
        assert "خمسون" in voice_str

        # Test FastAPI /chat/text route contract without triggering Speech Pipeline edge_tts SSL init
        test_app = FastAPI()

        @test_app.post("/chat/text")
        async def _chat_text(req: _ChatTextReq):
            return await arun_agent(session_id=req.session_id, text=req.text)

        with TestClient(test_app) as client:
            resp = client.post(
                "/chat/text",
                json={"session_id": "api_user", "text": "ما هي مواعيد العمل؟"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "response" in data
            assert data["session_id"] == "api_user"
