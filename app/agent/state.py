"""State and response contract definitions for the LangGraph customer service agent."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class PendingAction(TypedDict):
    """Represents a side-effect tool call awaiting explicit user confirmation."""
    action: Literal["send_email", "send_telegram"]
    args: Dict[str, Any]
    preview_ar: str


class AgentState(TypedDict, total=False):
    """LangGraph conversation state persisted across turns per session_id."""
    session_id: str
    messages: Annotated[List[AnyMessage], add_messages]
    customer_name: Optional[str]
    order_id: Optional[str]
    customer_email: Optional[str]
    current_issue: Optional[str]
    dialect: Literal["egyptian", "msa"]
    pending_action: Optional[PendingAction]
    last_action: Optional[str]
    needs_confirmation: bool
    needs_human: bool
    escalation_reason: Optional[str]
    faq_context: str
    faq_top_score: float
    final_response: str
    action_taken: Optional[str]


class AgentResult(TypedDict):
    """Structured response contract returned to the FastAPI Backend."""
    response: str
    action: Optional[str]
    needs_confirmation: bool
    needs_human: bool
    session_id: str
    customer_name: Optional[str]
    order_id: Optional[str]
    latency_ms: float
