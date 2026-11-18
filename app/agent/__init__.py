"""Arabic Customer Service AI Agent package exports."""

from __future__ import annotations

import asyncio
from typing import Optional

from .graph import ArabicCustomerServiceAgent, build_agent_graph, sanitize_plain_arabic
from .memory import clear_session_slots, get_session_slots
from .state import AgentResult, AgentState
from .tools import ACTION_LOG, clear_action_log, register_email_handler, register_telegram_handler

agent = ArabicCustomerServiceAgent()


async def arun_agent(session_id: str, text: str) -> AgentResult:
    """Async structured entrypoint for FastAPI."""
    return await agent.arun(session_id=session_id, text=text)


def run_agent(session_id: str, text: str) -> AgentResult:
    """Synchronous convenience wrapper."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(arun_agent(session_id=session_id, text=text))
    raise RuntimeError("run_agent() cannot be called inside a running event loop; await arun_agent() instead.")


async def respond(text: str, session_id: str = "default_session") -> str:
    """Async plain-text entrypoint matching the Voice Pipeline `AgentFn` contract."""
    return await agent.respond(text=text, session_id=session_id)


def reset_session(session_id: Optional[str] = None) -> None:
    """Clear slot cache for a session (or all sessions if None)."""
    clear_session_slots(session_id)


__all__ = [
    "ACTION_LOG",
    "AgentResult",
    "AgentState",
    "ArabicCustomerServiceAgent",
    "agent",
    "arun_agent",
    "build_agent_graph",
    "clear_action_log",
    "get_session_slots",
    "register_email_handler",
    "register_telegram_handler",
    "reset_session",
    "respond",
    "run_agent",
    "sanitize_plain_arabic",
]
