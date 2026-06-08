"""A tool the orchestrator calls to ask the user clarifying questions up-front.

Emits a ``clarification`` protocol event (the UI renders it as a question card and
re-enables input) and tells the model to stop. The user's reply arrives as the
next message on the SAME thread, so the agent then has the Q&A in context and
proceeds to research.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from ..events import emit


def build_clarify_tool() -> StructuredTool:
    async def request_clarification(questions: list[str]) -> str:
        """Ask the user 1-3 short clarifying questions BEFORE researching. Use only
        once, at the very start, when the request is ambiguous (unclear scope,
        timeframe, entity, or goal). After calling this, STOP — do not research."""
        qs = [str(q).strip() for q in (questions or []) if str(q).strip()]
        emit({"type": "clarification", "questions": qs})
        return (
            "Clarifying questions delivered to the user. STOP NOW: do not call any more "
            "tools and do not write a report. End your turn with a brief one-line note "
            "that you are waiting for their answer."
        )

    return StructuredTool.from_function(
        coroutine=request_clarification,
        name="request_clarification",
        description=(
            "Ask the user 1-3 clarifying questions when the request is ambiguous. Use "
            "ONLY at the very start, before any research, and at most once. After "
            "calling it, stop and wait for the user's reply."
        ),
    )
