"""A tool the orchestrator calls to ask the user clarifying questions up-front.

Emits a ``clarification`` protocol event (the UI renders it as a question card and
re-enables input) and tells the model to stop. The frontend collects the answers and
sends them back on the SAME thread paired with the questions (e.g. ``1. Q: … A: …``),
so the agent has unambiguous Q&A in context — a bare ``"the first"`` with no question is
easy to misread. The tool also echoes the questions into its result as insurance.
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
        # Echo the questions back into the tool result (prose, not just the tool-call
        # args) so they survive in context even if history is trimmed/summarized, and
        # so the model can pair them with the answer that arrives next turn.
        asked = "\n".join(f"{i}. {q}" for i, q in enumerate(qs, 1)) or "(none)"
        return (
            "Delivered these clarifying questions to the user:\n"
            f"{asked}\n\n"
            "STOP NOW: do not call any more tools and do not write a report. End your "
            "turn with a brief one-line note that you are waiting for their answer. Their "
            "reply arrives paired with these questions, so you will have the full Q&A."
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
