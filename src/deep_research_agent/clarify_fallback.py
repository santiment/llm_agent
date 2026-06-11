"""Make user elicitation consistent in the UI.

The intended elicitation channel is the ``request_clarification`` tool, which emits a
``clarification`` event the UI renders as a highlighted question card. But a model may
instead just *narrate* the questions as a plain assistant message (no tool call) — then
no event fires and the UI shows plain text. This middleware closes that gap: when the
orchestrator ends a turn BEFORE doing any research with a question-bearing text message
and did NOT call the tool, it emits the same ``clarification`` event, so the card always
appears regardless of model. Detection is deterministic (here, not in the model), so it
works even with models that under-use tools.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from .completion import _called
from .events import emit
from .turn import current_turn, did_research_work, text_of

log = logging.getLogger("deep_research_agent.clarify")

_MAX_QUESTIONS = 5


def _extract_questions(text: str) -> list[str]:
    """Pull user-facing questions out of a narrated message: split on newlines and
    sentence boundaries, keep the chunks that end in ``?``, and drop leading list
    markers / numbering / a ``preamble:`` lead-in. Returns ``[]`` when there are none,
    so a declarative direct answer (no ``?``) is never mistaken for an elicitation."""
    if not text or "?" not in text:
        return []
    out: list[str] = []
    for line in text.splitlines():
        for chunk in re.split(r"(?<=[.!?])\s+", line.strip()):
            q = chunk.strip().lstrip("-*•0123456789.) ").strip()
            if ":" in q:  # drop a "Before I research:" style lead-in
                q = q.rsplit(":", 1)[-1].strip()
            if q.endswith("?") and len(q) > 5:
                out.append(q)
    return out[:_MAX_QUESTIONS]


class ClarificationFallbackMiddleware(AgentMiddleware):
    """Emit a ``clarification`` event when the model asks the user questions in plain
    text (pre-research) instead of calling ``request_clarification``."""

    def after_model(self, state: dict, runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or getattr(last, "tool_calls", None):
            return None
        turn = current_turn(messages)
        # Tool path already emitted the event, or this is a report / mid-research stop.
        if _called(turn, "request_clarification") or _called(turn, "submit_report"):
            return None
        if did_research_work(turn):
            return None
        content = text_of(last.content)
        questions = _extract_questions(content)
        if not questions:
            return None
        emit({"type": "clarification", "questions": questions})
        return None


class ClarificationGuardMiddleware(AgentMiddleware):
    """Clarification is a TRIAGE-only step (before research). This blocks
    ``request_clarification`` once research has started this turn — so a weak model that
    melts down mid-run (e.g. confused by empty sub-agent results) can't pop a nonsensical
    clarification card after minutes of work. The tool never executes, so no
    ``clarification`` event is emitted; the model is told to finish instead.
    """

    async def awrap_tool_call(self, request, handler):
        # langchain renamed this field `call` -> `tool_call`; read both defensively.
        call = getattr(request, "tool_call", None) or getattr(request, "call", None) or {}
        name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
        if name != "request_clarification":
            return await handler(request)
        state = getattr(request, "state", None)
        messages = state.get("messages") if isinstance(state, dict) else None
        if not did_research_work(current_turn(messages or [])):
            return await handler(request)  # legitimate up-front clarification — allow it
        call_id = call.get("id", "") if isinstance(call, dict) else getattr(call, "id", "")
        log.warning("CLARIFY GUARD: blocked request_clarification after research began")
        return ToolMessage(
            content=(
                "You have ALREADY started researching this turn, so it is too late to ask "
                "the user to clarify — no question was shown to them. Do NOT ask the user "
                "anything. Finish with the data you have; if a sub-agent returned empty, "
                "gather that piece yourself, then deliver the report via `submit_report`."),
            tool_call_id=call_id or "", name="request_clarification")
