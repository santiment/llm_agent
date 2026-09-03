"""Pre-triage router — one tiny call before the full harness runs.

Every orchestrator step carries ~12k tokens of harness (our prompt, the `task` tool with
three sub-agent descriptions, tool schemas, data-source list). "What's the capital of
Bulgaria?" paid all of it for a ten-token answer. This middleware runs on the FIRST model
step of a fresh user turn: a ~250-token call on the research model decides SIMPLE (answer
from general knowledge, no data, no sources) or RESEARCH. SIMPLE gets a second small call
that writes the answer, and the turn ends there — the orchestrator never sees it. RESEARCH
(and any failure here) falls through to the full agent, whose own TRIAGE still handles
clarifications and follow-ups answerable from the report in context.

The router's one-word verdict carries the ``nostream`` tag so it never shows up in the UI
as "thinking"; the answer streams like any direct reply.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .events import emit
from .turn import current_turn, text_of

log = logging.getLogger("deep_research_agent.triage")

# Longer messages are research by construction (a brief, pasted data, a multi-part ask);
# skip the call rather than pay for an obvious verdict.
MAX_ROUTED_CHARS = 1_500
MAX_CONTEXT_CHARS = 400

ROUTER_PROMPT = """\
You are the triage step in front of a deep-research agent. Decide whether the user's latest \
message can be answered reliably from general knowledge in one to three sentences — with no \
data lookup, no current figures, no sources: greetings, thanks, small talk, definitions ("what \
is CPI?"), unit conversions, well-known stable facts ("what's the capital of Bulgaria?"). That \
is SIMPLE.
Everything else is RESEARCH: anything asking for current or historical data, prices, volumes, \
trends, sentiment, comparisons, analysis, recommendations, a report, or anything that refers \
back to earlier research results or needs the previous answer's content.
When in doubt, RESEARCH. Reply with exactly one word: SIMPLE or RESEARCH."""

ANSWER_PROMPT = """\
Answer the user's message directly from your own knowledge in one to three plain sentences. \
No research, no sources, no headings, no bullet lists, no offer to research further. A \
greeting or thanks gets a brief reply in kind."""


def _previous_question(earlier: list) -> str:
    """The last real user message before this turn, trimmed — so "and its population?" can
    be judged as a follow-up. Synthetic (named) HumanMessages are engine nudges, not users."""
    for m in reversed(earlier):
        if isinstance(m, HumanMessage) and not getattr(m, "name", None):
            return text_of(m.content).strip()[:MAX_CONTEXT_CHARS]
    return ""


def _user_block(question: str, context: str) -> str:
    if context:
        return f"Earlier the user asked: \u00ab{context}\u00bb\n\nLatest message: \u00ab{question}\u00bb"
    return f"Latest message: \u00ab{question}\u00bb"


def parse_verdict(text: str) -> str:
    """``simple`` only on an unambiguous SIMPLE; anything else (RESEARCH, both words, junk)
    is ``research`` — the safe default."""
    t = (text or "").strip().lower()
    if "simple" in t and "research" not in t:
        return "simple"
    return "research"


class TriageRouterMiddleware(AgentMiddleware):
    def __init__(self, model) -> None:
        super().__init__()
        self.router = model.with_config(tags=["nostream"])
        self.answerer = model

    # -- shared -----------------------------------------------------------------------

    def _routable(self, state: dict) -> tuple[str, str] | None:
        """``(question, context)`` when this is the first model step of a fresh user turn
        short enough to be worth a verdict; ``None`` otherwise."""
        messages = state.get("messages") or []
        turn = current_turn(messages)
        if len(turn) != 1 or not isinstance(turn[0], HumanMessage) or getattr(turn[0], "name", None):
            return None
        question = text_of(turn[0].content).strip()
        if not question or len(question) > MAX_ROUTED_CHARS:
            return None
        return question, _previous_question(messages[: len(messages) - len(turn)])

    def _router_input(self, question: str, context: str) -> list:
        return [SystemMessage(ROUTER_PROMPT), HumanMessage(_user_block(question, context))]

    def _answer_input(self, question: str, context: str) -> list:
        return [SystemMessage(ANSWER_PROMPT), HumanMessage(_user_block(question, context))]

    def _finish(self, verdict: str, answer: AIMessage | None) -> dict[str, Any] | None:
        emit({"type": "status", "state": "triage", "detail": verdict})
        if verdict != "simple":
            log.info("TRIAGE: research — running the full agent")
            return None
        if answer is None or getattr(answer, "tool_calls", None) or not text_of(answer.content).strip():
            log.warning("TRIAGE: simple verdict but no usable direct answer — running the full agent")
            return None
        log.info("TRIAGE: simple — answered directly (%d chars)", len(text_of(answer.content)))
        return {"jump_to": "end", "messages": [answer]}

    # -- hooks ------------------------------------------------------------------------

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict, runtime) -> dict[str, Any] | None:
        routable = self._routable(state)
        if routable is None:
            return None
        question, context = routable
        try:
            verdict = parse_verdict(text_of(self.router.invoke(self._router_input(question, context)).content))
            answer = self.answerer.invoke(self._answer_input(question, context)) if verdict == "simple" else None
        except Exception as exc:  # the router must never take a run down
            log.warning("TRIAGE router failed (%s) — running the full agent", exc)
            return None
        return self._finish(verdict, answer)

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: dict, runtime) -> dict[str, Any] | None:
        routable = self._routable(state)
        if routable is None:
            return None
        question, context = routable
        try:
            resp = await self.router.ainvoke(self._router_input(question, context))
            verdict = parse_verdict(text_of(resp.content))
            answer = (await self.answerer.ainvoke(self._answer_input(question, context))
                      if verdict == "simple" else None)
        except Exception as exc:
            log.warning("TRIAGE router failed (%s) — running the full agent", exc)
            return None
        return self._finish(verdict, answer)
