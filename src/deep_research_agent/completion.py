"""Force-completion middleware — prevents premature ReAct termination.

Failure mode: the model emits a prose message like "Now I will compare …" with NO
tool call. That is the agent loop's termination condition, so the run ends mid-research.

The answer is delivered exclusively via the `submit_report` tool. So, scoped to the
CURRENT turn (a thread accumulates messages across runs — see ``turn.current_turn``):
  - If `submit_report` has been called this turn, the work is DONE — never nudge.
  - If `request_clarification` was called this turn, the agent is waiting — never nudge.
  - Otherwise, if the model stopped with no tool call, it quit early: inject a nudge and
    jump back to the model so it either continues researching or calls `submit_report`.
Capped per turn to avoid loops. The cap counts nudge messages in the current turn, so
it resets automatically each new turn (no cross-turn state to carry).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .turn import NUDGE_NAME, current_turn, did_research_work, text_of

log = logging.getLogger("deep_research_agent.completion")

MAX_NUDGES = 4

_NUDGE = (
    "You ended mid-research without delivering the report. Do NOT describe what you will "
    "do next — act now. Either call the appropriate research tool to continue, or, if "
    "research is complete, call `submit_report(report_markdown=...)` with the full report. "
    "A message that only states intent is not allowed."
)


def _tc_name(tc) -> str:
    return (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")) or ""


def _called(messages: list, name: str) -> bool:
    """True if a tool with `name` was invoked anywhere in the given messages."""
    for m in messages:
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == name:
            return True
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                if _tc_name(tc) == name:
                    return True
    return False


def _looks_delivered(content: str) -> bool:
    """True when the final text is itself a delivered report/answer, not a bare intent
    stub. The model is supposed to deliver via `submit_report`, but some models write the
    report as a plain message instead — the citations fallback still surfaces it, so
    nudging it to "deliver the report" just nags an already-answered model into apology
    loops. Heuristic: substantial length, a markdown heading, or a Sources section."""
    t = content.strip()
    if len(t) >= 400:  # a real report is long; an intent stub ("I will now…") is short
        return True
    if re.search(r"(?m)^\s*#{1,3}\s", t):  # markdown heading → a report body
        return True
    if re.search(r"(?im)^\s*#*\s*sources\b", t):  # a Sources section
        return True
    return False


class ForceCompletionMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["model"])
    def after_model(self, state: dict, runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage):
            return None
        # Model is calling tools → the loop continues on its own.
        if getattr(last, "tool_calls", None):
            return None
        # Scope to the current turn: a prior turn's submit_report must NOT count here,
        # or a follow-up would terminate immediately and inherit the old report.
        turn = current_turn(messages)
        if _called(turn, "submit_report") or _called(turn, "request_clarification"):
            return None
        content = text_of(last.content)
        if not content.strip():
            return None
        # A plain-text ending with NO research work this turn is a legitimate direct
        # answer to a simple question — accept it, don't force a report (no nudge spam).
        # Only an ending that follows real research work is a premature mid-research stall.
        if not did_research_work(turn):
            return None
        # The model already produced a substantial report/answer as plain text (the
        # citations fallback will surface it). Don't nag it to "deliver the report" —
        # that is what drives the apology loop. Only short bare-intent stubs get nudged.
        if _looks_delivered(content):
            return None
        # Per-turn nudge cap: count nudges already injected this turn (self-resetting).
        nudges = sum(
            1
            for m in turn
            if isinstance(m, HumanMessage) and getattr(m, "name", None) == NUDGE_NAME
        )
        if nudges >= MAX_NUDGES:
            # Prime "why did research stop early?" cause: the model kept ending with a bare
            # intent message and never called a tool or submit_report, so after MAX_NUDGES we
            # stop forcing and the turn ends with no delivered report.
            log.warning(
                "FORCE-COMPLETION GAVE UP after %d nudges: model kept stopping mid-research "
                "with no tool call / no submit_report; ending turn (content=%r)",
                MAX_NUDGES, content[:120])
            return None  # give up forcing; never loop unbounded

        log.warning(
            "FORCE-COMPLETION nudge %d/%d: model stopped mid-research with no tool call "
            "(content=%r)", nudges + 1, MAX_NUDGES, content[:120])
        return {
            "jump_to": "model",
            "messages": [HumanMessage(content=_NUDGE, name=NUDGE_NAME)],
        }
