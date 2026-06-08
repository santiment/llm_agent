"""Budget middleware — the hard backstop against runaway runs.

This middleware enforces two ceilings on the ORCHESTRATOR loop,
checked before each model call (scoped to the current turn):

  - tool-call count (each ToolMessage = one returned call)
  - cumulative model tokens (summed from each AIMessage's ``usage_metadata``)

Two stages, mirroring ``ForceCompletionMiddleware``'s proven nudge pattern:
  - SOFT (>= 75% of either ceiling): inject ONE wrap-up instruction so the model stops
    gathering and calls ``submit_report`` with what it has — a graceful, real partial
    report. Capped at ``MAX_BUDGET_NUDGES`` so it can't loop.
  - HARD (>= the ceiling): jump straight to ``end``. ``ResearchOutputMiddleware`` then
    salvages whatever was gathered. Guaranteed stop.

Scope: the orchestrator. Sub-agents run in their own sub-graphs and do not share this
middleware; the per-call result cap in ``events.py`` and ``mcp_max_concurrency`` bound
them. A dedicated sub-agent budget is a follow-up.

Token accounting reads ``usage_metadata`` (``models.py`` sets ``stream_usage=True`` so it
is present even when streaming). When a model omits it we fall back to a chars/4 estimate
so the ceiling still bites.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage

from .events import emit
from .turn import (
    BUDGET_NUDGE_NAME,
    count_nudges,
    current_turn,
    tokens_in,
    tool_calls_in,
)

log = logging.getLogger("deep_research_agent.budget")

MAX_BUDGET_NUDGES = 2
_SOFT_FRACTION = 0.75

_WRAP_UP = (
    "You have reached the research budget for this run ({reason}). STOP gathering data now "
    "— do not call any more research tools or spawn sub-agents. Immediately call "
    "`submit_report(report_markdown=...)` with a complete report built from what you have "
    "ALREADY gathered: aggregate and summarize the findings; do NOT transcribe raw rows."
)


class BudgetMiddleware(AgentMiddleware):
    def __init__(self, *, max_tool_calls: int, max_total_tokens: int) -> None:
        super().__init__()
        self.max_tool_calls = max_tool_calls
        self.max_total_tokens = max_total_tokens

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict, runtime) -> dict[str, Any] | None:
        turn = current_turn(state.get("messages") or [])
        calls = tool_calls_in(turn)
        tokens = tokens_in(turn)

        over_hard = calls >= self.max_tool_calls or tokens >= self.max_total_tokens
        over_soft = (
            calls >= self.max_tool_calls * _SOFT_FRACTION
            or tokens >= self.max_total_tokens * _SOFT_FRACTION
        )
        if not over_soft:
            return None

        # Name the binding ceiling (which knob to raise) for the log + UI status event.
        which = (
            "tool_calls"
            if calls / self.max_tool_calls >= tokens / max(1, self.max_total_tokens)
            else "tokens"
        )
        reason = f"{calls}/{self.max_tool_calls} tool calls, ~{tokens:,}/{self.max_total_tokens:,} tokens"
        if over_hard:
            # Guaranteed stop. after_agent salvages what was gathered; we add no message
            # so the salvage picks the model's last real text, not a synthetic stub.
            log.warning("BUDGET HARD STOP (%s): %s — ending run", which, reason)
            emit(
                {
                    "type": "status",
                    "state": "budget_halt",
                    "reason": which,
                    "tool_calls": calls,
                    "tokens": tokens,
                }
            )
            return {"jump_to": "end"}

        # SOFT: ask the model to wrap up and deliver — once (capped), then let the hard
        # ceiling stop it if it ignores us.
        nudges = count_nudges(turn, BUDGET_NUDGE_NAME)
        if nudges >= MAX_BUDGET_NUDGES:
            log.warning(
                "BUDGET SOFT: nudge cap reached (%s); awaiting hard stop", reason
            )
            return None
        log.warning("BUDGET SOFT (%s): nudging to wrap up — %s", which, reason)
        emit(
            {
                "type": "status",
                "state": "budget_soft",
                "reason": which,
                "tool_calls": calls,
                "tokens": tokens,
            }
        )
        return {
            "messages": [
                HumanMessage(
                    content=_WRAP_UP.format(reason=reason), name=BUDGET_NUDGE_NAME
                )
            ]
        }
