"""Budget middleware — the hard backstop against runaway runs.

This middleware enforces three ceilings on the ORCHESTRATOR loop,
checked before each model call (scoped to the current turn):

  - tool-call count (each ToolMessage = one returned call)
  - cumulative model tokens (summed from each AIMessage's ``usage_metadata``)
  - wall-clock time since the run started (``RunMeter.elapsed_s``) — the other two do
    not bound time: slow sub-agents retrying against a dead sandbox ran 90 minutes
    under both.

Two stages, mirroring ``ForceCompletionMiddleware``'s proven nudge pattern:
  - SOFT (>= 75% of either ceiling): inject ONE wrap-up instruction so the model stops
    gathering and calls ``submit_report`` with what it has — a graceful, real partial
    report. Capped at ``MAX_BUDGET_NUDGES`` so it can't loop.
  - HARD (>= the ceiling): jump straight to ``end``. ``ResearchOutputMiddleware`` then
    salvages whatever was gathered. Guaranteed stop.

Scope: the orchestrator. Sub-agents run in their own sub-graphs and do not share this
middleware; the per-call result cap in ``events.py`` and ``mcp_max_concurrency`` bound
them. A dedicated sub-agent budget is a follow-up.

Token accounting reads ``usage_metadata`` when the provider supplies it (``models.py``
deliberately does NOT set ``stream_usage`` — see the note there about mis-merged trailing
usage chunks). When it is absent we fall back to a chars/4 estimate so the ceiling still
bites.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage

from .compaction import turn_spend
from .events import emit
from .turn import BUDGET_NUDGE_NAME, count_nudges, current_turn

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
    def __init__(self, *, max_tool_calls: int, max_total_tokens: int,
                 max_run_seconds: int = 0, meter=None) -> None:
        super().__init__()
        self.max_tool_calls = max_tool_calls
        self.max_total_tokens = max_total_tokens
        self.max_run_seconds = max_run_seconds  # 0 = no time ceiling
        self.meter = meter  # RunMeter — its clock is the run's wall-clock

    def _elapsed(self) -> float:
        if not self.max_run_seconds or self.meter is None:
            return 0.0
        return float(self.meter.elapsed_s() or 0.0)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict, runtime) -> dict[str, Any] | None:
        turn = current_turn(state.get("messages") or [])
        calls, tokens = turn_spend(state, turn)  # transcript + compacted-away spend
        elapsed = self._elapsed()

        # Fraction of each ceiling used; the largest names the binding knob to raise.
        used = {
            "tool_calls": calls / max(1, self.max_tool_calls),
            "tokens": tokens / max(1, self.max_total_tokens),
            "time": (elapsed / self.max_run_seconds) if self.max_run_seconds else 0.0,
        }
        over_hard = max(used.values()) >= 1.0
        over_soft = max(used.values()) >= _SOFT_FRACTION
        if not over_soft:
            return None

        which = max(used, key=used.get)
        reason = (f"{calls}/{self.max_tool_calls} tool calls, "
                  f"~{tokens:,}/{self.max_total_tokens:,} tokens")
        if self.max_run_seconds:
            reason += f", {elapsed / 60:.0f}/{self.max_run_seconds / 60:.0f} minutes"

        def status(state: str) -> None:
            emit({"type": "status", "state": state, "reason": which,
                  "tool_calls": calls, "tokens": tokens, "elapsed_s": elapsed})

        if over_hard:
            # Guaranteed stop. after_agent salvages what was gathered; we add no message
            # so the salvage picks the model's last real text, not a synthetic stub.
            log.warning("BUDGET HARD STOP (%s): %s — ending run", which, reason)
            status("budget_halt")
            return {"jump_to": "end"}

        # SOFT: ask the model to wrap up and deliver — once (capped), then let the hard
        # ceiling stop it if it ignores us.
        if count_nudges(turn, BUDGET_NUDGE_NAME) >= MAX_BUDGET_NUDGES:
            log.warning("BUDGET SOFT: nudge cap reached (%s); awaiting hard stop", reason)
            return None
        log.warning("BUDGET SOFT (%s): nudging to wrap up — %s", which, reason)
        status("budget_soft")
        return {"messages": [HumanMessage(content=_WRAP_UP.format(reason=reason),
                                          name=BUDGET_NUDGE_NAME)]}
