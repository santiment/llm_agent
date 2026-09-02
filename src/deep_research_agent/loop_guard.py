"""Repeated-identical-call guard (after crush's loop detector).

Fingerprints every completed tool call this turn as sha256((tool, args, result prefix))
and counts how often the MOST RECENT call recurs in the last ``WINDOW`` calls, so the
guard goes quiet as soon as the model changes behavior. ``SOFT_REPEATS`` → nudge (at most
``MAX_LOOP_NUDGES`` per turn); ``HARD_REPEATS`` → jump to ``end``. Stateless; one instance
is shared by the orchestrator and every sub-agent.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .events import emit
from .turn import (LOOP_NUDGE_NAME, count_nudges, current_turn, raw_text, tc_args,
                   tc_id, tc_name)

log = logging.getLogger("deep_research_agent.loop_guard")

WINDOW = 10
SOFT_REPEATS = 3
HARD_REPEATS = 6
MAX_LOOP_NUDGES = 2
_RESULT_PREFIX = 1_000

_NUDGE = (
    "You are repeating the IDENTICAL tool call (same tool, same arguments) and receiving "
    "the identical result. Repeating it again will not produce new information. Change the "
    "arguments, use a different tool, or — if you already have enough — deliver your final "
    "output now: the report via `submit_report` if you are the orchestrator, or your "
    "consolidated findings if you are a sub-agent."
)


def _fingerprints(messages: list) -> list[str]:
    """One sha256 per completed tool call, in order: (name, args, result prefix)."""
    args_by_id: dict[str, tuple[str, dict]] = {}
    prints: list[str] = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                args_by_id[tc_id(tc)] = (tc_name(tc), tc_args(tc))
        elif isinstance(m, ToolMessage):
            name, args = args_by_id.get(
                getattr(m, "tool_call_id", "") or "",
                (getattr(m, "name", "") or "", {}),
            )
            raw = json.dumps([name, args, raw_text(m.content)[:_RESULT_PREFIX]],
                             sort_keys=True, default=str)
            prints.append(hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest())
    return prints


class LoopGuardMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict, runtime) -> dict[str, Any] | None:
        turn = current_turn(state.get("messages") or [])
        window = _fingerprints(turn)[-WINDOW:]
        if not window:
            return None
        repeats = window.count(window[-1])
        if repeats < SOFT_REPEATS:
            return None

        if repeats >= HARD_REPEATS:
            log.warning("LOOP GUARD HARD STOP: last call repeated %d× in the last %d "
                        "calls — ending run", repeats, len(window))
            emit({"type": "status", "state": "loop_halt", "repeats": repeats})
            return {"jump_to": "end"}

        if count_nudges(turn, LOOP_NUDGE_NAME) >= MAX_LOOP_NUDGES:
            return None
        log.warning("LOOP GUARD: last call repeated %d× in the last %d calls — nudging",
                    repeats, len(window))
        emit({"type": "status", "state": "loop_detected", "repeats": repeats})
        return {"messages": [HumanMessage(content=_NUDGE, name=LOOP_NUDGE_NAME)]}
