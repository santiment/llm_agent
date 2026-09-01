"""Repeated-identical-call guard — crush's loop detector, middleware-shaped.

A weak model can wedge into re-issuing the SAME tool call (same tool, same args)
and getting the same result back — burning budget without gaining information.
The per-args permanent-failure memo in ``events.py`` already short-circuits
repeated FAILED calls; this guard catches the successful-but-useless loop (e.g.
re-fetching the identical page or metric forever).

Detection (adapted from crush's ``loop_detection.go``): fingerprint every
COMPLETED call this turn as ``sha256((tool, args, result-prefix))``, then count
how often the MOST RECENT call's fingerprint recurs in the last ``WINDOW``
calls. Keying on the most recent call means the guard goes quiet the moment the
model changes behavior — old repeats sliding out of the window never re-trigger.

Two stages, the codebase's usual pattern:
  - ``SOFT_REPEATS`` identical → inject a break-the-loop nudge (capped at
    ``MAX_LOOP_NUDGES`` per turn so the nudge itself can't loop);
  - ``HARD_REPEATS`` identical → jump to ``end``. On the orchestrator,
    ``ResearchOutputMiddleware`` then salvages what was gathered; on a sub-agent
    the findings-so-far return to the orchestrator, which is still strictly
    better than burning the remaining budget on the same call.

Stateless and turn-scoped (everything is derived from the messages), so ONE
instance is safely shared by the orchestrator and every sub-agent.
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

WINDOW = 10          # look-back over the last N completed calls
SOFT_REPEATS = 3     # identical calls within the window → nudge
HARD_REPEATS = 6     # identical calls within the window → end the run
MAX_LOOP_NUDGES = 2
_RESULT_PREFIX = 1_000  # of the result, enough to distinguish without hashing megabytes

_NUDGE = (
    "You are repeating the IDENTICAL tool call (same tool, same arguments) and receiving "
    "the identical result. Repeating it again will not produce new information. Change the "
    "arguments, use a different tool, or — if you already have enough — deliver your final "
    "output now: the report via `submit_report` if you are the orchestrator, or your "
    "consolidated findings if you are a sub-agent."
)


def _fingerprints(messages: list) -> list[str]:
    """One sha256 per COMPLETED tool call, in completion order: (name, args, result
    prefix). Args come from the requesting AIMessage via tool_call_id; a ToolMessage
    whose request isn't found still fingerprints on (name, result)."""
    args_by_id: dict[str, tuple[str, dict]] = {}
    prints: list[str] = []
    # Single pass: a ToolMessage always follows its requesting AIMessage in the
    # transcript, so its args are registered by the time we reach it.
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
        # Repeats of the LATEST call only — a loop that is still going on right now.
        repeats = window.count(window[-1])
        if repeats < SOFT_REPEATS:
            return None

        if repeats >= HARD_REPEATS:
            log.warning("LOOP GUARD HARD STOP: last call repeated %d× in the last %d "
                        "calls — ending run", repeats, len(window))
            emit({"type": "status", "state": "loop_halt", "repeats": repeats})
            return {"jump_to": "end"}

        if count_nudges(turn, LOOP_NUDGE_NAME) >= MAX_LOOP_NUDGES:
            return None  # already told it twice; let the hard stop end the loop
        log.warning("LOOP GUARD: last call repeated %d× in the last %d calls — nudging",
                    repeats, len(window))
        emit({"type": "status", "state": "loop_detected", "repeats": repeats})
        return {"messages": [HumanMessage(content=_NUDGE, name=LOOP_NUDGE_NAME)]}
