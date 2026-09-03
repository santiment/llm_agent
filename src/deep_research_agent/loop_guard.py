"""Loop guards: repeated identical tool calls (before_model) and runaway output (after_model).

Tool-call loop (after crush's loop detector): fingerprints every completed tool call this
turn as sha256((tool, args, result prefix)) and counts how often the MOST RECENT call recurs
in the last ``WINDOW`` calls, so the guard goes quiet as soon as the model changes behavior.
``SOFT_REPEATS`` → nudge (at most ``MAX_LOOP_NUDGES`` per turn); ``HARD_REPEATS`` → jump to
``end``.

Runaway output: a small model can degenerate into repetition — "BTC2026, BTC2027, …
BTC2259" for thousands of tokens — or overrun the per-call output cap
(``max_output_tokens``). The message is trimmed to its useful head, the model gets one
nudge to continue, and a second runaway in the same turn ends the run.

Stateless; one instance is shared by the orchestrator and every sub-agent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .events import emit
from .turn import (LOOP_NUDGE_NAME, RUNAWAY_NUDGE_NAME, count_nudges, current_turn,
                   raw_text, tc_args, tc_id, tc_name)

log = logging.getLogger("deep_research_agent.loop_guard")

WINDOW = 10
SOFT_REPEATS = 3
HARD_REPEATS = 6
MAX_LOOP_NUDGES = 2
_RESULT_PREFIX = 1_000

# Runaway output. A response shorter than RUNAWAY_MIN_CHARS is never "runaway" (a short
# repetitive list is fine). The tail is judged by its last RUNAWAY_TAIL_SEGMENTS fragments
# (split on line/clause breaks, digits normalized so a counter loop reads as ONE fragment):
# runaway when one fragment is at least RUNAWAY_MIN_REPEATS of them and RUNAWAY_SHARE of the
# tail, or when the tail has almost no distinct fragments at all (a repeating paragraph).
RUNAWAY_MIN_CHARS = 2_000
RUNAWAY_TAIL_SEGMENTS = 300
RUNAWAY_MIN_REPEATS = 60
RUNAWAY_SHARE = 0.6
RUNAWAY_MAX_DISTINCT_SHARE = 0.15
RUNAWAY_KEEP_CHARS = 1_500
MAX_RUNAWAY_NUDGES = 1
_SEGMENT_BREAK = re.compile(r"[\n,;.]+")
_DIGITS = re.compile(r"\d+")

_NUDGE = (
    "You are repeating the IDENTICAL tool call (same tool, same arguments) and receiving "
    "the identical result. Repeating it again will not produce new information. Change the "
    "arguments, use a different tool, or — if you already have enough — deliver your final "
    "output now: the report via `submit_report` if you are the orchestrator, or your "
    "consolidated findings if you are a sub-agent."
)


_RUNAWAY_NUDGE = (
    "Your previous message was cut: it degenerated into repetition (or ran past the output "
    "limit) and the repeated part was removed. Do NOT regenerate it. Continue from where the "
    "useful content ended, briefly: make your next tool call, or deliver your final output "
    "\u2014 the report via `submit_report` if you are the orchestrator, or your consolidated "
    "findings if you are a sub-agent. NEVER hand-type data rows (a time series, a JSON array, "
    "a table): if the data is a tool result already saved to a /workspace file, pass that "
    "file path to `execute` or a recipe \u2014 do not retype it; if it is still in your "
    "context, write it to a file in ONE `execute` with `json.dump`, never by typing rows into "
    "the message. Long material goes into a file, never into one message."
)


def runaway_repetition(text: str) -> str | None:
    """The fragment a runaway tail keeps repeating (digits as ``#``), or None when the text
    reads as varied output. See the RUNAWAY_* constants for the thresholds."""
    if len(text) < RUNAWAY_MIN_CHARS:
        return None
    segments = [_DIGITS.sub("#", seg.strip().strip("`*-_ |")) for seg in _SEGMENT_BREAK.split(text)]
    tail = [seg for seg in segments if seg][-RUNAWAY_TAIL_SEGMENTS:]
    if len(tail) < RUNAWAY_MIN_REPEATS:
        return None
    counts = Counter(tail)
    fragment, n = counts.most_common(1)[0]
    if n >= RUNAWAY_MIN_REPEATS and n / len(tail) >= RUNAWAY_SHARE:
        return fragment
    if len(counts) / len(tail) <= RUNAWAY_MAX_DISTINCT_SHARE:
        return fragment
    return None


# Digit runs are normalized to "#" by the detector, so a fragment made only of "#" and
# separators means the model was typing NUMBERS — rows of a series it had just fetched,
# pasted by hand into a script or file (data that already sits in a saved file).
_NUMERIC_FRAGMENT = re.compile(r"[#\s.,:;|/_\-]*")


def describe_runaway(fragment: str) -> str:
    if _NUMERIC_FRAGMENT.fullmatch(fragment or ""):
        return ("the model was pasting raw numbers inline — data rows that already live in a "
                "saved file")
    return f"the model kept repeating {fragment!r}"


def _cut_at_output_cap(m: AIMessage) -> bool:
    # `in`, not `==`: some OpenRouter streams merge the final chunk's metadata twice
    # ("lengthlength", "stopstop").
    rm = getattr(m, "response_metadata", None) or {}
    return "length" in str(rm.get("finish_reason") or "")


def _trimmed(m: AIMessage, text: str, why: str) -> AIMessage:
    """``m`` with the same id (so it REPLACES itself in state), its text cut to the useful
    head plus a marker, and its tool calls dropped — their arguments were part of the
    runaway, or were left dangling by the cut."""
    head = text[:RUNAWAY_KEEP_CHARS]
    note = f"\n\n[runaway output: {len(text) - len(head):,} characters removed here — {why}]"
    extra = {k: v for k, v in (m.additional_kwargs or {}).items()
             if k not in ("tool_calls", "function_call")}
    return m.model_copy(update={"content": head + note, "tool_calls": [],
                                "invalid_tool_calls": [], "additional_kwargs": extra})


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

    @hook_config(can_jump_to=["model", "end"])
    def after_model(self, state: dict, runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage):
            return None
        text = raw_text(last.content)
        tool_calls = getattr(last, "tool_calls", None) or []
        args_text = json.dumps([tc_args(tc) for tc in tool_calls], default=str) if tool_calls else ""
        fragment = runaway_repetition(text) or runaway_repetition(args_text)
        if fragment is None and not _cut_at_output_cap(last):
            return None
        why = (describe_runaway(fragment) if fragment is not None
               else "the response hit the output cap")
        size = len(text) + len(args_text)
        turn = current_turn(messages)
        if count_nudges(turn, RUNAWAY_NUDGE_NAME) >= MAX_RUNAWAY_NUDGES:
            log.warning("RUNAWAY OUTPUT again (%s, %d chars) — ending run", why, size)
            emit({"type": "status", "state": "runaway_halt", "detail": why, "chars": size})
            return {"jump_to": "end", "messages": [_trimmed(last, text, why)]}
        log.warning("RUNAWAY OUTPUT (%s, %d chars) — trimming and nudging", why, size)
        emit({"type": "status", "state": "runaway_output", "detail": why, "chars": size})
        return {"jump_to": "model",
                "messages": [_trimmed(last, text, why),
                             HumanMessage(content=_RUNAWAY_NUDGE, name=RUNAWAY_NUDGE_NAME)]}
