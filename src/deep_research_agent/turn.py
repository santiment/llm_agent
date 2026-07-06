"""Scope thread messages to the *current* turn.

A LangGraph thread accumulates every message across every run (multi-turn chat).
Middleware that asks "did we deliver a report this turn?" or "what did we submit?"
MUST look only at the current turn — otherwise a follow-up inherits the previous
turn's ``submit_report`` (the agent thinks it is already done, and the prior
report leaks into the new answer).

The current turn = every message from the most recent genuine user message
onward. ``ForceCompletionMiddleware`` injects synthetic ``HumanMessage`` nudges
mid-turn; those are tagged with ``NUDGE_NAME`` so they are not mistaken for the
start of a new user turn (and so the per-turn nudge count self-resets each turn).
"""

from __future__ import annotations

import json
import re
from typing import Iterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

NUDGE_NAME = "dra_completion_nudge"
# Other middlewares' nudge names. Defined here (not in their own modules) to avoid
# circular imports, and folded into the synthetic-name set below so an injected nudge is
# never mistaken for the start of a new user turn. FINDINGS_NUDGE_NAME is injected into
# SUB-AGENT state only (findings_gate.py) — registered here anyway so any middleware
# that ever calls current_turn() on that state can't misread the bounce as a turn start.
BUDGET_NUDGE_NAME = "dra_budget_nudge"
FINDINGS_NUDGE_NAME = "dra_findings_format_nudge"
RESUBMIT_NUDGE_NAME = "dra_resubmit_nudge"
_SYNTHETIC_NUDGE_NAMES = {
    NUDGE_NAME, BUDGET_NUDGE_NAME, FINDINGS_NUDGE_NAME, RESUBMIT_NUDGE_NAME,
}

# Terminal/control tools — invoking these is how a turn *ends*, not "research work".
_TERMINAL_TOOLS = {"submit_report", "request_clarification"}


def current_turn(messages: list) -> list:
    """Messages belonging to the in-progress turn (from the last real user message)."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, HumanMessage) and getattr(m, "name", None) not in _SYNTHETIC_NUDGE_NAMES:
            return messages[i:]
    return list(messages)


def tc_name(tc) -> str:
    """Name of a tool call, whether it is a dict or an object (both shapes occur)."""
    return (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")) or ""


def tc_args(tc) -> dict:
    """Args of a tool call (same dict-or-object duality), always a dict."""
    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
    return args if isinstance(args, dict) else {}


def tool_call_of(request) -> tuple[str, dict, str]:
    """``(name, args, id)`` of a ``wrap_tool_call`` request's tool call.

    langchain renamed the request field ``call`` -> ``tool_call``; read both so a
    version bump can't silently turn a gate into a no-op (it did exactly that once).
    This shim lives HERE, once — every middleware that intercepts tool calls goes
    through it, so the next rename is a one-line fix."""
    call = getattr(request, "tool_call", None) or getattr(request, "call", None) or {}
    if isinstance(call, dict):
        return (call.get("name") or "", call.get("args") or {}, call.get("id") or "")
    return (getattr(call, "name", "") or "", getattr(call, "args", None) or {},
            getattr(call, "id", "") or "")


def tool_calls_of(message) -> Iterator[tuple[str, dict]]:
    """``(name, args)`` for every tool call an AIMessage requested; nothing for any
    other message type. The ONE place that knows tool_calls may be absent and that each
    entry is dict-or-object — every caller that reads a requested call goes through it."""
    if not isinstance(message, AIMessage):
        return
    for tc in getattr(message, "tool_calls", None) or []:
        yield tc_name(tc), tc_args(tc)


def tool_names_in(messages: list) -> Iterator[str]:
    """Every tool invocation visible in ``messages``, as a name: a returned ToolMessage,
    or a tool call the model requested on an AIMessage. ONE definition of "the agent
    invoked a tool", so the questions asked of it below can't drift apart."""
    for m in messages:
        if isinstance(m, ToolMessage):
            yield getattr(m, "name", "") or ""
        else:
            for name, _args in tool_calls_of(m):
                yield name


def called(messages: list, name: str) -> bool:
    """True if a tool with ``name`` was invoked anywhere in the given messages."""
    return any(n == name for n in tool_names_in(messages))


def looks_delivered(content: str) -> bool:
    """True when a final text is itself a delivered report/answer, not a bare intent
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


def did_research_work(messages: list) -> bool:
    """True if the agent took any research action this turn (planning, search, MCP,
    subagent) — i.e. anything beyond the terminal submit/clarify tools.

    This is the line between a *research report* (must be delivered via submit_report)
    and a *direct conversational answer* (a simple question answered from knowledge,
    which legitimately ends the turn as plain text — no report card, no nudging)."""
    return any(n not in _TERMINAL_TOOLS for n in tool_names_in(messages))


_CHARS_PER_TOKEN = 4  # fallback estimate when usage_metadata is absent


def tokens_in(messages: list) -> int:
    """Cumulative model tokens across the turn's AIMessages, with fallbacks (response
    metadata, then a chars/4 estimate) so the count still reflects reality on models that
    omit usage metadata. Shared by BudgetMiddleware (enforcement) and the end-of-run summary
    so both read the same number."""
    total = 0
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        um = getattr(m, "usage_metadata", None)
        t = um.get("total_tokens") if isinstance(um, dict) else None
        if not t:
            rm = getattr(m, "response_metadata", None)
            t = (rm.get("token_usage") or {}).get("total_tokens") if isinstance(rm, dict) else None
        if not t:
            content = m.content if isinstance(m.content, str) else str(m.content)
            t = len(content) // _CHARS_PER_TOKEN
        total += int(t or 0)
    return total


def tool_calls_in(messages: list) -> int:
    """Completed tool calls this turn (one returned ToolMessage == one call)."""
    return sum(1 for m in messages if isinstance(m, ToolMessage))


def count_nudges(messages: list, name: str) -> int:
    """How many synthetic nudge HumanMessages with ``name`` were injected this turn."""
    return sum(1 for m in messages
               if isinstance(m, HumanMessage) and getattr(m, "name", None) == name)


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def is_json_object_dump(text: str) -> bool:
    """True when a message is essentially a raw JSON object (optionally in a ```json
    fence) rather than prose — e.g. a weak orchestrator echoing the sub-agent findings
    schema ({"summary":…, "findings":[…]}) instead of writing a report. Used to steer it
    to a markdown report, and to refuse salvaging the blob AS a report."""
    t = (text or "").strip()
    m = _JSON_FENCE.search(t)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{"):
        return False
    try:
        return isinstance(json.loads(t), dict)
    except ValueError:
        # A streamed/truncated blob won't fully parse — sniff the findings shape instead.
        return bool(re.match(r'\{\s*"(summary|findings)"\s*:', t))


def text_of(content) -> str:
    """Flatten a message's content to plain text. Models reached via OpenRouter may return
    AIMessage.content as a LIST of blocks (reasoning + text, etc.) rather than a str. The
    middleware that decides "did the model stop with a bare intent message?" reads this; if
    it only handled str it would see nothing for a list and silently do nothing — letting a
    mid-research stall end the run with no report and no nudge."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and isinstance(b.get("text"), str):
                parts.append(b["text"])
        return " ".join(parts)
    return ""
