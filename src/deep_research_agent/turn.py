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


def _tc_name(tc) -> str:
    return (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")) or ""


def did_research_work(messages: list) -> bool:
    """True if the agent took any research action this turn (planning, search, MCP,
    subagent) — i.e. anything beyond the terminal submit/clarify tools.

    This is the line between a *research report* (must be delivered via submit_report)
    and a *direct conversational answer* (a simple question answered from knowledge,
    which legitimately ends the turn as plain text — no report card, no nudging)."""
    for m in messages:
        if isinstance(m, ToolMessage) and getattr(m, "name", "") not in _TERMINAL_TOOLS:
            return True
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                if _tc_name(tc) not in _TERMINAL_TOOLS:
                    return True
    return False


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
