"""In-flight context compaction — the run shrinks instead of dying.

A LangGraph thread replays every message on every model call, and a long research
turn accumulates tool stubs, sub-agent findings and narration without bound.
``BudgetMiddleware`` *stops* a run at its ceilings; nothing *shrank* one — a run
that outgrew the model's context window died with a provider error. This is the
missing shrink step, adapted from crush's auto-compaction:

  - TRIGGER on the estimated size of the NEXT model call crossing
    ``trigger_tokens``. The estimate anchors on real ``usage_metadata`` when the
    provider reports it and falls back to chars/4. The knob is absolute
    (DRA_COMPACTION_TOKENS) because an OpenRouter slug doesn't carry its context
    window size; 0 disables the feature (crush likewise disables compaction when
    the window is unknown — never compact blind).
  - SUMMARIZE everything before a recent tail on the cheap long-context
    ``utility_model``. The prompt mandates the sections a successor needs —
    findings WITH exact figures and sources, offloaded file paths, gaps, next
    steps — because the final report's citations are rebuilt from them.
  - REPLACE history with ``[summary, anchor, tail]``. The summary is a
    HumanMessage tagged ``COMPACTION_SUMMARY_NAME`` placed BEFORE the anchor (the
    turn's real user message), so ``current_turn()`` and every turn-scoped gate
    keep working unchanged.
  - BOOKKEEP what was dropped: the current turn's summarized-away tool calls and
    tokens go to state (``compacted_tool_calls`` / ``compacted_tokens``), keyed to
    the anchor's message id (``compaction_anchor_id``) so a stale counter from a
    previous turn can never leak into a new one. ``compacted_counts`` is how
    budget.py / citations.py / metering.py add them back — compaction must never
    grant fresh budget.

Failure policy: any summarizer problem logs a warning and compacts NOTHING — the
run continues exactly as if this middleware didn't exist (and may still die on
the provider's context limit, as it would have anyway).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import (AIMessage, HumanMessage, RemoveMessage,
                                     SystemMessage, ToolMessage)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from typing_extensions import NotRequired

from .events import emit
from .turn import (CHARS_PER_TOKEN, COMPACTION_SUMMARY_NAME, current_turn, raw_text,
                   text_of, tokens_in, tool_calls_in, tool_calls_of, turn_anchor_index)

log = logging.getLogger("deep_research_agent.compaction")

# Keep at least this many messages verbatim behind the anchor (never split an
# AIMessage from its ToolMessages — the tail start walks back over orphans).
DEFAULT_KEEP_RECENT = 12
# Don't bother summarizing fewer messages than this — the summary itself plus the
# model call would cost more than it frees.
_MIN_TO_SUMMARIZE = 6
# Transcript bounds for the summarizer call itself (it runs on a flash-class model).
_MAX_ENTRY_CHARS = 2_000
_MAX_TRANSCRIPT_CHARS = 400_000

_SUMMARY_SYSTEM = """You are compressing a deep-research agent's working context so a \
successor agent can continue the research seamlessly. Write a DETAILED HANDOFF SUMMARY \
in markdown with EXACTLY these sections:

## Original Request
The user's research question, plus any clarifications, constraints or scope decisions made.

## Research Progress
For each research unit investigated so far: what was asked, the KEY FINDINGS with their \
exact figures, dates and named entities, and the SOURCE of every finding (the URL for web \
sources; the exact data-source label otherwise). Preserve every number and every source \
attribution — the final report's citations are built from them.

## Offloaded Files
Every /workspace file path mentioned, with what each file contains and what still needs \
to be done with it. Write "None." if there are none.

## Gaps and Contradictions
What could not be determined, numbers that conflict between sources, follow-ups already \
identified as necessary. Write "None." if there are none.

## Next Steps
What remains before the final report can be written, as concrete actions.

Include everything a successor needs and NOTHING about the mechanics of this summary. \
Never invent anything that is not in the transcript."""

_SUMMARY_PREFIX = (
    "[Context was compacted to stay within the model's window. The messages before this "
    "point were replaced by the following summary of the research so far — its figures "
    "and source attributions are authoritative.]\n\n"
)


class CompactionState(AgentState):
    """State extension: what compaction summarized out of the CURRENT turn, so budget
    ceilings and usage accounting still see the whole turn."""
    compacted_tool_calls: NotRequired[int]
    compacted_tokens: NotRequired[int]
    compaction_anchor_id: NotRequired[str]


def compacted_counts(state: dict) -> tuple[int, int]:
    """``(tool_calls, tokens)`` summarized out of the current turn — ``(0, 0)`` unless a
    compaction happened THIS turn. Keyed to the anchor message id: a counter left over
    from a previous turn (new user question, same thread) never inflates the new turn."""
    anchor_id = state.get("compaction_anchor_id")
    if not anchor_id:
        return 0, 0
    msgs = state.get("messages") or []
    i = turn_anchor_index(msgs)
    if i < 0 or getattr(msgs[i], "id", None) != anchor_id:
        return 0, 0
    return int(state.get("compacted_tool_calls") or 0), int(state.get("compacted_tokens") or 0)


def turn_spend(state: dict) -> tuple[int, int]:
    """The current turn's REAL spend as ``(tool_calls, tokens)`` — what is visible in
    the transcript PLUS what compaction summarized away. The one accessor every
    enforcement/classification consumer (budget.py, citations.py) reads, so
    "compaction never grants fresh budget" holds by construction, not by each call
    site remembering to fold the counters in."""
    turn = current_turn(state.get("messages") or [])
    compacted_calls, compacted_tokens = compacted_counts(state)
    return tool_calls_in(turn) + compacted_calls, tokens_in(turn) + compacted_tokens


def _short(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + " …[truncated]"


def _context_estimate(messages: list) -> int:
    """Tokens the NEXT model call will roughly carry. The latest AIMessage with real
    usage_metadata already measured the whole prompt up to itself (input + output), so
    anchor there and add chars/4 for everything after it; chars/4 over everything when
    no usage exists (same fallback philosophy as turn.tokens_in)."""
    base, last = 0, -1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        um = getattr(m, "usage_metadata", None) if isinstance(m, AIMessage) else None
        if isinstance(um, dict) and um.get("input_tokens"):
            base = int(um.get("input_tokens") or 0) + int(um.get("output_tokens") or 0)
            last = i
            break
    chars = sum(len(raw_text(m.content)) for m in messages[last + 1:])
    return base + chars // CHARS_PER_TOKEN


def _transcript(messages: list) -> str:
    """The to-be-summarized messages as a plain-text transcript for the summarizer."""
    lines: list[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        role = type(m).__name__.removesuffix("Message").lower()
        name = getattr(m, "name", None)
        label = f"{role} ({name})" if name else role
        calls = ", ".join(f"{n}({_short(str(a), 120)})" for n, a in tool_calls_of(m))
        body = (f"[requested tool calls: {calls}]\n" if calls else "") + _short(text_of(m.content), _MAX_ENTRY_CHARS)
        lines.append(f"--- {label} ---\n{body}")
    text = "\n".join(lines)
    if len(text) > _MAX_TRANSCRIPT_CHARS:
        # The newest entries matter most for continuing — trim from the oldest end.
        text = "[…oldest messages trimmed…]\n" + text[-_MAX_TRANSCRIPT_CHARS:]
    return text


class ContextCompactionMiddleware(AgentMiddleware):
    state_schema = CompactionState

    def __init__(self, model, *, trigger_tokens: int,
                 keep_recent: int = DEFAULT_KEEP_RECENT) -> None:
        super().__init__()
        self.model = model
        self.trigger_tokens = int(trigger_tokens)
        self.keep_recent = max(1, int(keep_recent))

    # -- pure planning ----------------------------------------------------------

    def _plan(self, messages: list) -> dict[str, Any] | None:
        """Decide whether/where to compact. Returns the partition or None (don't)."""
        if self.trigger_tokens <= 0 or len(messages) < _MIN_TO_SUMMARIZE + self.keep_recent:
            return None
        estimate = _context_estimate(messages)
        if estimate < self.trigger_tokens:
            return None
        anchor_i = turn_anchor_index(messages)
        if anchor_i < 0:
            return None
        anchor = messages[anchor_i]
        if getattr(anchor, "id", None) is None:
            # Unreachable behind LangGraph's add_messages reducer (it assigns every
            # message an id) — but the counters are keyed to this id, so without one
            # we bail rather than mutate a state-owned message to invent it.
            return None
        # Tail = the newest keep_recent messages, but never before the anchor and never
        # starting on a ToolMessage (its AIMessage must stay adjacent or providers reject
        # the transcript) — walk back until the boundary is clean.
        tail_start = max(anchor_i + 1, len(messages) - self.keep_recent)
        while anchor_i + 1 < tail_start < len(messages) and isinstance(messages[tail_start], ToolMessage):
            tail_start -= 1
        summarized = messages[:tail_start]
        # Everything summarized that belongs to the CURRENT turn — its calls/tokens must
        # keep counting against the budget after they leave the transcript.
        zone_current = messages[anchor_i + 1:tail_start]
        if len(summarized) < _MIN_TO_SUMMARIZE:
            return None
        return {
            "estimate": estimate,
            "anchor": anchor,
            "tail": list(messages[tail_start:]),
            "summarized": summarized,
            "dropped_calls": tool_calls_in(zone_current),
            "dropped_tokens": tokens_in(zone_current),
        }

    def _summary_input(self, plan: dict[str, Any]) -> list:
        return [SystemMessage(content=_SUMMARY_SYSTEM),
                HumanMessage(content="Transcript to compress:\n\n" + _transcript(plan["summarized"]))]

    def _apply(self, state: dict, plan: dict[str, Any], summary: str) -> dict[str, Any]:
        anchor = plan["anchor"]  # always carries an id — _plan bailed otherwise
        prev_calls, prev_tokens = compacted_counts(state)
        summary_msg = HumanMessage(content=_SUMMARY_PREFIX + summary,
                                   name=COMPACTION_SUMMARY_NAME)
        kept = 2 + len(plan["tail"])
        log.warning("COMPACTION: ~%s tokens estimated; summarized %d messages, kept %d "
                    "(carrying %d calls / ~%s tokens into the budget counters)",
                    f"{plan['estimate']:,}", len(plan["summarized"]), kept,
                    prev_calls + plan["dropped_calls"],
                    f"{prev_tokens + plan['dropped_tokens']:,}")
        emit({"type": "status", "state": "compacted",
              "tokens_estimate": plan["estimate"],
              "messages_summarized": len(plan["summarized"]), "messages_kept": kept})
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), summary_msg, anchor,
                         *plan["tail"]],
            "compacted_tool_calls": prev_calls + plan["dropped_calls"],
            "compacted_tokens": prev_tokens + plan["dropped_tokens"],
            "compaction_anchor_id": anchor.id,
        }

    # -- hooks: sync + async differ ONLY in how the summarizer is invoked ---------

    def _begin(self, state: dict) -> dict[str, Any] | None:
        """Plan + the single "compacting" emit; None when there is nothing to do."""
        plan = self._plan(state.get("messages") or [])
        if plan is not None:
            emit({"type": "status", "state": "compacting",
                  "tokens_estimate": plan["estimate"]})
        return plan

    def _finish(self, state: dict, plan: dict[str, Any],
                response: Any) -> dict[str, Any] | None:
        summary = text_of(response.content).strip()
        if not summary:
            log.warning("COMPACTION: summarizer returned empty text — leaving context as-is")
            return None
        return self._apply(state, plan, summary)

    def before_model(self, state: dict, runtime) -> dict[str, Any] | None:
        plan = self._begin(state)
        if plan is None:
            return None
        try:
            response = self.model.invoke(self._summary_input(plan))
        except Exception as exc:
            log.warning("COMPACTION: summarizer failed (%s) — leaving context as-is", exc)
            return None
        return self._finish(state, plan, response)

    async def abefore_model(self, state: dict, runtime) -> dict[str, Any] | None:
        plan = self._begin(state)
        if plan is None:
            return None
        try:
            response = await self.model.ainvoke(self._summary_input(plan))
        except Exception as exc:
            log.warning("COMPACTION: summarizer failed (%s) — leaving context as-is", exc)
            return None
        return self._finish(state, plan, response)
