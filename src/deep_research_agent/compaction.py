"""In-flight context compaction: shrink a long transcript instead of dying on the window.

When the estimated size of the next model call crosses ``trigger_tokens`` (absolute, since an
OpenRouter slug does not expose its window; 0 disables), everything before a recent tail is
summarized on the tier's compaction model and the history becomes ``[summary, anchor, tail]``. The
summary is a HumanMessage tagged ``COMPACTION_SUMMARY_NAME`` placed BEFORE the turn's anchor
(the real user message), so ``current_turn()`` keeps working. The dropped tool calls/tokens
are kept in state keyed to the anchor id (``compacted_counts``) so budget/metering still see
the whole turn — compaction never grants fresh budget. Any summarizer problem compacts nothing.
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

DEFAULT_KEEP_RECENT = 12       # messages kept verbatim behind the anchor
_MIN_TO_SUMMARIZE = 6          # fewer messages are not worth a summarizer call
_MAX_ENTRY_CHARS = 2_000       # transcript bounds for the summarizer input
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
    """What compaction summarized out of the CURRENT turn."""
    compacted_tool_calls: NotRequired[int]
    compacted_tokens: NotRequired[int]
    compaction_anchor_id: NotRequired[str]


def compacted_counts(state: dict) -> tuple[int, int]:
    """``(tool_calls, tokens)`` summarized out of the current turn; ``(0, 0)`` unless the
    stored anchor id matches the current turn's anchor (stale counters never leak)."""
    anchor_id = state.get("compaction_anchor_id")
    if not anchor_id:
        return 0, 0
    msgs = state.get("messages") or []
    i = turn_anchor_index(msgs)
    if i < 0 or getattr(msgs[i], "id", None) != anchor_id:
        return 0, 0
    return int(state.get("compacted_tool_calls") or 0), int(state.get("compacted_tokens") or 0)


def turn_spend(state: dict, turn: list | None = None) -> tuple[int, int]:
    """The turn's real ``(tool_calls, tokens)``: transcript plus compacted-away spend.
    The one accessor budget.py and citations.py read; they pass the ``current_turn`` slice
    they already hold so the messages are not walked twice."""
    if turn is None:
        turn = current_turn(state.get("messages") or [])
    compacted_calls, compacted_tokens = compacted_counts(state)
    return tool_calls_in(turn) + compacted_calls, tokens_in(turn) + compacted_tokens


def _short(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + " …[truncated]"


def _context_estimate(messages: list) -> int:
    """Rough tokens of the next model call: the latest AIMessage with usage_metadata already
    measured the prompt up to itself (input + output); add chars/4 for everything after."""
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

    def _plan(self, messages: list) -> dict[str, Any] | None:
        """The partition to compact, or None when there is nothing to do."""
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
            return None  # counters are keyed to this id; never invent one
        # The tail never starts before the anchor and never on a ToolMessage (its AIMessage
        # must stay adjacent or providers reject the transcript).
        tail_start = max(anchor_i + 1, len(messages) - self.keep_recent)
        while anchor_i + 1 < tail_start < len(messages) and isinstance(messages[tail_start], ToolMessage):
            tail_start -= 1
        summarized = messages[:tail_start]
        if len(summarized) < _MIN_TO_SUMMARIZE:
            return None
        zone_current = messages[anchor_i + 1:tail_start]  # this turn's summarized-away work
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
        anchor = plan["anchor"]
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

    def _begin(self, state: dict) -> dict[str, Any] | None:
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
