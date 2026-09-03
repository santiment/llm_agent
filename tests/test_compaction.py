"""Pin in-flight context compaction (compaction.py): the trigger estimate, the
partition (summary BEFORE the anchor, tail never split from its AIMessage), the
budget bookkeeping (compacted counters keyed to the anchor id), and the failure
policy (any summarizer problem compacts NOTHING).

Runs with plain Python (``python tests/test_compaction.py``) — no pytest needed — and
is also pytest-discoverable. No network, no API keys.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from deep_research_agent.budget import BudgetMiddleware
from deep_research_agent.compaction import ContextCompactionMiddleware, compacted_counts
from deep_research_agent.turn import COMPACTION_SUMMARY_NAME, current_turn


class _FakeModel:
    """Stands in for the utility model; records calls, returns a fixed summary."""

    def __init__(self, text: str = "HANDOFF SUMMARY", fail: bool = False) -> None:
        self.text, self.fail, self.calls = text, fail, 0

    def invoke(self, messages):
        self.calls += 1
        if self.fail:
            raise RuntimeError("summarizer down")
        return AIMessage(self.text)

    async def ainvoke(self, messages):
        return self.invoke(messages)


def _turn(pairs: int, *, chars: int = 4_000) -> list:
    """One research turn: an anchor + `pairs` AIMessage/ToolMessage exchanges of
    ~chars/4 estimated tokens per message (no usage_metadata → chars/4 path)."""
    msgs: list = [HumanMessage("research X", id="anchor-1")]
    for i in range(pairs):
        msgs.append(AIMessage("a" * chars, id=f"ai-{i}"))
        msgs.append(ToolMessage("r" * chars, tool_call_id=f"c{i}", id=f"tool-{i}"))
    return msgs


def _mw(model=None, trigger: int = 5_000, keep: int = 4) -> ContextCompactionMiddleware:
    return ContextCompactionMiddleware(model or _FakeModel(),
                                       trigger_tokens=trigger, keep_recent=keep)


def test_below_threshold_is_a_no_op() -> None:
    model = _FakeModel()
    mw = _mw(model, trigger=10_000_000)
    assert mw.before_model({"messages": _turn(10)}, None) is None
    assert model.calls == 0  # didn't even call the summarizer


def test_zero_trigger_disables() -> None:
    assert _mw(trigger=0).before_model({"messages": _turn(10)}, None) is None


def test_compacts_summary_before_anchor_and_counts_dropped_work() -> None:
    msgs = _turn(10)  # ~10 tool calls, ~20 big messages, est ≫ 5k tokens
    update = _mw().before_model({"messages": msgs}, None)
    assert update is not None
    out = update["messages"]
    assert isinstance(out[0], RemoveMessage) and out[0].id == REMOVE_ALL_MESSAGES
    # summary is a synthetic-named HumanMessage placed BEFORE the anchor …
    assert isinstance(out[1], HumanMessage) and out[1].name == COMPACTION_SUMMARY_NAME
    assert "HANDOFF SUMMARY" in out[1].content
    assert out[2].id == "anchor-1"
    # … so current_turn still anchors on the real user message.
    rebuilt = out[1:]
    assert current_turn(rebuilt)[0].id == "anchor-1"
    # keep_recent=4 → tail is the last 4 messages, and it never starts on a ToolMessage
    # (walked back to include the requesting AIMessage if needed).
    tail = out[3:]
    assert not isinstance(tail[0], ToolMessage)
    assert tail[-1] is msgs[-1]
    # Dropped-from-this-turn bookkeeping: everything before the tail was summarized.
    assert update["compaction_anchor_id"] == "anchor-1"
    kept_tool_msgs = sum(1 for m in tail if isinstance(m, ToolMessage))
    assert update["compacted_tool_calls"] == 10 - kept_tool_msgs
    assert update["compacted_tokens"] > 0


def test_tail_never_splits_an_ai_tool_pair() -> None:
    # keep_recent=3 on pair-structured messages lands the naive cut on a ToolMessage.
    update = _mw(keep=3).before_model({"messages": _turn(10)}, None)
    tail = update["messages"][3:]
    assert isinstance(tail[0], AIMessage)
    assert isinstance(tail[1], ToolMessage)


def test_counters_accumulate_across_compactions() -> None:
    state = {"messages": _turn(10), "compacted_tool_calls": 7,
             "compacted_tokens": 9_000, "compaction_anchor_id": "anchor-1"}
    update = _mw().before_model(state, None)
    assert update["compacted_tool_calls"] > 7      # previous + newly dropped
    assert update["compacted_tokens"] > 9_000


def test_stale_counters_from_a_previous_turn_are_discarded() -> None:
    # Same thread, NEW user turn: stored counters key to the OLD anchor id.
    state = {"messages": _turn(10), "compacted_tool_calls": 7,
             "compacted_tokens": 9_000, "compaction_anchor_id": "old-anchor"}
    assert compacted_counts(state) == (0, 0)
    update = _mw().before_model(state, None)
    kept_tool_msgs = sum(1 for m in update["messages"][3:] if isinstance(m, ToolMessage))
    assert update["compacted_tool_calls"] == 10 - kept_tool_msgs  # no stale +7


def test_summarizer_failure_compacts_nothing() -> None:
    assert _mw(_FakeModel(fail=True)).before_model({"messages": _turn(10)}, None) is None


def test_empty_summary_compacts_nothing() -> None:
    assert _mw(_FakeModel(text="  ")).before_model({"messages": _turn(10)}, None) is None


def test_async_path_matches_sync() -> None:
    update = asyncio.run(_mw().abefore_model({"messages": _turn(10)}, None))
    assert update is not None
    assert update["messages"][1].name == COMPACTION_SUMMARY_NAME


def test_usage_metadata_beats_chars_estimate() -> None:
    # Small text, but real usage says the context is huge → must compact.
    msgs = _turn(10, chars=10)
    msgs[9] = AIMessage("a", id="ai-4", usage_metadata={
        "input_tokens": 200_000, "output_tokens": 500, "total_tokens": 200_500})
    assert _mw(trigger=100_000).before_model({"messages": msgs}, None) is not None


def test_budget_still_bites_after_compaction() -> None:
    # 2 visible tool calls + 8 compacted (anchor-matched) = 10 ≥ hard ceiling → end.
    msgs = [HumanMessage("q", id="anchor-1"),
            ToolMessage("r1", tool_call_id="1"), ToolMessage("r2", tool_call_id="2")]
    state = {"messages": msgs, "compacted_tool_calls": 8, "compacted_tokens": 0,
             "compaction_anchor_id": "anchor-1"}
    mw = BudgetMiddleware(max_tool_calls=10, max_total_tokens=10**9)
    assert mw.before_model(state, None) == {"jump_to": "end"}
    # Stale anchor → compacted counts ignored → far under budget.
    state["compaction_anchor_id"] = "other"
    assert mw.before_model(state, None) is None


if __name__ == "__main__":
    test_below_threshold_is_a_no_op()
    test_zero_trigger_disables()
    test_compacts_summary_before_anchor_and_counts_dropped_work()
    test_tail_never_splits_an_ai_tool_pair()
    test_counters_accumulate_across_compactions()
    test_stale_counters_from_a_previous_turn_are_discarded()
    test_summarizer_failure_compacts_nothing()
    test_empty_summary_compacts_nothing()
    test_async_path_matches_sync()
    test_usage_metadata_beats_chars_estimate()
    test_budget_still_bites_after_compaction()
    print("OK — context compaction verified.")
