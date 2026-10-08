"""Guard the Phase-0 runaway backstops added after a run did ~500 MCP calls / ~7M
tokens and emitted a ~1,800-page row dump.

Three independent guards are pinned here:
  - ``cap_result`` bounds a single tool result before it enters context (events.py).
  - ``BudgetMiddleware`` enforces cumulative tool-call + token ceilings: a soft wrap-up
    nudge (capped) then a hard jump to ``end`` (budget.py).
  - ``current_turn`` must not treat the budget nudge as a new user turn (turn.py) — else
    the cap's own counter resets and never bites.

Runs with plain Python (``python tests/test_budget_caps.py``) — no pytest needed — and is
also pytest-discoverable.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research_agent.budget import (
    MAX_BUDGET_NUDGES,
    BudgetMiddleware,
)
from deep_research_agent.events import cap_result
from deep_research_agent.turn import (
    BUDGET_NUDGE_NAME,
    current_turn,
    tokens_in as _tokens_in,
    tool_calls_in as _tool_calls_in,
)


def _tool_msgs(n: int) -> list:
    return [ToolMessage(f"r{i}", tool_call_id=str(i)) for i in range(n)]


def _is_budget_nudge(update: dict) -> bool:
    msgs = (update or {}).get("messages") or []
    return any(getattr(m, "name", None) == BUDGET_NUDGE_NAME for m in msgs)


def test_cap_result_rows_chars_and_noop() -> None:
    capped, note = cap_result(list(range(10)), max_rows=3)
    assert len(capped) == 4 and note, (capped, note)  # 3 kept + 1 sentinel
    assert capped[-1].get("_truncated"), capped[-1]

    capped, note = cap_result("x" * 100, max_chars=10)
    assert capped.startswith("x" * 10) and "truncated" in capped and note

    assert cap_result("short", max_chars=100) == ("short", None)   # under limit
    assert cap_result([1, 2], max_rows=5) == ([1, 2], None)         # under limit
    assert cap_result("anything", max_chars=0) == ("anything", None)  # disabled


def test_token_and_call_counters() -> None:
    msgs = [
        HumanMessage("q"),
        AIMessage("a", usage_metadata={"input_tokens": 4, "output_tokens": 6, "total_tokens": 10}),
        ToolMessage("res", tool_call_id="1"),
        AIMessage("b", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
    ]
    assert _tool_calls_in(msgs) == 1
    assert _tokens_in(msgs) == 12


def test_budget_nudge_is_not_a_turn_boundary() -> None:
    msgs = [HumanMessage("real"), AIMessage("a"),
            HumanMessage("nudge", name=BUDGET_NUDGE_NAME), AIMessage("b")]
    assert current_turn(msgs)[0].content == "real", current_turn(msgs)


def test_under_budget_does_nothing() -> None:
    mw = BudgetMiddleware(max_tool_calls=10, max_total_tokens=10_000)
    state = {"messages": [HumanMessage("q"), *_tool_msgs(2)]}
    assert mw.before_model(state, None) is None


def test_soft_calls_nudges_then_stops_nudging() -> None:
    # soft = 75% of 4 = 3; hard = 4. Three calls -> soft (nudge), still < hard.
    mw = BudgetMiddleware(max_tool_calls=4, max_total_tokens=10_000)
    turn = [HumanMessage("q"), *_tool_msgs(3)]
    update = mw.before_model({"messages": turn}, None)
    assert _is_budget_nudge(update), update
    assert "jump_to" not in update

    # Already nudged MAX times -> stop nudging (let the hard cap stop it), still < hard.
    turn_nudged = turn + [HumanMessage("n", name=BUDGET_NUDGE_NAME)] * MAX_BUDGET_NUDGES
    assert mw.before_model({"messages": turn_nudged}, None) is None


def test_hard_calls_jumps_to_end() -> None:
    mw = BudgetMiddleware(max_tool_calls=4, max_total_tokens=10_000)
    update = mw.before_model({"messages": [HumanMessage("q"), *_tool_msgs(4)]}, None)
    assert update == {"jump_to": "end"}, update


class _Clock:
    def __init__(self, elapsed):
        self.elapsed = elapsed

    def elapsed_s(self):
        return self.elapsed


def test_time_ceiling_soft_then_hard(capture_events) -> None:
    # Calls and tokens far under budget; only the clock is binding.
    mw = BudgetMiddleware(max_tool_calls=1_000, max_total_tokens=10**9,
                          max_run_seconds=1_000, meter=_Clock(800))
    update = mw.before_model({"messages": [HumanMessage("q")]}, None)
    assert _is_budget_nudge(update) and "jump_to" not in update
    assert "13/17 minutes" in update["messages"][0].content
    soft = [e for e in capture_events if e.get("state") == "budget_soft"]
    assert soft and soft[0]["reason"] == "time" and soft[0]["elapsed_s"] == 800

    mw.meter.elapsed = 1_000
    assert mw.before_model({"messages": [HumanMessage("q")]}, None) == {"jump_to": "end"}
    assert [e["reason"] for e in capture_events if e.get("state") == "budget_halt"] == ["time"]


def test_no_time_ceiling_without_cap_or_meter() -> None:
    assert BudgetMiddleware(max_tool_calls=10, max_total_tokens=10_000, max_run_seconds=0,
                            meter=_Clock(10**9)).before_model({"messages": [HumanMessage("q")]}, None) is None
    assert BudgetMiddleware(max_tool_calls=10, max_total_tokens=10_000, max_run_seconds=60,
                            meter=None).before_model({"messages": [HumanMessage("q")]}, None) is None


def test_hard_tokens_jumps_to_end() -> None:
    mw = BudgetMiddleware(max_tool_calls=1_000, max_total_tokens=1_000)
    msgs = [HumanMessage("q"),
            AIMessage("a", usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 1_000})]
    update = mw.before_model({"messages": msgs}, None)
    assert update == {"jump_to": "end"}, update



def _cached_step(total: int, cached: int) -> AIMessage:
    return AIMessage("a", usage_metadata={"input_tokens": total - 500, "output_tokens": 500,
                                          "total_tokens": total,
                                          "input_token_details": {"cache_read": cached}})


def test_cached_input_counts_at_a_discount() -> None:
    # Every step re-sends the context; the cached prefix is billed at ~0.1x, so it must not
    # stop a run at full weight. Metering still reports the real total.
    from deep_research_agent.turn import CACHED_INPUT_WEIGHT, budget_tokens, message_tokens

    step = _cached_step(100_000, 90_000)
    assert message_tokens(step) == 100_000
    assert budget_tokens(step) == int(100_000 - (1 - CACHED_INPUT_WEIGHT) * 90_000)  # 19,000
    # OpenAI-shaped response metadata (no usage_metadata) is read too.
    raw = AIMessage("a", response_metadata={"token_usage": {
        "total_tokens": 100_000, "prompt_tokens_details": {"cached_tokens": 90_000}}})
    assert budget_tokens(raw) == budget_tokens(step)
    # 20 such steps: 2M real tokens, but well under a 1M budget's hard stop.
    mw = BudgetMiddleware(max_tool_calls=1_000, max_total_tokens=1_000_000)
    assert mw.before_model({"messages": [HumanMessage("q"), *[_cached_step(100_000, 90_000)
                                                            for _ in range(20)]]}, None) is None


def test_compacted_spend_keeps_both_units() -> None:
    from deep_research_agent.compaction import compacted_budget_tokens, compacted_counts, turn_spend

    anchor = HumanMessage("q", id="a1")
    state = {"messages": [anchor], "compaction_anchor_id": "a1",
             "compacted_tool_calls": 3, "compacted_tokens": 500_000, "compacted_budget_tokens": 80_000}
    assert compacted_counts(state) == (3, 500_000)             # metering: real totals
    assert turn_spend(state) == (3, 80_000)                    # budget: discounted
    legacy = {k: v for k, v in state.items() if k != "compacted_budget_tokens"}
    assert compacted_budget_tokens(legacy) == 500_000          # pre-discount thread: count it all


if __name__ == "__main__":
    test_cap_result_rows_chars_and_noop()
    test_token_and_call_counters()
    test_budget_nudge_is_not_a_turn_boundary()
    test_under_budget_does_nothing()
    test_soft_calls_nudges_then_stops_nudging()
    test_hard_calls_jumps_to_end()
    test_hard_tokens_jumps_to_end()
    test_cached_input_counts_at_a_discount()
    test_compacted_spend_keeps_both_units()
    print("OK — budget caps + result capping verified.")
