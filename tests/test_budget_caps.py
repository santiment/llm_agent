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


def test_hard_tokens_jumps_to_end() -> None:
    mw = BudgetMiddleware(max_tool_calls=1_000, max_total_tokens=1_000)
    msgs = [HumanMessage("q"),
            AIMessage("a", usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 1_000})]
    update = mw.before_model({"messages": msgs}, None)
    assert update == {"jump_to": "end"}, update


if __name__ == "__main__":
    test_cap_result_rows_chars_and_noop()
    test_token_and_call_counters()
    test_budget_nudge_is_not_a_turn_boundary()
    test_under_budget_does_nothing()
    test_soft_calls_nudges_then_stops_nudging()
    test_hard_calls_jumps_to_end()
    test_hard_tokens_jumps_to_end()
    print("OK — budget caps + result capping verified.")
