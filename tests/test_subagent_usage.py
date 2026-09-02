"""Pin the sub-agent usage rollup + cost accounting (metering.py): ``sum_usage``'s
fallback chain matches ``turn.tokens_in``, OpenRouter's actual cost is read
defensively from response metadata, ``SubagentUsageMiddleware`` folds each finished
sub-agent run into the meter, and the run's ``usage`` event carries the fleet
breakdown, the grand total, and the compacted counters.

Runs with plain Python (``python tests/test_subagent_usage.py``) — no pytest needed —
and is also pytest-discoverable.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research_agent import metering
from deep_research_agent.metering import (RunMeter, SubagentUsageMiddleware,
                                          UsageMeterMiddleware, sum_usage)


def test_sum_usage_fallback_chain_and_cost() -> None:
    msgs = [
        HumanMessage("q"),
        # 1: full usage_metadata + OpenRouter cost in response metadata
        AIMessage("a", usage_metadata={"input_tokens": 100, "output_tokens": 50,
                                       "total_tokens": 150},
                  response_metadata={"token_usage": {"total_tokens": 150, "cost": 0.012}}),
        # 2: no usage_metadata → response_metadata token_usage
        AIMessage("b", response_metadata={"token_usage": {"total_tokens": 80}}),
        # 3: nothing at all → chars/4 estimate
        AIMessage("x" * 400),
        # 4: junk cost value must be ignored, not raise
        AIMessage("c", usage_metadata={"input_tokens": 1, "output_tokens": 1,
                                       "total_tokens": 2},
                  response_metadata={"token_usage": {"cost": "n/a"}}),
        ToolMessage("r", tool_call_id="1"),  # not an AIMessage — not counted
    ]
    u = sum_usage(msgs)
    assert u["model_calls"] == 4
    assert u["input_tokens"] == 101 and u["output_tokens"] == 51
    assert u["total_tokens"] == 150 + 80 + 100 + 2
    assert u["cost_usd"] == 0.012


def test_meter_rolls_up_per_role() -> None:
    m = RunMeter()
    m.record_subagent_usage("research-subagent", {
        "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
        "model_calls": 3, "cost_usd": 0.01})
    m.record_subagent_usage("research-subagent", {
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        "model_calls": 1, "cost_usd": 0.002})
    m.record_subagent_usage("extract-subagent", {
        "input_tokens": 7, "output_tokens": 3, "total_tokens": 10,
        "model_calls": 1, "cost_usd": 0.0})
    r = m.subagent_usage["research-subagent"]
    assert r["runs"] == 2 and r["total_tokens"] == 165 and r["model_calls"] == 4
    assert round(r["cost_usd"], 6) == 0.012
    assert m.subagent_usage["extract-subagent"]["runs"] == 1


def test_subagent_middleware_meters_its_own_state() -> None:
    meter = RunMeter()
    mw = SubagentUsageMiddleware(meter, "research-subagent")
    mw.after_agent({"messages": [
        HumanMessage("unit"),
        AIMessage("f", usage_metadata={"input_tokens": 30, "output_tokens": 20,
                                       "total_tokens": 50}),
    ]}, None)
    u = meter.subagent_usage["research-subagent"]
    assert u["runs"] == 1 and u["total_tokens"] == 50 and u["model_calls"] == 1


def test_subagent_middleware_folds_its_own_compaction() -> None:
    # A sub-agent that compacted its own context still spent those tokens — the
    # rollup must include them, same as the orchestrator total.
    meter = RunMeter()
    mw = SubagentUsageMiddleware(meter, "research-subagent")
    mw.after_agent({
        "messages": [
            HumanMessage("unit", id="sub-anchor"),
            AIMessage("f", usage_metadata={"input_tokens": 30, "output_tokens": 20,
                                           "total_tokens": 50}),
        ],
        "compacted_tool_calls": 6, "compacted_tokens": 40_000,
        "compaction_anchor_id": "sub-anchor",
    }, None)
    u = meter.subagent_usage["research-subagent"]
    assert u["total_tokens"] == 50 + 40_000


def test_usage_event_carries_fleet_cost_and_compaction() -> None:
    captured: dict = {}
    orig = metering.emit
    metering.emit = lambda e: captured.update(e) if e.get("type") == "usage" else None
    try:
        meter = RunMeter()
        meter.record_subagent_usage("research-subagent", {
            "input_tokens": 500, "output_tokens": 100, "total_tokens": 600,
            "model_calls": 4, "cost_usd": 0.03})
        mw = UsageMeterMiddleware(meter, max_tool_calls=80,
                                  max_total_tokens=2_000_000, recursion_limit=4500)
        state = {
            "messages": [
                HumanMessage("q", id="anchor-1"),
                AIMessage("a", usage_metadata={"input_tokens": 100, "output_tokens": 50,
                                               "total_tokens": 150},
                          response_metadata={"token_usage": {"total_tokens": 150,
                                                             "cost": 0.02}}),
            ],
            # a compaction happened this turn (anchor id matches)
            "compacted_tool_calls": 12, "compacted_tokens": 40_000,
            "compaction_anchor_id": "anchor-1",
        }
        mw.after_agent(state, None)
    finally:
        metering.emit = orig

    assert captured["total_tokens"] == 150 + 40_000        # orchestrator + compacted
    assert captured["compacted_tool_calls"] == 12
    assert captured["subagents"]["research-subagent"]["total_tokens"] == 600
    assert captured["total_tokens_all_agents"] == 150 + 40_000 + 600
    assert captured["cost_usd"] == 0.05                    # orchestrator + fleet


if __name__ == "__main__":
    test_sum_usage_fallback_chain_and_cost()
    test_meter_rolls_up_per_role()
    test_subagent_middleware_meters_its_own_state()
    test_subagent_middleware_folds_its_own_compaction()
    test_usage_event_carries_fleet_cost_and_compaction()
    print("OK — sub-agent usage rollup + cost accounting verified.")


def test_subagent_middleware_marks_start_and_done_with_role_and_model() -> None:
    from deep_research_agent import events

    captured: list[dict] = []
    orig = events._writer
    events._writer = lambda: captured.append
    try:
        m = RunMeter()
        mw = SubagentUsageMiddleware(m, "extract-subagent", model="qwen/qwen3-30b")
        mw.before_agent({"messages": []}, None)
        state = {"messages": [
            HumanMessage("read the file"),
            AIMessage("done", usage_metadata={"input_tokens": 1000, "output_tokens": 20,
                                              "total_tokens": 1020}),
        ]}
        mw.after_agent(state, None)
    finally:
        events._writer = orig

    assert [e["state"] for e in captured] == ["subagent_start", "subagent_done"]
    assert all(e["type"] == "status" for e in captured)
    assert captured[0]["role"] == "extract-subagent" and captured[0]["model"] == "qwen/qwen3-30b"
    assert captured[1]["model_calls"] == 1 and captured[1]["total_tokens"] == 1020
    assert m.subagent_usage["extract-subagent"]["runs"] == 1
    # Old two-arg construction still works (model optional).
    SubagentUsageMiddleware(m, "research-subagent").before_agent({"messages": []}, None)
