"""Pin the per-run usage ledger: RunMeter accumulates across calls (incl. errors and
caps), and UsageMeterMiddleware emits one `usage` event / RESEARCH USAGE log with the
global tool/size counters plus orchestrator-level token + model-call counts.

Runs with plain Python (``python tests/test_metering.py``) — no pytest needed.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research_agent import metering
from deep_research_agent.metering import RunMeter, UsageMeterMiddleware


def test_meter_accumulates() -> None:
    m = RunMeter()
    m.record_tool_result(ok=True, result_bytes=1000, result_rows=50)
    m.record_tool_result(ok=True, result_bytes=70_000, result_rows=2000, capped=True)
    m.record_tool_result(ok=False)
    assert (m.tool_calls, m.tool_errors, m.capped_calls, m.result_rows, m.result_bytes) \
        == (3, 1, 1, 2050, 71_000)


def test_usage_event_has_all_categories() -> None:
    captured = {}
    orig = metering.emit  # middleware calls the `emit` name bound inside metering.py
    metering.emit = lambda e: captured.update(e) if e.get("type") == "usage" else None
    try:
        m = RunMeter()
        m.record_tool_result(ok=True, result_bytes=2048, result_rows=120, capped=True)
        mw = UsageMeterMiddleware(m, max_tool_calls=80, max_total_tokens=2_000_000,
                                  recursion_limit=4500)
        state = {"messages": [
            HumanMessage("q"),
            AIMessage("a", usage_metadata={"input_tokens": 100, "output_tokens": 50,
                                            "total_tokens": 150}),
            AIMessage("b", usage_metadata={"input_tokens": 30, "output_tokens": 20,
                                            "total_tokens": 50}),
            ToolMessage("r", tool_call_id="1"),
        ]}
        mw.after_agent(state, None)
    finally:
        metering.emit = orig

    assert captured["type"] == "usage"
    assert captured["tool_calls"] == 1 and captured["capped_calls"] == 1
    assert captured["result_rows"] == 120 and captured["result_bytes"] == 2048
    assert captured["input_tokens"] == 130 and captured["output_tokens"] == 70
    assert captured["total_tokens"] == 200 and captured["model_calls"] == 2
    assert captured["tool_calls_in_context"] == 1
    assert captured["limits"]["recursion_limit"] == 4500


if __name__ == "__main__":
    test_meter_accumulates()
    test_usage_event_has_all_categories()
    print("OK — usage ledger verified.")
