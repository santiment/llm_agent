"""Pin the per-run usage ledger: RunMeter accumulates across calls (incl. errors and
caps), and UsageMeterMiddleware emits one `usage` event / RESEARCH USAGE log with the
global tool/size counters plus orchestrator-level token + model-call counts.

Runs with plain Python (``python tests/test_metering.py``) — no pytest needed.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research_agent.metering import RunMeter, UsageMeterMiddleware, fmt_elapsed
from conftest import capture_events_cm


def test_meter_accumulates() -> None:
    m = RunMeter()
    m.record_tool_result(ok=True, result_bytes=1000, result_rows=50)
    m.record_tool_result(ok=True, result_bytes=70_000, result_rows=2000, capped=True)
    m.record_tool_result(ok=False)
    assert (m.tool_calls, m.tool_errors, m.capped_calls, m.result_rows, m.result_bytes) \
        == (3, 1, 1, 2050, 71_000)


def test_usage_event_has_all_categories() -> None:
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
    with capture_events_cm() as emitted:
        mw.after_agent(state, None)
    captured = next(e for e in emitted if e.get("type") == "usage")

    assert captured["type"] == "usage"
    assert captured["tool_calls"] == 1 and captured["capped_calls"] == 1
    assert captured["result_rows"] == 120 and captured["result_bytes"] == 2048
    assert captured["input_tokens"] == 130 and captured["output_tokens"] == 70
    assert captured["total_tokens"] == 200 and captured["model_calls"] == 2
    assert captured["tool_calls_in_context"] == 1
    assert captured["limits"]["recursion_limit"] == 4500
    # before_agent never ran here → no clock, but the keys are still there (schema).
    assert captured["elapsed_s"] is None and captured["elapsed"] == "n/a"


def test_run_time_is_on_run_start_and_usage() -> None:
    assert fmt_elapsed(None) == "n/a" and fmt_elapsed(0) == "0.0s" and fmt_elapsed(12.34) == "12.3s"
    assert fmt_elapsed(65) == "1m 05s" and fmt_elapsed(252) == "4m 12s"
    assert fmt_elapsed(3852) == "1h 04m 12s"
    m = RunMeter()
    mw = UsageMeterMiddleware(m, max_tool_calls=1, max_total_tokens=1, recursion_limit=1)
    assert m.elapsed_s() is None
    with capture_events_cm() as emitted:
        mw.before_agent({}, None)
        m.started_mono -= 252                      # pretend the run took 4m 12s
        mw.after_agent({"messages": []}, None)
    start, usage = emitted
    assert start["type"] == "run_start"
    assert "T" in start["started_at"] and start["started_at"].endswith("Z")
    assert usage["type"] == "usage" and 252 <= usage["elapsed_s"] < 253
    assert usage["elapsed"] in ("4m 12s", "4m 13s")
    assert usage["started_at"] == start["started_at"] and usage["finished_at"] >= start["started_at"]


if __name__ == "__main__":
    test_meter_accumulates()
    test_usage_event_has_all_categories()
    test_run_time_is_on_run_start_and_usage()
    print("OK — usage ledger verified.")
