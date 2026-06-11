"""Middleware that intercepts a tool call must actually fire.

Both guards hang off ``awrap_tool_call(request, handler)`` and read the tool name from
``request.tool_call``. langchain renamed that field (``call`` -> ``tool_call``), which
silently turned the report quality gate into a no-op — a bad report sailed through
because the name never matched. These tests pin the wiring against the INSTALLED
``ToolCallRequest`` so a future rename fails loudly here instead of in production:

  - ReportQualityGateMiddleware bounces a defective report (and passes a clean one);
  - ClarificationGuardMiddleware blocks request_clarification AFTER research started
    (no event), and allows it up-front.

Runs with plain Python (``python tests/test_middleware_wiring.py``) — no pytest needed.
"""

from __future__ import annotations

import asyncio

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research_agent.clarify_fallback import ClarificationGuardMiddleware
from deep_research_agent.report_gate import ReportQualityGateMiddleware

_HANDLER_SENTINEL = ToolMessage(content="HANDLER_RAN", tool_call_id="x", name="t")


async def _handler(_request):
    return _HANDLER_SENTINEL


def _req(name: str, args: dict, state: dict) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "c1"},
        tool=None, state=state, runtime=None)


def _run(mw, req):
    return asyncio.run(mw.awrap_tool_call(req, _handler))


# --- report quality gate actually fires (the no-op regression) --------------

_BAD_REPORT = (  # empty Sources entries — only the gate's bounce catches these
    "# BTC\n\nIt held support[1] near the lows[2].\n\n## Sources\n- [1]\n- [2]\n")
_GOOD_REPORT = (
    "# BTC\n\nIt held support[1].\n\n## Sources\n- [1] [Note](https://x.com/a)\n")


def test_report_gate_bounces_defective_report() -> None:
    mw = ReportQualityGateMiddleware()
    out = _run(mw, _req("submit_report", {"report_markdown": _BAD_REPORT}, {}))
    assert isinstance(out, ToolMessage) and out is not _HANDLER_SENTINEL
    assert "submit_report" in out.content.lower() or "fix" in out.content.lower()
    assert "EMPTY" in out.content or "[1]" in out.content  # the empty-source problem


def test_report_gate_passes_clean_report() -> None:
    mw = ReportQualityGateMiddleware()
    out = _run(mw, _req("submit_report", {"report_markdown": _GOOD_REPORT}, {}))
    assert out is _HANDLER_SENTINEL  # handler ran -> report delivered


def test_report_gate_ignores_other_tools() -> None:
    mw = ReportQualityGateMiddleware()
    out = _run(mw, _req("web_search", {"query": "x"}, {}))
    assert out is _HANDLER_SENTINEL


# --- clarification guard ----------------------------------------------------

def _researched_state() -> dict:
    return {"messages": [HumanMessage("Analyze bitcoin"),
                         ToolMessage("rows", tool_call_id="d1", name="fetch_metric_data")]}


def test_clarify_blocked_after_research() -> None:
    mw = ClarificationGuardMiddleware()
    out = _run(mw, _req("request_clarification", {"questions": ["what?"]}, _researched_state()))
    assert isinstance(out, ToolMessage) and out is not _HANDLER_SENTINEL
    assert "already started researching" in out.content.lower()
    assert "submit_report" in out.content


def test_clarify_allowed_before_research() -> None:
    mw = ClarificationGuardMiddleware()
    state = {"messages": [HumanMessage("do something vague")]}  # no tool calls yet
    out = _run(mw, _req("request_clarification", {"questions": ["what?"]}, state))
    assert out is _HANDLER_SENTINEL  # up-front clarification passes through


def test_clarify_guard_ignores_other_tools() -> None:
    mw = ClarificationGuardMiddleware()
    out = _run(mw, _req("web_search", {"query": "x"}, _researched_state()))
    assert out is _HANDLER_SENTINEL


if __name__ == "__main__":
    test_report_gate_bounces_defective_report()
    test_report_gate_passes_clean_report()
    test_report_gate_ignores_other_tools()
    test_clarify_blocked_after_research()
    test_clarify_allowed_before_research()
    test_clarify_guard_ignores_other_tools()
    print("OK — tool-call guards fire against the installed ToolCallRequest.")
