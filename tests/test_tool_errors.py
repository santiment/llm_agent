"""A failed MCP tool call must not kill the run.

Regression for the run aborted by `ToolException: Metric 'nvt' is not supported...`:
the wrapper used to re-raise, LangGraph's ToolNode propagated it, and the whole
research died over one mistyped metric — the model never saw the error message that
told it exactly how to recover. Pins the hardened contract (``events.instrument_tool``):

  - an exception becomes a MODEL-READABLE tool result (never a raise) carrying the
    error plus how-to-proceed guidance;
  - errors are classified permanent / transient / unknown (server ``[permanent]`` /
    ``[transient]`` tags win over the marker heuristics);
  - an identical retry of a permanently-failed call is answered locally — the
    server is not hammered with a call that cannot succeed.

Runs with plain Python (``python tests/test_tool_errors.py``) — no pytest needed —
and is also pytest-discoverable.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from deep_research_agent.events import classify_tool_error, instrument_tool


class _EmptyArgs(BaseModel):
    pass


class _FakeTool:
    """Minimal stand-in exposing only what ``instrument_tool`` reads."""

    def __init__(self, behavior):
        self.name = "fake_mcp_tool"
        self.description = "fake"
        self.args_schema = _EmptyArgs
        self._behavior = behavior
        self.calls = 0

    async def ainvoke(self, _kwargs):
        self.calls += 1
        return await self._behavior()


class _FakeMeter:
    def __init__(self):
        self.ok = 0
        self.failed = 0

    def record_tool_result(self, ok, **_kwargs):
        if ok:
            self.ok += 1
        else:
            self.failed += 1


def _invoke(wrapped, kwargs=None):
    return asyncio.run(wrapped.ainvoke(kwargs or {}))


def test_classify_tool_error() -> None:
    assert classify_tool_error("Metric 'nvt' is not supported. Use the "
                               "metrics_and_assets_discovery_tool.") == "permanent"
    assert classify_tool_error("Slug 'btcc' mistyped or not supported.") == "permanent"
    assert classify_tool_error("upstream connection timed out") == "transient"
    assert classify_tool_error("[permanent] anything at all") == "permanent"
    assert classify_tool_error("[transient] metric not supported") == "transient"  # tag wins
    assert classify_tool_error("something nobody anticipated") == "unknown"


def test_error_returns_guidance_instead_of_raising() -> None:
    async def boom():
        raise Exception("Metric 'nvt' is not supported. Use the discovery tool.")

    meter = _FakeMeter()
    wrapped = instrument_tool(_FakeTool(boom), "mcp", meter=meter)
    result = _invoke(wrapped)  # would raise before the fix
    assert isinstance(result, str), result
    assert "TOOL ERROR" in result and "permanent" in result
    assert "not supported" in result
    assert "discovery" in result.lower()  # the actionable part survives
    assert meter.failed == 1 and meter.ok == 0


def test_transient_error_suggests_one_retry() -> None:
    async def flaky():
        raise Exception("connection timed out talking to upstream")

    result = _invoke(instrument_tool(_FakeTool(flaky), "mcp"))
    assert "transient" in result and "ONCE" in result


def test_server_tag_is_honored_and_stripped() -> None:
    async def tagged():
        raise Exception("[permanent] Slug 'btcc' does not exist.")

    result = _invoke(instrument_tool(_FakeTool(tagged), "mcp"))
    assert "permanent" in result
    assert "[permanent]" not in result  # tag consumed, not shown to the model


def test_repeated_permanent_call_short_circuits() -> None:
    async def boom():
        raise Exception("Metric 'nvt' is not supported.")

    tool = _FakeTool(boom)
    wrapped = instrument_tool(tool, "mcp")
    first = _invoke(wrapped)
    second = _invoke(wrapped)  # identical args -> answered locally
    assert tool.calls == 1, "identical permanently-failed call must not hit the server"
    assert "REPEATED CALL" in second and "Do NOT repeat" in second
    assert "REPEATED CALL" not in first


def test_success_path_unchanged() -> None:
    async def fine():
        return "rows"

    meter = _FakeMeter()
    assert _invoke(instrument_tool(_FakeTool(fine), "mcp", meter=meter)) == "rows"
    assert meter.ok == 1 and meter.failed == 0


if __name__ == "__main__":
    test_classify_tool_error()
    test_error_returns_guidance_instead_of_raising()
    test_transient_error_suggests_one_retry()
    test_server_tag_is_honored_and_stripped()
    test_repeated_permanent_call_short_circuits()
    test_success_path_unchanged()
    print("OK — tool failures return guidance, never kill the run.")


def test_exception_group_is_unwrapped_to_the_http_cause() -> None:
    """The MCP client wraps request failures in an anyio TaskGroup; the model used to see
    only "unhandled errors in a TaskGroup (1 sub-exception)" — here for an HTTP 401."""
    import httpx

    from deep_research_agent.events import exception_message

    req = httpx.Request("POST", "http://mcp.local/mcp")
    resp = httpx.Response(401, request=req,
                          text='{"error":"unauthorized","error_description":"Authorization header required"}')
    leaf = httpx.HTTPStatusError("Client error '401 Unauthorized' for url 'http://mcp.local/mcp'\n"
                                 "For more information check: https://developer.mozilla.org/...",
                                 request=req, response=resp)
    group = ExceptionGroup("unhandled errors in a TaskGroup (1 sub-exception)", [leaf])

    msg = exception_message(group)
    assert "TaskGroup" not in msg
    assert msg.startswith("HTTP 401") and "Authorization header required" in msg
    assert classify_tool_error(msg) == "permanent"

    async def boom():
        raise group

    tool = _FakeTool(boom)
    meter = _FakeMeter()
    wrapped = instrument_tool(tool, kind="mcp", meter=meter)
    out = _invoke(wrapped)
    assert "TOOL ERROR (fake_mcp_tool, permanent)" in out and "HTTP 401" in out
    assert "REJECTED THE CREDENTIALS" in out and "Do NOT retry" in out
    # an identical retry is answered locally — the server is not hit again
    out2 = _invoke(wrapped)
    assert "REPEATED CALL" in out2 and tool.calls == 1
    # a plain exception still reads as before
    assert exception_message(ValueError("Metric 'nvt' is not supported")) == "Metric 'nvt' is not supported"
