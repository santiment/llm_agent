"""Guard the MCP backpressure contract.

langchain-mcp-adapters opens a NEW connection per tool call, so the agent's
fan-out (orchestrator + parallel sub-researchers) must be bounded. ``instrument_tool``
takes a shared ``asyncio.Semaphore`` that admits at most N calls at once and retries
a rate-limit signal a few times. These tests pin both behaviors.

Runs with plain Python (``python tests/test_mcp_concurrency.py``) — no pytest needed —
and is also pytest-discoverable.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from deep_research_agent.events import instrument_tool


class _EmptyArgs(BaseModel):
    pass


def _expect_raise(make_coro) -> None:
    """Run ``make_coro()`` to completion; fail unless it raised. Kept as a plain
    helper (not ``pytest.raises``) so this file still runs under bare ``python``."""
    try:
        asyncio.run(make_coro())
    except Exception:
        return
    raise AssertionError("expected the call to raise")


class _FakeTool:
    """Minimal stand-in exposing only what ``instrument_tool`` reads."""

    def __init__(self, behavior):
        self.name = "fake_mcp_tool"
        self.description = "fake"
        self.args_schema = _EmptyArgs
        self._behavior = behavior

    async def ainvoke(self, _kwargs):
        return await self._behavior()


def test_semaphore_caps_concurrency() -> None:
    """A shared semaphore must hold simultaneous calls at or below its size, no
    matter how many are fired at once."""
    limit = 3
    gate = asyncio.Semaphore(limit)
    live = 0
    peak = 0

    async def behavior():
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)  # hold the slot so overlap is observable
        live -= 1
        return "ok"

    wrapped = instrument_tool(_FakeTool(behavior), kind="mcp", semaphore=gate)

    async def drive():
        await asyncio.gather(*(wrapped.ainvoke({}) for _ in range(20)))

    asyncio.run(drive())
    assert peak <= limit, f"peak concurrency {peak} exceeded cap {limit}"
    assert peak == limit, f"expected the cap {limit} to be saturated, saw {peak}"


def test_waits_then_succeeds_on_rate_limit() -> None:
    """A 429 is waited out (not failed); a success on a later attempt is returned."""
    calls = 0

    async def behavior():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("HTTP 429 Too Many Requests")
        return "recovered"

    wrapped = instrument_tool(
        _FakeTool(behavior), kind="mcp", rate_limit_max_wait=60.0, base_delay=0.0
    )
    result = asyncio.run(wrapped.ainvoke({}))
    assert result == "recovered"
    assert calls == 3, f"expected it to keep retrying until success, got {calls}"


def test_rate_limit_gives_up_after_budget() -> None:
    """A server that never recovers must not hang forever — past the wait budget the
    call finally fails."""
    calls = 0

    async def behavior():
        nonlocal calls
        calls += 1
        raise RuntimeError("429 Too Many Requests")

    # delays 0.01, 0.02, 0.04 → waits 0.01 then 0.03; next would hit 0.07 > 0.05 → gives up.
    wrapped = instrument_tool(
        _FakeTool(behavior), kind="mcp", rate_limit_max_wait=0.05, base_delay=0.01
    )
    _expect_raise(lambda: wrapped.ainvoke({}))


def test_non_rate_limit_error_is_not_retried() -> None:
    """Connection/fd errors must propagate immediately — retrying them would only
    hammer an already-struggling server."""
    calls = 0

    async def behavior():
        nonlocal calls
        calls += 1
        raise RuntimeError("connection refused")

    wrapped = instrument_tool(
        _FakeTool(behavior), kind="mcp", rate_limit_max_wait=60.0, base_delay=0.0
    )
    _expect_raise(lambda: wrapped.ainvoke({}))
    assert calls == 1, f"non-rate-limit error must not retry, got {calls} calls"


if __name__ == "__main__":
    test_semaphore_caps_concurrency()
    test_waits_then_succeeds_on_rate_limit()
    test_rate_limit_gives_up_after_budget()
    test_non_rate_limit_error_is_not_retried()
    print("OK — MCP concurrency cap + retry behavior verified.")
