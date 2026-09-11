"""A throttled model provider must not end the run — and neither must one dead sub-agent.

Regression for the run killed by ``OpenAIRateLimitError: Error code: 429 … temporarily
rate-limited upstream`` on a sub-agent's model call: the SDK's three sub-second retries
were spent in ~4 s, the exception escaped the sub-agent, unwound the orchestrator's
``task`` call and discarded every other unit's work (UI: "An internal error occurred").
Pins (``model_errors.py``):

  - ModelBackoffMiddleware waits out retryable model errors — capped exponential backoff,
    a provider's Retry-After honored — and calls again until the cumulative wait would
    exceed its budget, then the error stands; non-transient errors and LangGraph control
    flow propagate at once; a 0 budget disables the waiting;
  - SubagentFailureMiddleware turns an exception out of ``task`` into an error
    ToolMessage naming the sub-agent and the cause; other tools are untouched;
  - the budget knob (DRA_MODEL_RATE_LIMIT_MAX_WAIT / configurable) parses like its MCP
    twin, and both middlewares are wired: backoff LAST on every role, failure isolation
    on the two roles that hold a `task` tool.

Runs with plain Python (``python tests/test_model_errors.py``) — no pytest needed for
the unit tests; the wiring test needs pytest's monkeypatch and the shared conftest.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.exceptions import (ModelAPIError, ModelInvalidRequestError,
                                       ModelRateLimitError, ModelTimeoutError)
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

import deep_research_agent.events as events
from deep_research_agent.config import ResearchConfig
from deep_research_agent.model_errors import (ModelBackoffMiddleware, SubagentFailureMiddleware,
                                              is_transient_model_error, retry_after_seconds)

_ENV_KEYS = ("DRA_MODEL_RATE_LIMIT_MAX_WAIT",)


# --- helpers ---------------------------------------------------------------------------

class _Captured:
    """Route ``events.emit`` into a list and record every sleep the middleware asks for."""

    def __init__(self, mw: ModelBackoffMiddleware) -> None:
        self.events: list[dict] = []
        self.sleeps: list[float] = []
        self._mw = mw
        self._orig = None

    def __enter__(self):
        self._orig = events._writer
        events._writer = lambda: self.events.append

        async def asleep(delay: float) -> None:
            self.sleeps.append(delay)

        self._mw._asleep = asleep
        return self

    def __exit__(self, *_exc):
        events._writer = self._orig

    def states(self) -> list[str]:
        return [e["state"] for e in self.events if e.get("type") == "status"]


def _failing_then(results: list, exc_factory):
    """A handler that raises ``exc_factory()`` until ``results`` runs out of exceptions:
    each entry is either the exception to raise or the value to return."""
    calls = {"n": 0}

    async def handler(_request):
        i = calls["n"]
        calls["n"] += 1
        item = results[i]
        if isinstance(item, BaseException):
            raise item
        return item

    handler.calls = calls  # type: ignore[attr-defined]
    return handler


def _run(mw: ModelBackoffMiddleware, handler):
    return asyncio.run(mw.awrap_model_call(None, handler))


def _mw(**kwargs) -> ModelBackoffMiddleware:
    kwargs.setdefault("max_wait", 120.0)
    kwargs.setdefault("jitter", False)
    return ModelBackoffMiddleware("research-subagent", "deepseek/deepseek-v4.1-flash", **kwargs)


_429 = ("Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, "
        "'metadata': {'raw': 'deepseek/deepseek-v4.1-flash is temporarily rate-limited "
        "upstream. Please retry shortly'}}}")


# --- classification ----------------------------------------------------------------------

def test_transient_classification() -> None:
    assert is_transient_model_error(ModelRateLimitError(_429))
    assert is_transient_model_error(ModelAPIError("502 Bad Gateway"))
    assert is_transient_model_error(ModelTimeoutError("timed out"))
    assert not is_transient_model_error(ModelInvalidRequestError("bad schema"))
    # Unclassified exceptions: only a rate-limit marker in the text makes them transient.
    assert is_transient_model_error(RuntimeError("HTTP 429 Too Many Requests"))
    assert not is_transient_model_error(RuntimeError("boom"))


def test_retry_after_from_header_then_message() -> None:
    class _Resp:
        status_code = 429
        text = "slow down"

        def __init__(self, headers):
            self.headers = headers

    exc = ModelRateLimitError("429")
    exc.response = _Resp({"retry-after": "7"})
    assert retry_after_seconds(exc) == 7.0
    exc.response = _Resp({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})  # date form: skip
    assert retry_after_seconds(exc) is None
    assert retry_after_seconds(ModelRateLimitError("429: retry after 3 seconds")) == 3.0
    assert retry_after_seconds(ModelRateLimitError(_429)) is None


# --- backoff -----------------------------------------------------------------------------

def test_transient_error_is_waited_out_then_succeeds() -> None:
    mw = _mw()
    handler = _failing_then([ModelRateLimitError(_429), ModelRateLimitError(_429), "ok"],
                            None)
    with _Captured(mw) as cap:
        assert _run(mw, handler) == "ok"
    assert handler.calls["n"] == 3
    assert cap.sleeps == [2.0, 4.0]  # base 2 s, doubling
    assert cap.states() == ["rate_limited", "rate_limited"]
    first = cap.events[0]
    assert first["role"] == "research-subagent" and first["model"] == "deepseek/deepseek-v4.1-flash"
    assert first["retry"] == 1 and first["wait_s"] == 2.0 and first["budget_s"] == 120.0
    assert "ModelRateLimitError" in first["detail"] and "rate-limited upstream" in first["detail"]
    assert cap.events[1]["retry"] == 2 and cap.events[1]["waited_s"] == 2.0


def test_budget_bounds_the_waiting_then_the_error_stands() -> None:
    mw = _mw(max_wait=10.0)
    handler = _failing_then([ModelRateLimitError(_429)] * 10, None)
    with _Captured(mw) as cap:
        with pytest.raises(ModelRateLimitError):
            _run(mw, handler)
    # 2 + 4 = 6 s spent; the next 8 s wait would overrun the 10 s budget → give up.
    assert cap.sleeps == [2.0, 4.0]
    assert handler.calls["n"] == 3
    assert cap.states() == ["rate_limited", "rate_limited", "model_unavailable"]
    gave_up = cap.events[-1]
    assert gave_up["retries"] == 2 and gave_up["waited_s"] == 6.0 and gave_up["budget_s"] == 10.0


def test_delay_is_capped_at_max_delay() -> None:
    mw = _mw(max_wait=1000.0, max_delay=30.0)
    handler = _failing_then([ModelRateLimitError(_429)] * 7 + ["ok"], None)
    with _Captured(mw) as cap:
        assert _run(mw, handler) == "ok"
    assert cap.sleeps == [2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]


def test_retry_after_hint_wins_over_backoff() -> None:
    mw = _mw()
    handler = _failing_then([ModelRateLimitError("429: retry after 9 seconds"), "ok"], None)
    with _Captured(mw) as cap:
        assert _run(mw, handler) == "ok"
    assert cap.sleeps == [9.0]


def test_non_transient_error_propagates_at_once() -> None:
    mw = _mw()
    handler = _failing_then([ModelInvalidRequestError("bad request"), "never"], None)
    with _Captured(mw) as cap:
        with pytest.raises(ModelInvalidRequestError):
            _run(mw, handler)
    assert cap.sleeps == [] and cap.events == [] and handler.calls["n"] == 1


def test_graph_control_flow_propagates() -> None:
    mw = _mw()
    handler = _failing_then([GraphBubbleUp(), "never"], None)
    with _Captured(mw) as cap:
        with pytest.raises(GraphBubbleUp):
            _run(mw, handler)
    assert cap.sleeps == [] and cap.events == []


def test_zero_budget_disables_the_waiting() -> None:
    mw = _mw(max_wait=0.0)
    handler = _failing_then([ModelRateLimitError(_429), "never"], None)
    with _Captured(mw) as cap:
        with pytest.raises(ModelRateLimitError):
            _run(mw, handler)
    assert cap.sleeps == [] and handler.calls["n"] == 1
    assert cap.states() == ["model_unavailable"]


def test_jitter_stays_within_twenty_percent() -> None:
    mw = _mw(jitter=True)
    handler = _failing_then([ModelRateLimitError(_429), "ok"], None)
    with _Captured(mw) as cap:
        assert _run(mw, handler) == "ok"
    assert len(cap.sleeps) == 1 and 1.6 <= cap.sleeps[0] <= 2.4


# --- a dead sub-agent is a tool error, not a dead run ---------------------------------------

_SENTINEL = ToolMessage(content="HANDLER_RAN", tool_call_id="c1", name="task")


def _req(name: str, args: dict) -> ToolCallRequest:
    return ToolCallRequest(tool_call={"name": name, "args": args, "id": "c1"},
                           tool=None, state={}, runtime=None)


def _run_tool(mw, req, handler):
    return asyncio.run(mw.awrap_tool_call(req, handler))


def _raising(exc):
    async def handler(_request):
        raise exc
    return handler


async def _ok(_request):
    return _SENTINEL


def test_task_failure_becomes_an_error_tool_result() -> None:
    mw = SubagentFailureMiddleware()
    req = _req("task", {"description": "Research BTC", "subagent_type": "research-subagent"})
    with _Captured(_mw()) as cap:
        out = _run_tool(mw, req, _raising(ModelRateLimitError(_429)))
    assert isinstance(out, ToolMessage) and out is not _SENTINEL
    assert out.status == "error" and out.name == "task" and out.tool_call_id == "c1"
    assert "research-subagent" in out.content and "ModelRateLimitError" in out.content
    assert "NOT researched" in out.content and "Retry that task ONCE" in out.content
    assert cap.states() == ["subagent_failed"]
    assert cap.events[0]["role"] == "research-subagent"
    assert "rate-limited upstream" in cap.events[0]["detail"]


def test_task_success_passes_through() -> None:
    req = _req("task", {"description": "x", "subagent_type": "research-subagent"})
    assert _run_tool(SubagentFailureMiddleware(), req, _ok) is _SENTINEL


def test_other_tools_are_untouched() -> None:
    req = _req("execute", {"command": "python x.py"})
    with pytest.raises(RuntimeError):
        _run_tool(SubagentFailureMiddleware(), req, _raising(RuntimeError("sandbox down")))


def test_task_control_flow_propagates() -> None:
    req = _req("task", {"description": "x", "subagent_type": "research-subagent"})
    with pytest.raises(GraphBubbleUp):
        _run_tool(SubagentFailureMiddleware(), req, _raising(GraphBubbleUp()))


# --- the knob -----------------------------------------------------------------------------

def _cfg(env: dict[str, str] | None = None, **configurable) -> ResearchConfig:
    saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
    os.environ.update(env or {})
    try:
        return ResearchConfig.from_runnable_config({"configurable": configurable})
    finally:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update({k: v for k, v in saved.items() if v is not None})


def test_budget_knob_parses_like_its_mcp_twin() -> None:
    assert _cfg().model_rate_limit_max_wait == 120.0 == ResearchConfig.model_rate_limit_max_wait
    assert _cfg({"DRA_MODEL_RATE_LIMIT_MAX_WAIT": "30"}).model_rate_limit_max_wait == 30.0
    assert _cfg({"DRA_MODEL_RATE_LIMIT_MAX_WAIT": "30"},
                model_rate_limit_max_wait=45).model_rate_limit_max_wait == 45.0
    assert _cfg(model_rate_limit_max_wait=0).model_rate_limit_max_wait == 0.0  # off, not default
    assert _cfg(model_rate_limit_max_wait=-5).model_rate_limit_max_wait == 0.0


# --- wiring -------------------------------------------------------------------------------

def test_wired_on_every_role(monkeypatch) -> None:
    from conftest import make_graph_capture

    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    config = {"configurable": {"openai_api_key": "k", "mcp_servers": [],
                               "sandbox_url": "http://sandbox.invalid:8080",
                               "model_rate_limit_max_wait": 42}}
    cfg = ResearchConfig.from_runnable_config(config)
    captured = make_graph_capture(monkeypatch, config)

    def backoff_of(mws: list) -> ModelBackoffMiddleware:
        found = [m for m in mws if isinstance(m, ModelBackoffMiddleware)]
        assert len(found) == 1, found
        # Last among the model-call wrappers: nothing after it rewrites the request.
        after = mws[mws.index(found[0]) + 1:]
        assert not [m for m in after if "awrap_model_call" in vars(type(m))], after
        return found[0]

    orch = backoff_of(captured["middleware"])
    assert (orch.role, orch.model, orch.max_wait) == ("orchestrator", cfg.research_model, 42.0)
    assert any(isinstance(m, SubagentFailureMiddleware) for m in captured["middleware"])

    specs = {s["name"]: s for s in captured["subagents"]}
    expected = {"research-subagent": cfg.subagent_model, "extract-subagent": cfg.utility_model,
                "coding-subagent": cfg.coding_model}
    assert set(specs) == set(expected)
    for name, model in expected.items():
        mws = specs[name]["middleware"]
        bo = backoff_of(mws)
        assert (bo.role, bo.model, bo.max_wait) == (name, model, 42.0), name
        assert mws[-1] is bo, name
        holds_task = name == "research-subagent"  # it nests the extract/coding workers
        assert any(isinstance(m, SubagentFailureMiddleware) for m in mws) is holds_task, name


if __name__ == "__main__":
    test_transient_classification()
    test_retry_after_from_header_then_message()
    test_transient_error_is_waited_out_then_succeeds()
    test_budget_bounds_the_waiting_then_the_error_stands()
    test_delay_is_capped_at_max_delay()
    test_retry_after_hint_wins_over_backoff()
    test_non_transient_error_propagates_at_once()
    test_graph_control_flow_propagates()
    test_zero_budget_disables_the_waiting()
    test_jitter_stays_within_twenty_percent()
    test_task_failure_becomes_an_error_tool_result()
    test_task_success_passes_through()
    test_other_tools_are_untouched()
    test_task_control_flow_propagates()
    test_budget_knob_parses_like_its_mcp_twin()
    print("OK — model backoff + sub-agent failure isolation verified.")
