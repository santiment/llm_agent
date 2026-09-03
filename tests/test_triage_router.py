"""Pre-triage router: a pure-knowledge question is answered in two tiny calls and the turn
ends; everything else — and every failure — falls through to the full agent."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research_agent.triage import (MAX_ROUTED_CHARS, TriageRouterMiddleware,
                                        parse_verdict)
from deep_research_agent.turn import BUDGET_NUDGE_NAME


class _Model:
    """Scripted fake: returns the given replies in order; records every call."""

    def __init__(self, *replies, tags=None):
        self.replies = list(replies)
        self.calls: list[list] = []
        self.tags = tags or []

    def with_config(self, tags=None, **_):
        twin = _Model(*self.replies, tags=tags)
        twin.calls = self.calls        # share the ledger so tests see both calls
        twin.replies = self.replies
        return twin

    def invoke(self, messages, **_):
        self.calls.append([self.tags, messages])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_parse_verdict_defaults_to_research():
    assert parse_verdict("SIMPLE") == "simple"
    assert parse_verdict(" simple.") == "simple"
    assert parse_verdict("RESEARCH") == "research"
    assert parse_verdict("SIMPLE or RESEARCH") == "research"
    assert parse_verdict("") == "research"


def test_simple_question_is_answered_and_the_turn_ends(capture_events):
    model = _Model(AIMessage("SIMPLE"), AIMessage("Sofia is the capital of Bulgaria.", id="a1"))
    mw = TriageRouterMiddleware(model)
    out = mw.before_model({"messages": [HumanMessage("what\'s the capital of bulgaria")]}, None)
    assert out["jump_to"] == "end"
    assert out["messages"][0].content == "Sofia is the capital of Bulgaria."
    router_call, answer_call = model.calls
    assert router_call[0] == ["nostream"] and answer_call[0] == []   # verdict hidden, answer streams
    assert "capital of bulgaria" in router_call[1][1].content
    assert [e["detail"] for e in capture_events if e.get("state") == "triage"] == ["simple"]


def test_research_verdict_falls_through_with_one_call(capture_events):
    model = _Model(AIMessage("RESEARCH"))
    out = TriageRouterMiddleware(model).before_model(
        {"messages": [HumanMessage("Bitcoin social positioning last 24h")]}, None)
    assert out is None and len(model.calls) == 1
    assert [e["detail"] for e in capture_events if e.get("state") == "triage"] == ["research"]


def test_only_the_first_step_of_a_fresh_user_turn_is_routed():
    model = _Model(AIMessage("SIMPLE"))
    mw = TriageRouterMiddleware(model)
    mid_turn = [HumanMessage("q"), AIMessage("", tool_calls=[{"name": "task", "args": {}, "id": "1"}]),
                ToolMessage("r", tool_call_id="1")]
    assert mw.before_model({"messages": mid_turn}, None) is None
    nudge = [HumanMessage("q"), AIMessage("x"), HumanMessage("wrap up", name=BUDGET_NUDGE_NAME)]
    assert mw.before_model({"messages": nudge}, None) is None
    assert mw.before_model({"messages": [HumanMessage("x" * (MAX_ROUTED_CHARS + 1))]}, None) is None
    assert model.calls == []


def test_previous_question_is_passed_as_context():
    model = _Model(AIMessage("RESEARCH"))
    msgs = [HumanMessage("Analyze ETH fees"), AIMessage("## Report"), HumanMessage("and for BTC?")]
    TriageRouterMiddleware(model).before_model({"messages": msgs}, None)
    user = model.calls[0][1][1].content
    assert "Earlier the user asked" in user and "Analyze ETH fees" in user and "and for BTC?" in user


def test_router_failure_runs_the_full_agent(caplog):
    model = _Model(RuntimeError("upstream 502"))
    with caplog.at_level("WARNING"):
        out = TriageRouterMiddleware(model).before_model({"messages": [HumanMessage("hi")]}, None)
    assert out is None and "TRIAGE router failed" in caplog.text


def test_simple_verdict_without_a_usable_answer_runs_the_full_agent():
    model = _Model(AIMessage("SIMPLE"), AIMessage("", tool_calls=[{"name": "task", "args": {}, "id": "1"}]))
    assert TriageRouterMiddleware(model).before_model({"messages": [HumanMessage("hi")]}, None) is None
