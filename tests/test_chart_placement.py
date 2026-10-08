"""Fetching a series is not showing it: a chart reaches the UI only when a report PLACES it.

Regression for the "why did sentiment peak around August 1" run that showed price volatility,
open interest, exchange inflow, social volume … as cards: every series a sub-agent fetched became
a `chart` event on the spot, and the UI rendered each on arrival. Pins:

  - the tool wrapper HOLDS the chart (events.stash_chart — bounded by count and age) and only
    names it to the model; nothing is emitted at fetch time;
  - `submit_report` (tools/report.deliver_charts) emits the held charts the report places, in
    report order, BEFORE the `report` event; a placement the run no longer holds leaves the
    text; the salvage path does the same;
  - the prompts route a "why did X peak on <date>" question to the messages of that window,
    not a metric sweep, and keep a skill's full playbook for the skill's own question;
  - every model carries its `role` in run metadata so the messages stream can tell the
    orchestrator's narration from a worker's STATUS/OUTPUT/NOTES handoff.
"""

from __future__ import annotations

import asyncio

import pytest

import deep_research_agent.events as events
from conftest import capture_events_cm
from deep_research_agent import prompts
from deep_research_agent.events import (CHART_TTL_S, MAX_STASHED_CHARTS, clear_charts,
                                        emit_placed_charts, stash_chart, stashed_chart)
from deep_research_agent.report_hygiene import chart_refs, drop_chart_refs
from deep_research_agent.series import find_series
from deep_research_agent.tools.report import build_submit_report_tool, deliver_charts
from test_events_protocol import _points


@pytest.fixture(autouse=True)
def _fresh():
    clear_charts()
    yield
    clear_charts()


def _held(n: int = 20, **kw) -> str:
    return stash_chart(find_series(_points(n)), **kw)


# --- held, not emitted ---------------------------------------------------------------------

def test_stash_holds_and_emits_only_on_placement_in_report_order():
    with capture_events_cm() as ev:
        a, b, c = _held(), _held(), _held()
    assert ev == [] and all(len(x) == 8 for x in (a, b, c))
    with capture_events_cm() as ev:
        assert emit_placed_charts([c, a]) == ([c, a], [])
    assert [e["id"] for e in ev] == [c, a] and all(e["type"] == "chart" for e in ev)
    assert stashed_chart(c) is not None                        # still held: a follow-up may place it again
    assert stashed_chart(b) is not None


def test_unknown_evicted_or_expired_ids_are_reported_missing():
    a = _held()
    assert emit_placed_charts(["deadbeef", a]) == ([a], ["deadbeef"])
    # Age: an entry older than the TTL goes on the next stash.
    born, event = events._CHARTS[""][a]                     # "" = outside a LangGraph run
    events._CHARTS[""][a] = (born - CHART_TTL_S - 1, event)
    b = _held()
    assert stashed_chart(a) is None and stashed_chart(b) is not None
    # Count: the oldest leaves first once the bound is hit.
    clear_charts()
    ids = [_held(12) for _ in range(MAX_STASHED_CHARTS + 3)]
    assert len(events._CHARTS[""]) == MAX_STASHED_CHARTS
    assert all(stashed_chart(i) is None for i in ids[:3]) and stashed_chart(ids[-1]) is not None


def test_charts_are_held_per_thread():
    # One run's fan-out must not evict another's charts, and a report may only place charts
    # its own thread fetched — an id pasted from someone else's report resolves to nothing.
    from unittest.mock import patch

    with patch.object(events, "_chart_scope", return_value="thread-a"):
        a = _held()
    with patch.object(events, "_chart_scope", return_value="thread-b"):
        others = [_held(12) for _ in range(MAX_STASHED_CHARTS + 1)]
        assert stashed_chart(a) is None
        assert emit_placed_charts([a]) == ([], [a])
    with patch.object(events, "_chart_scope", return_value="thread-a"):
        assert stashed_chart(a) is not None                  # B's flood never touched A's
        assert stashed_chart(others[-1]) is None


def test_thread_scope_comes_from_the_langgraph_config():
    from langchain_core.runnables import RunnableLambda

    seen = RunnableLambda(lambda _: events._chart_scope()).invoke(
        None, {"configurable": {"thread_id": "t-123"}})
    assert seen == "t-123" and events._chart_scope() == ""


# --- the report places; dangling placements leave the text -------------------------------

def test_drop_chart_refs_removes_only_the_named_placements():
    md = "# T\n\nintro\n\n[chart:aaaaaaaa]\n\ntext [chart:bbbbbbbb] inline\n\n[chart:bbbbbbbb]\n"
    out = drop_chart_refs(md, ["bbbbbbbb"])
    assert "[chart:aaaaaaaa]" in out and "text [chart:bbbbbbbb] inline" in out  # inline is prose, untouched
    assert chart_refs(out) == ["aaaaaaaa"]
    assert drop_chart_refs(md, []) == md


def test_submit_report_emits_placed_charts_before_the_report_and_drops_dangling():
    a = _held(source_label="Santiment")
    md = f"# Peak\n\nNegativity peaked on the 1st[1].\n\n[chart:{a}]\n\n[chart:0badc0de]\n\n## Sources\n- [1] Santiment\n"
    tool = build_submit_report_tool()
    with capture_events_cm() as ev:
        reply = asyncio.run(tool.ainvoke({"report_markdown": md}))
    assert "DONE" in reply
    kinds = [e["type"] for e in ev]
    assert kinds == ["chart", "report"]                          # the chart precedes what places it
    assert ev[0]["id"] == a
    assert f"[chart:{a}]" in ev[1]["markdown"] and "0badc0de" not in ev[1]["markdown"]


def test_deliver_charts_is_a_no_op_without_placements():
    with capture_events_cm() as ev:
        assert deliver_charts("# T\n\nno charts here\n") == "# T\n\nno charts here\n"
    assert ev == []


def test_salvaged_report_places_charts_too():
    import deep_research_agent.citations as citations
    assert citations.deliver_charts is deliver_charts             # the fallback emit goes through it


# --- prompts: attribution, not a sweep -----------------------------------------------------

def test_orchestrator_routes_a_why_question_to_the_messages_of_the_window():
    p = prompts.orchestrator_prompt("", "")
    assert "EXPLAINING A SIGNAL" in p
    assert "is an ATTRIBUTION job, not a survey" in p
    assert "Do NOT sweep other metrics" in p
    assert "do NOT ask for a day-by-day table of many series" in p
    assert "A skill is named ONLY when the ASK is the skill's job" in p


def test_subagent_pins_the_event_then_reads_the_text():
    p = prompts.subagent_prompt("", "")
    assert "EXPLAINING A SIGNAL" in p
    assert "have `extract-subagent` name the drivers with counts and quotes" in p
    assert "Do not fetch other metrics as a sweep" in p


def test_skill_scopes_itself_out_of_date_attribution():
    from pathlib import Path
    text = Path(__file__).resolve().parents[1].joinpath("skills/crowd-positioning/SKILL.md").read_text()
    assert "**Scope.**" in text and "that is attribution" in text and "skip signals 1–4" in text


# --- role metadata on the messages stream ---------------------------------------------------

def test_every_model_carries_its_role_in_run_metadata(monkeypatch):
    from conftest import make_graph_capture
    from deep_research_agent.config import ResearchConfig
    from deep_research_agent.models import build_chat_model

    cfg = ResearchConfig.from_runnable_config({"configurable": {"openai_api_key": "k"}})
    # LangChain adds its own keys (lc_versions) to the model's metadata; ours rides alongside.
    assert build_chat_model(cfg.research_model, cfg, role="coding-subagent").metadata["role"] == "coding-subagent"
    assert "role" not in (build_chat_model(cfg.research_model, cfg).metadata or {})

    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    captured = make_graph_capture(monkeypatch, {"configurable": {
        "openai_api_key": "k", "mcp_servers": [], "sandbox_url": "http://sandbox.invalid:8080"}})
    assert captured["model"].metadata["role"] == "orchestrator"
    roles = {s["name"]: s["model"].metadata["role"] for s in captured["subagents"]}
    assert roles == {"research-subagent": "research-subagent", "extract-subagent": "extract-subagent",
                     "coding-subagent": "coding-subagent"}


def test_findings_event_carries_no_chart_tokens_but_the_parent_keeps_them():
    # The UI never got a chart event for a finding's token; the orchestrator needs it to place.
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from conftest import make_state
    from deep_research_agent.findings_gate import SubagentFindingsMiddleware

    a = _held()
    handoff = AIMessage('{"summary": "price fell [chart:%s]", "findings": [{"finding": '
                        '"BTC -8.4%% over 30d\\n[chart:%s]", "evidence": "first/last", '
                        '"source": "Santiment"}], "gaps": []}' % (a, a))
    with capture_events_cm() as ev:
        update = SubagentFindingsMiddleware().after_model(
            make_state(HumanMessage("unit: market"), ToolMessage("rows", tool_call_id="1"), handoff), None)
    found = [e for e in ev if e["type"] == "subagent_findings"][0]
    assert "[chart:" not in found["summary"] and "[chart:" not in found["findings"][0]["finding"]
    parent_sees = update["messages"][0].content if update else handoff.content
    assert f"[chart:{a}]" in parent_sees


def test_persisted_report_matches_what_the_user_got():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from conftest import make_state
    from deep_research_agent.citations import ResearchOutputMiddleware
    from deep_research_agent.turn import REPORT_DELIVERED

    a = _held()
    rep = f"# R\n\nText [1].\n\n[chart:{a}]\n\n[chart:0badc0de]\n\n## Sources\n- [1] https://x.example/a\n"
    call = {"name": "submit_report", "args": {"report_markdown": rep}, "id": "t1"}
    out = ResearchOutputMiddleware(max_tool_calls=80, max_total_tokens=1_000_000).after_agent(
        make_state(HumanMessage("q"), AIMessage("", tool_calls=[call]),
                   ToolMessage(REPORT_DELIVERED, tool_call_id="t1")), None)
    assert f"[chart:{a}]" in out["final_report"] and "0badc0de" not in out["final_report"]


def test_total_held_charts_are_bounded_across_threads():
    from unittest.mock import patch

    with patch.object(events, "MAX_STASHED_CHARTS_TOTAL", 10), patch.object(events, "MAX_STASHED_CHARTS", 4):
        first = None
        for t in range(6):
            with patch.object(events, "_chart_scope", return_value=f"t{t}"):
                ids = [_held(12) for _ in range(3)]
                first = first or ids[0]
        assert sum(len(h) for h in events._CHARTS.values()) <= 10
        with patch.object(events, "_chart_scope", return_value="t0"):
            assert stashed_chart(first) is None                # the oldest in the process went first


def test_global_eviction_never_loses_the_chart_being_stashed():
    # The current thread's only chart is the process's oldest: evicting it must not orphan
    # the dict the new chart goes into.
    from unittest.mock import patch

    with patch.object(events, "MAX_STASHED_CHARTS_TOTAL", 2):
        with patch.object(events, "_chart_scope", return_value="mine"):
            _held()
        with patch.object(events, "_chart_scope", return_value="other"):
            _held()
        with patch.object(events, "_chart_scope", return_value="mine"):
            new = _held()
            assert stashed_chart(new) is not None
