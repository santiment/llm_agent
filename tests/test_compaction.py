"""Pin in-flight context compaction (compaction.py): the trigger estimate, the
partition (summary BEFORE the anchor, tail never split from its AIMessage), the
budget bookkeeping (compacted counters keyed to the anchor id), and the failure
policy (any summarizer problem compacts NOTHING).

Runs with plain Python (``python tests/test_compaction.py``) — no pytest needed — and
is also pytest-discoverable. No network, no API keys.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from deep_research_agent.budget import BudgetMiddleware
from deep_research_agent.compaction import ContextCompactionMiddleware, compacted_counts
from deep_research_agent.turn import COMPACTION_SUMMARY_NAME, current_turn


class _FakeModel:
    """Stands in for the utility model; records calls, returns a fixed summary."""

    def __init__(self, text: str = "HANDOFF SUMMARY", fail: bool = False) -> None:
        self.text, self.fail, self.calls = text, fail, 0

    def invoke(self, messages):
        self.calls += 1
        if self.fail:
            raise RuntimeError("summarizer down")
        return AIMessage(self.text)

    async def ainvoke(self, messages):
        return self.invoke(messages)


def _turn(pairs: int, *, chars: int = 4_000) -> list:
    """One research turn: an anchor + `pairs` AIMessage/ToolMessage exchanges of
    ~chars/4 estimated tokens per message (no usage_metadata → chars/4 path)."""
    msgs: list = [HumanMessage("research X", id="anchor-1")]
    for i in range(pairs):
        msgs.append(AIMessage("a" * chars, id=f"ai-{i}"))
        msgs.append(ToolMessage("r" * chars, tool_call_id=f"c{i}", id=f"tool-{i}"))
    return msgs


def _mw(model=None, trigger: int = 5_000, keep: int = 4) -> ContextCompactionMiddleware:
    return ContextCompactionMiddleware(model or _FakeModel(),
                                       trigger_tokens=trigger, keep_recent=keep)


def test_below_threshold_is_a_no_op() -> None:
    model = _FakeModel()
    mw = _mw(model, trigger=10_000_000)
    assert mw.before_model({"messages": _turn(10)}, None) is None
    assert model.calls == 0  # didn't even call the summarizer


def test_zero_trigger_disables() -> None:
    assert _mw(trigger=0).before_model({"messages": _turn(10)}, None) is None


def test_compacts_summary_before_anchor_and_counts_dropped_work() -> None:
    msgs = _turn(10)  # ~10 tool calls, ~20 big messages, est ≫ 5k tokens
    update = _mw().before_model({"messages": msgs}, None)
    assert update is not None
    out = update["messages"]
    assert isinstance(out[0], RemoveMessage) and out[0].id == REMOVE_ALL_MESSAGES
    # summary is a synthetic-named HumanMessage placed BEFORE the anchor …
    assert isinstance(out[1], HumanMessage) and out[1].name == COMPACTION_SUMMARY_NAME
    assert "HANDOFF SUMMARY" in out[1].content
    assert out[2].id == "anchor-1"
    # … so current_turn still anchors on the real user message.
    rebuilt = out[1:]
    assert current_turn(rebuilt)[0].id == "anchor-1"
    # keep_recent=4 → tail is the last 4 messages, and it never starts on a ToolMessage
    # (walked back to include the requesting AIMessage if needed).
    tail = out[3:]
    assert not isinstance(tail[0], ToolMessage)
    assert tail[-1] is msgs[-1]
    # Dropped-from-this-turn bookkeeping: everything before the tail was summarized.
    assert update["compaction_anchor_id"] == "anchor-1"
    kept_tool_msgs = sum(1 for m in tail if isinstance(m, ToolMessage))
    assert update["compacted_tool_calls"] == 10 - kept_tool_msgs
    assert update["compacted_tokens"] > 0
    assert 0 < update["compacted_budget_tokens"] <= update["compacted_tokens"]   # budget units


def test_tail_never_splits_an_ai_tool_pair() -> None:
    # keep_recent=3 on pair-structured messages lands the naive cut on a ToolMessage.
    update = _mw(keep=3).before_model({"messages": _turn(10)}, None)
    tail = update["messages"][3:]
    assert isinstance(tail[0], AIMessage)
    assert isinstance(tail[1], ToolMessage)


def test_counters_accumulate_across_compactions() -> None:
    state = {"messages": _turn(10), "compacted_tool_calls": 7,
             "compacted_tokens": 9_000, "compaction_anchor_id": "anchor-1"}
    update = _mw().before_model(state, None)
    assert update["compacted_tool_calls"] > 7      # previous + newly dropped
    assert update["compacted_tokens"] > 9_000
    assert update["compacted_budget_tokens"] > 9_000   # legacy state: its real total carries over


def test_stale_counters_from_a_previous_turn_are_discarded() -> None:
    # Same thread, NEW user turn: stored counters key to the OLD anchor id.
    state = {"messages": _turn(10), "compacted_tool_calls": 7,
             "compacted_tokens": 9_000, "compaction_anchor_id": "old-anchor"}
    assert compacted_counts(state) == (0, 0)
    update = _mw().before_model(state, None)
    kept_tool_msgs = sum(1 for m in update["messages"][3:] if isinstance(m, ToolMessage))
    assert update["compacted_tool_calls"] == 10 - kept_tool_msgs  # no stale +7


def test_summarizer_failure_compacts_nothing() -> None:
    assert _mw(_FakeModel(fail=True)).before_model({"messages": _turn(10)}, None) is None


def test_empty_summary_compacts_nothing() -> None:
    assert _mw(_FakeModel(text="  ")).before_model({"messages": _turn(10)}, None) is None


def test_async_path_matches_sync() -> None:
    update = asyncio.run(_mw().abefore_model({"messages": _turn(10)}, None))
    assert update is not None
    assert update["messages"][1].name == COMPACTION_SUMMARY_NAME


def test_usage_metadata_beats_chars_estimate() -> None:
    # Small text, but real usage says the context is huge → must compact.
    msgs = _turn(10, chars=10)
    msgs[9] = AIMessage("a", id="ai-4", usage_metadata={
        "input_tokens": 200_000, "output_tokens": 500, "total_tokens": 200_500})
    assert _mw(trigger=100_000).before_model({"messages": msgs}, None) is not None


def test_budget_still_bites_after_compaction() -> None:
    # 2 visible tool calls + 8 compacted (anchor-matched) = 10 ≥ hard ceiling → end.
    msgs = [HumanMessage("q", id="anchor-1"),
            ToolMessage("r1", tool_call_id="1"), ToolMessage("r2", tool_call_id="2")]
    state = {"messages": msgs, "compacted_tool_calls": 8, "compacted_tokens": 0,
             "compaction_anchor_id": "anchor-1"}
    mw = BudgetMiddleware(max_tool_calls=10, max_total_tokens=10**9)
    assert mw.before_model(state, None) == {"jump_to": "end"}
    # Stale anchor → compacted counts ignored → far under budget.
    state["compaction_anchor_id"] = "other"
    assert mw.before_model(state, None) is None


if __name__ == "__main__":
    test_below_threshold_is_a_no_op()
    test_zero_trigger_disables()
    test_compacts_summary_before_anchor_and_counts_dropped_work()
    test_tail_never_splits_an_ai_tool_pair()
    test_counters_accumulate_across_compactions()
    test_stale_counters_from_a_previous_turn_are_discarded()
    test_summarizer_failure_compacts_nothing()
    test_empty_summary_compacts_nothing()
    test_async_path_matches_sync()
    test_usage_metadata_beats_chars_estimate()
    test_budget_still_bites_after_compaction()
    print("OK — context compaction verified.")


# --- the trigger sits near the window; the summarizer's input bound follows it -------------

def test_default_trigger_is_near_the_window_and_transcript_bound_follows() -> None:
    from deep_research_agent.compaction import _MAX_TRANSCRIPT_CHARS, _transcript
    from deep_research_agent.config import ResearchConfig

    assert ResearchConfig.compaction_tokens == 600_000              # sub-agents: their context is their data
    assert ResearchConfig.orchestrator_compaction_tokens == 200_000 # orchestrator: plans + findings only
    cfg = ResearchConfig.from_runnable_config({"configurable": {"orchestrator_compaction_tokens": 0,
                                                                "compaction_tokens": 300_000}})
    assert (cfg.orchestrator_compaction_tokens, cfg.compaction_tokens) == (0, 300_000)   # per run
    assert _mw(trigger=800_000).transcript_chars() == 1_600_000
    assert _mw(trigger=5_000).transcript_chars() == _MAX_TRANSCRIPT_CHARS   # never below the floor
    long = [HumanMessage(content="x" * 1_500) for _ in range(10)]   # ~15k chars of entries
    assert "trimmed" not in _transcript(long, max_chars=100_000)
    trimmed = _transcript(long, max_chars=3_000)
    assert trimmed.startswith("[…oldest messages trimmed…]") and len(trimmed) <= 3_000 + 40


# --- the trigger follows the role model's window ----------------------------------------------

def test_compaction_trigger_is_the_lower_of_absolute_and_window_fraction() -> None:
    from deep_research_agent.compaction import compaction_trigger

    assert compaction_trigger(800_000, 1_048_576, 0.8) == 800_000       # 1M window: absolute wins
    assert compaction_trigger(800_000, 262_144, 0.8) == 209_715         # 256k window: 80% of it
    assert compaction_trigger(800_000, None, 0.8) == 170_000            # no feed: conservative
    assert compaction_trigger(100_000, None, 0.8) == 100_000            # ... unless the knob is lower
    assert compaction_trigger(800_000, 262_144, 0) == 800_000           # window rule off
    assert compaction_trigger(0, 262_144, 0.8) == 0                     # off stays off


def test_summarizer_bound_respects_the_compaction_models_window() -> None:
    assert _mw(trigger=800_000).transcript_chars() == 1_600_000
    mw = ContextCompactionMiddleware(_FakeModel(), trigger_tokens=800_000, summarizer_window=262_144)
    assert mw.transcript_chars() == int(1.5 * 262_144)                   # the summarizer's window binds
    mw = ContextCompactionMiddleware(_FakeModel(), trigger_tokens=800_000, summarizer_window=100_000)
    assert mw.transcript_chars() == 150_000      # the floor would overflow a 100k window: 1.5 chars/token cap


def test_window_is_the_smallest_the_routing_admits() -> None:
    import deep_research_agent.provider_routing as pr
    from deep_research_agent.config import ResearchConfig

    cfg = ResearchConfig.from_runnable_config({"configurable": {"openai_api_key": "k"}})

    def ep(provider, prompt, completion, ctx, uptime=99.5, status=0):
        return {"provider_name": provider, "tag": provider.lower(), "status": status,
                "uptime_last_30m": uptime, "context_length": ctx,
                "pricing": {"prompt": str(prompt / 1e6), "completion": str(completion / 1e6)}}

    feed = [ep("Cheap", 0.05, 0.16, 1_048_576), ep("Small", 0.06, 0.18, 262_144),
            ep("Pricey", 0.44, 1.32, 1_310_720), ep("Down", 0.05, 0.16, 131_072, uptime=50, status=-2)]
    routed = {"ignore": ["down"]}
    assert pr.context_window(cfg, feed, routed) == 262_144                # every healthy one; Small binds
    assert pr.context_window(cfg, feed, {"ignore": ["small"]}) == 1_048_576
    assert pr.context_window(cfg, feed, {}) == 262_144                    # relaxed: every healthy one
    everyone = {"ignore": ["cheap", "small", "pricey"]}
    assert pr.context_window(cfg, feed, everyone) == 262_144              # admits none -> all healthy
    assert pr.context_window(cfg, [], routed) is None and pr.context_window(cfg, None, routed) is None
    assert pr.context_window(cfg, [ep("NoCtx", 0.05, 0.16, None)], {}) is None


def test_context_windows_is_none_off_openrouter_and_when_the_feed_is_unreachable(monkeypatch) -> None:
    import deep_research_agent.provider_routing as pr
    from deep_research_agent.config import ResearchConfig

    async def unreachable(slug, ttl, timeout=5.0):
        return None

    monkeypatch.setattr(pr, "fetch_endpoints", unreachable)
    cfg = ResearchConfig.from_runnable_config({"configurable": {"openai_api_key": "k"}})
    assert asyncio.run(pr.context_windows(cfg, ["a/b", "a/b", "c/d"], {})) == {"a/b": None, "c/d": None}
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    local = ResearchConfig.from_runnable_config({"configurable": {"openai_api_key": "k"}})
    assert not local.is_openrouter
    assert asyncio.run(pr.context_windows(local, ["a/b"], {})) == {"a/b": None}


def test_every_role_gets_its_own_trigger_from_its_models_window(monkeypatch) -> None:
    import deep_research_agent.provider_routing as pr
    from conftest import make_graph_capture
    from deep_research_agent.config import ResearchConfig

    # `high`: planner, fleet, utility, compaction and coder are four distinct models; give the
    # fleet/compaction model (the same slug) a small window and everything else a big one.
    config = {"configurable": {"openai_api_key": "k", "mcp_servers": [], "model_tier": "high",
                               "sandbox_url": "http://sandbox.invalid:8080"}}
    cfg = ResearchConfig.from_runnable_config(config)
    assert cfg.subagent_model == cfg.compaction_model != cfg.research_model

    def ep(ctx):
        return {"provider_name": "P", "tag": "p", "status": 0, "uptime_last_30m": 99.9,
                "context_length": ctx, "pricing": {"prompt": "0.0000001", "completion": "0.0000004"}}

    async def fake_fetch(slug, ttl, timeout=5.0):
        return [ep(262_144 if slug == cfg.subagent_model else 1_050_000)]

    monkeypatch.setattr(pr, "fetch_endpoints", fake_fetch)
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    captured = make_graph_capture(monkeypatch, config)

    def compactor(mws):
        found = [m for m in mws if isinstance(m, ContextCompactionMiddleware)]
        assert len(found) == 1
        return found[0]

    orch = compactor(captured["middleware"])
    assert orch.trigger_tokens == 200_000                                # the orchestrator's own absolute
    assert orch.summarizer_window == 262_144                             # the compaction model's own window
    specs = {s["name"]: compactor(s["middleware"]) for s in captured["subagents"]}
    assert specs["research-subagent"].trigger_tokens == 209_715           # 80% of its 256k window
    assert specs["extract-subagent"].trigger_tokens == 600_000           # 1.05M window: absolute wins
    assert specs["coding-subagent"].trigger_tokens == 600_000
    assert all(c.model.model_name == cfg.compaction_model for c in specs.values())  # summarizer stays the tier's
    # Budget still runs after compaction on the orchestrator (it must see the shrunk transcript).
    mws = captured["middleware"]
    assert mws.index(orch) < mws.index(next(m for m in mws if isinstance(m, BudgetMiddleware)))
