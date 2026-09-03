"""Runaway output guard (loop_guard.py ``after_model``): a response that degenerates into
repetition, or is cut at the output cap, is trimmed to its head and answered with ONE nudge;
the second runaway in a turn ends the run. Also pins that the cap and the temperature reach
the chat model. No network."""

from __future__ import annotations

from itertools import cycle

from langchain_core.messages import AIMessage, HumanMessage

from deep_research_agent.config import ResearchConfig
from deep_research_agent.loop_guard import (RUNAWAY_KEEP_CHARS, LoopGuardMiddleware,
                                            runaway_repetition)
from deep_research_agent.models import build_chat_model
from deep_research_agent.turn import RUNAWAY_NUDGE_NAME

_WORDS = [f"{a}{b}" for a in ("etf", "spot", "whale", "halving", "fomo", "degen", "sats", "moon",
                              "dip", "ape", "chain", "fee") for b in ("", "-talk", "-flow", "-fear",
                                                                      "-cycle", "-season", "-bet",
                                                                      "-thread", "-meme", "-panic")]
# Varied findings prose: every line differs once digits are normalized.
PROSE = "\n".join(f"- {w} came up in {i * 7} posts, mostly on {src}."
                  for i, (w, src) in enumerate(zip(_WORDS, cycle(("twitter", "reddit", "telegram")))))
RUNAWAY = PROSE + "\n\nTrend words: " + ", ".join(f"`BTC{y}`" for y in range(2026, 2300))
PARAGRAPH_LOOP = "I will now analyze the full dataset in a single safe execution. " * 200
TABLE = "\n".join(f"| 2026-07-{d:02d} | {79_000 + d * 13} |" for d in range(1, 41))


def _guard(*msgs):
    state = {"messages": [HumanMessage("what is the crowd saying about BTC?"), *msgs]}
    return LoopGuardMiddleware().after_model(state, None)


def test_detector_flags_counter_loops_and_repeating_paragraphs_only() -> None:
    assert runaway_repetition(RUNAWAY) == "BTC#"          # one # per digit run
    assert runaway_repetition(PARAGRAPH_LOOP) is not None
    assert runaway_repetition(PROSE) is None                 # varied lines
    assert runaway_repetition(TABLE) is None                 # 40 rows is a table, not a runaway
    assert runaway_repetition("BTC2026, " * 50) is None      # short: under the size floor


def test_runaway_text_is_trimmed_in_place_and_nudged_once() -> None:
    update = _guard(AIMessage(RUNAWAY, id="ai-1"))
    assert update["jump_to"] == "model"
    trimmed, nudge = update["messages"]
    assert trimmed.id == "ai-1"                              # same id → replaces, not appends
    assert trimmed.content.startswith(RUNAWAY[:200])
    assert len(trimmed.content) < RUNAWAY_KEEP_CHARS + 200
    assert "[runaway output:" in trimmed.content and "BTC#" in trimmed.content
    assert "BTC2250" not in trimmed.content
    assert nudge.name == RUNAWAY_NUDGE_NAME


def test_runaway_tool_call_arguments_count_and_the_call_is_dropped() -> None:
    call = {"name": "write_file", "args": {"path": "/workspace/notes.md", "content": RUNAWAY},
            "id": "c1"}
    update = _guard(AIMessage("", id="ai-2", tool_calls=[call],
                              additional_kwargs={"tool_calls": [{"id": "c1"}], "x": 1}))
    trimmed = update["messages"][0]
    assert trimmed.tool_calls == [] and "tool_calls" not in trimmed.additional_kwargs
    assert trimmed.additional_kwargs == {"x": 1}


def test_a_response_cut_at_the_output_cap_is_treated_the_same() -> None:
    # Doubled metadata ("lengthlength") is what some OpenRouter streams produce.
    cut = AIMessage("Findings so far: the crowd is", id="ai-3",
                    response_metadata={"finish_reason": "lengthlength"})
    update = _guard(cut)
    assert update["jump_to"] == "model" and "output cap" in update["messages"][0].content


def test_second_runaway_in_the_turn_ends_the_run() -> None:
    update = _guard(AIMessage(RUNAWAY, id="ai-4"), HumanMessage("n", name=RUNAWAY_NUDGE_NAME),
                    AIMessage(RUNAWAY, id="ai-5"))
    assert update["jump_to"] == "end"
    assert [m.id for m in update["messages"]] == ["ai-5"]    # trimmed, no further nudge


def test_normal_responses_pass() -> None:
    assert _guard(AIMessage(PROSE, id="ai-6")) is None
    assert _guard(AIMessage("", id="ai-7", tool_calls=[{"name": "web_search", "args": {"q": "btc"},
                                                          "id": "c2"}])) is None


def test_cap_and_temperature_reach_the_chat_model(monkeypatch) -> None:
    for var in ("DRA_MAX_OUTPUT_TOKENS", "DRA_TEMPERATURE", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    cfg = ResearchConfig.from_runnable_config({"configurable": {}})
    assert cfg.max_output_tokens == 16_000 and cfg.temperature == 0.3
    model = build_chat_model("openai/gpt-4o-mini", cfg)
    assert model.max_tokens == 16_000 and model.temperature == 0.3
    assert model.extra_body["max_tokens"] == 16_000              # OpenRouter's own name too

    monkeypatch.setenv("DRA_MAX_OUTPUT_TOKENS", "0")
    uncapped = build_chat_model("openai/gpt-4o-mini",
                                ResearchConfig.from_runnable_config({"configurable": {}}))
    assert uncapped.max_tokens is None
    assert not (uncapped.extra_body or {}).get("max_tokens")


def test_numeric_runaway_is_described_as_pasted_data():
    from deep_research_agent.loop_guard import describe_runaway, runaway_repetition
    rows = "\n".join(f"{1_700_000_000 + i * 3600}, {12345 + i}" for i in range(400))
    assert runaway_repetition(rows) == "#"
    assert describe_runaway("#") == ("the model was pasting raw numbers inline — data rows that "
                                     "already live in a saved file")
    assert describe_runaway("BTC#") == "the model kept repeating 'BTC#'"

