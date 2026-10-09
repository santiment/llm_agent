"""The sub-agent findings gate (findings_gate.py).

Pins two contracts:
  - ``extract_findings`` / ``findings_problems`` accept the mandated JSON shape
    (bare / fenced / wrapped in chatty-model prose; empty findings allowed) and
    name what is wrong otherwise;
  - ``SubagentFindingsMiddleware`` bounces a non-conforming or tool-less final
    message back to the model EXACTLY once (cap counted from message names in
    state, so parallel sub-agents sharing the instance can't interfere), and
    never blocks delivery.

Runs with plain Python (``python tests/test_findings_gate.py``) — no pytest needed —
and is also pytest-discoverable. No network, no API keys.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research_agent.findings_gate import (
    SubagentFindingsMiddleware,
    extract_findings,
    findings_problems,
)
from deep_research_agent.turn import FINDINGS_NUDGE_NAME, current_turn
from conftest import capture_events_cm, make_state

VALID = (
    '{"summary": "BTC unit done: 3 figures gathered.",'
    ' "findings": [{"finding": "Active addresses up 12% w/w",'
    ' "evidence": "1.02M vs 0.91M", "source": "Data Provider"}],'
    ' "gaps": ["funding rates unavailable"]}'
)


# --- findings contract ------------------------------------------------------

def test_valid_findings_bare_fenced_and_prose_wrapped() -> None:
    assert findings_problems(VALID) == []
    assert findings_problems(f"```json\n{VALID}\n```") == []
    assert findings_problems(f"```\n{VALID}\n```") == []
    assert findings_problems(f"Here are my findings:\n{VALID}\nHope this helps!") == []


def test_empty_findings_list_is_allowed() -> None:
    assert findings_problems('{"summary": "no data for this unit", "findings": []}') == []


def test_problems_are_named_specifically() -> None:
    assert findings_problems("just some prose, no JSON at all")  # unparseable
    probs = findings_problems('{"findings": [{"finding": "x"}]}')
    assert any('"summary"' in p for p in probs), probs
    assert any('"source"' in p for p in probs), probs
    probs = findings_problems('{"summary": "s", "findings": "not-a-list"}')
    assert any("must be a list" in p for p in probs), probs
    probs = findings_problems('{"summary": "s", "findings": [], "gaps": "oops"}')
    assert any('"gaps"' in p for p in probs), probs


def test_extract_findings_none_on_garbage() -> None:
    assert extract_findings("") is None
    assert extract_findings("{broken json") is None
    assert extract_findings('["a", "list"]') is None  # an object is required


def test_chatty_model_preamble_objects() -> None:
    # Two objects in prose: each must parse independently; the findings-shaped
    # one (the last) wins.
    two = f'{{"status": "thinking"}} ok, here it is: {VALID}'
    assert findings_problems(two) == [], findings_problems(two)
    # A diagnostic fence BEFORE the real fenced findings.
    fences = f'```json\n{{"status": "ok"}}\n```\nresult:\n```json\n{VALID}\n```'
    assert findings_problems(fences) == [], findings_problems(fences)
    # Preamble object + findings object NOT in a fence, with trailing prose.
    mixed = f'{{"note": 1}}\n{VALID}\ndone!'
    assert findings_problems(mixed) == []
    # First fence has trailing junk between "}" and its closing fence — must not
    # spoil extraction of the clean second fence.
    junk = f'```json\n{{"status": "ok"}} note\n```\n```json\n{VALID}\n```'
    assert findings_problems(junk) == [], findings_problems(junk)


def test_evidence_is_optional_in_validator() -> None:
    no_evidence = ('{"summary": "s", "findings": '
                   '[{"finding": "x up 5%", "source": "Data Provider"}]}')
    assert findings_problems(no_evidence) == []


# --- middleware -------------------------------------------------------------

def test_bounces_prose_once_then_accepts() -> None:
    mw = SubagentFindingsMiddleware()
    work = [HumanMessage("unit: BTC"), ToolMessage("rows", tool_call_id="1")]
    bad = AIMessage("I looked at BTC and it seems fine.")

    update = mw.after_model(make_state(*work, bad), None)
    assert update and update.get("jump_to") == "model", update
    nudge = update["messages"][0]
    assert getattr(nudge, "name", None) == FINDINGS_NUDGE_NAME
    assert "RETURN FORMAT" in nudge.content  # points at the prompt, no schema restated

    # Same instance, nudge now present in state -> accepted as-is (cap is in state,
    # not on the instance).
    again = mw.after_model(make_state(*work, nudge, bad), None)
    assert again is None


def test_valid_json_and_tool_calls_pass_through() -> None:
    mw = SubagentFindingsMiddleware()
    did_work = ToolMessage("rows", tool_call_id="1")
    assert mw.after_model(make_state(did_work, AIMessage(VALID)), None) is None
    working = AIMessage("", tool_calls=[
        {"name": "web_search", "args": {"query": "x"}, "id": "1"}])
    assert mw.after_model(make_state(working), None) is None
    assert mw.after_model(make_state(AIMessage("")), None) is None        # empty content
    assert mw.after_model(make_state(HumanMessage("hi")), None) is None   # not an AIMessage
    assert mw.after_model(make_state(), None) is None                     # no messages


def test_provenance_findings_without_tools_bounce() -> None:
    mw = SubagentFindingsMiddleware()
    # Non-empty findings with ZERO tool calls in state -> fabricated from memory -> bounce.
    update = mw.after_model(make_state(HumanMessage("unit: BTC"), AIMessage(VALID)), None)
    assert update and update.get("jump_to") == "model", update
    assert "tool" in update["messages"][0].content
    # Honest empty findings with no tool calls is legitimate -> accepted.
    empty = '{"summary": "no data available for this unit", "findings": []}'
    assert mw.after_model(make_state(HumanMessage("unit: X"), AIMessage(empty)), None) is None


def test_findings_nudge_is_not_a_turn_boundary() -> None:
    nudge = HumanMessage("fix the format", name=FINDINGS_NUDGE_NAME)
    msgs = [HumanMessage("real user turn"), AIMessage("prose"), nudge, AIMessage(VALID)]
    turn = current_turn(msgs)
    assert turn[0] is msgs[0], "findings nudge must not start a new turn"


def test_accepted_findings_emit_structured_event() -> None:
    # On clean accept, a `subagent_findings` event fires carrying the parsed object +
    # a unit label (the sub-agent's task assignment) — that's what the UI renders.
    with capture_events_cm() as captured:
        mw = SubagentFindingsMiddleware()
        state = make_state(HumanMessage("Research BTC on-chain activity"),
                           ToolMessage("rows", tool_call_id="1"), AIMessage(VALID))
        assert mw.after_model(state, None) is None  # accepted

    ev = next((e for e in captured if e.get("type") == "subagent_findings"), None)
    assert ev, captured
    assert ev["unit"] == "Research BTC on-chain activity"
    assert ev["summary"] and isinstance(ev["findings"], list) and ev["findings"]
    assert ev["findings"][0]["source"] == "Data Provider"


if __name__ == "__main__":
    test_valid_findings_bare_fenced_and_prose_wrapped()
    test_empty_findings_list_is_allowed()
    test_problems_are_named_specifically()
    test_extract_findings_none_on_garbage()
    test_chatty_model_preamble_objects()
    test_evidence_is_optional_in_validator()
    test_bounces_prose_once_then_accepts()
    test_valid_json_and_tool_calls_pass_through()
    test_provenance_findings_without_tools_bounce()
    test_findings_nudge_is_not_a_turn_boundary()
    test_accepted_findings_emit_structured_event()
    print("OK — structured-findings gate verified.")


def test_findings_evidence_with_raw_series_is_bounced() -> None:
    import json

    rows = "\n".join(f"2026-09-01T{h:02d}:00:00Z: bearish=0.05, bullish=0.38, neutral=0.57"
                     for h in range(9, 16))
    obj = {"summary": "Mood stayed bullish.",
           "findings": [{"finding": "Bullish share held near 38%.", "evidence": rows,
                         "source": "Santiment social messages"}]}
    probs = findings_problems(json.dumps(obj))
    assert any("transcribes a time series" in p for p in probs)
    obj["findings"][0]["evidence"] = "bullish share 0.36–0.41 across 7 hours, flat, peak 11:00"
    assert findings_problems(json.dumps(obj)) == []


def test_a_series_pasted_into_any_field_is_bounced() -> None:
    """The case that reached a user's screen: 70 `date,value` points on ONE line, in
    "finding" rather than "evidence" — the line-based series check saw nothing."""
    import json

    series = "; ".join(f"2026-06-{d:02d},-0.20{d}" for d in range(5, 25))
    for field in ("finding", "evidence", "summary"):
        obj = {"summary": "MVRV stayed negative.",
               "findings": [{"finding": "MVRV held near -0.21.", "source": "Santiment"}]}
        if field == "summary":
            obj["summary"] = f"Complete daily series: {series}"
        else:
            obj["findings"][0][field] = f"Complete daily series: {series}"
        probs = findings_problems(json.dumps(obj))
        assert any("transcribes a time series" in p for p in probs), (field, probs)
        assert any("20 dated points" in p for p in probs), (field, probs)
        assert any("CONCLUSION" in p for p in probs), (field, probs)

    # Citing a few dated values is normal prose, not a transcription.
    fine = {"summary": "MVRV fell from -0.2026 on 2026-06-05 to -0.2120 on 2026-08-13.",
            "findings": [{"finding": "Trough -0.2211 on 2026-07-21, mean -0.211, flat.",
                          "source": "Santiment"}]}
    assert findings_problems(json.dumps(fine)) == []


def test_an_oversize_field_is_bounced_as_a_dump() -> None:
    """A message/post list has no dates to spot, so sheer length is the backstop."""
    import json

    from deep_research_agent.findings_gate import MAX_FIELD_CHARS

    dump = "; ".join(f"user{i} said the market looks strong right now" for i in range(40))
    assert len(dump) > MAX_FIELD_CHARS
    obj = {"summary": "Crowd is bullish.",
           "findings": [{"finding": "62% of 480 messages are bullish.", "evidence": dump,
                         "source": "Santiment social messages"}]}
    probs = findings_problems(json.dumps(obj))
    assert any("data dump, not a finding" in p for p in probs), probs

    obj["findings"][0]["evidence"] = 'bullish 62% of 480; typical: "market looks strong"'
    assert findings_problems(json.dumps(obj)) == []


def test_a_source_field_is_never_flagged_for_length_or_dates() -> None:
    import json

    obj = {"summary": "s", "findings": [
        {"finding": "f", "source": "https://example.com/data?from=2026-06-05,1&to=2026-06-06,2"}]}
    assert findings_problems(json.dumps(obj)) == []


def test_source_that_names_a_file_path_or_recipe_call_is_bounced() -> None:
    import json
    bad = json.dumps({"summary": "s", "findings": [
        {"finding": "support at 77k (11 voices)", "source": "R.price_levels(d) on /workspace/data/social_messages-6debc408.json"},
        {"finding": "94th pct", "source": "price_usd_90d.json"},
        {"finding": "organic 62%", "source": "computed via execute over the offloaded file"},
        {"finding": "fine", "source": "Santiment social messages"},
        {"finding": "fine too", "source": "https://example.com/execute-order/data.json"},
    ]})
    probs = findings_problems(bad)
    assert len(probs) == 3, probs
    assert all("names a file, path or function" in p for p in probs)
    assert "findings[0]" in probs[0] and "findings[1]" in probs[1] and "findings[2]" in probs[2]


def test_exhausted_nudge_with_a_pasted_table_is_sanitized_not_accepted() -> None:
    # The route a 31-row markdown table took to the orchestrator: bounced once, then
    # accepted verbatim. Now the rows are collapsed and the rest of the handoff is kept.
    import json

    table = "| date | price_usd |\n|---|---|\n" + "\n".join(
        f"| 2026-08-{d:02d} | {63000 + d * 137.5} |" for d in range(10, 22))
    obj = {"summary": "Full contents of the file:\n" + table,
           "findings": [{"finding": "The file has 12 rows.", "evidence": table,
                         "source": "Santiment"}], "gaps": []}
    bad = AIMessage("```json\n" + json.dumps(obj) + "\n```")
    nudge = HumanMessage("fix", name=FINDINGS_NUDGE_NAME)
    work = [HumanMessage("unit: BTC"), ToolMessage("rows", tool_call_id="1")]

    mw = SubagentFindingsMiddleware()
    with capture_events_cm() as captured:
        update = mw.after_model(make_state(*work, nudge, bad), None)
    assert update and "jump_to" not in update                  # accepted, but rewritten
    cleaned = update["messages"][0]
    assert cleaned.id == bad.id                                # same id: replaces in state
    text = cleaned.content
    assert "2026-08-15" not in text and "Raw series of 12 timestamped rows" in text
    assert "The file has 12 rows." in text and '"source": "Santiment"' in text
    ev = next(e for e in captured if e["type"] == "subagent_findings")
    assert "2026-08-15" not in json.dumps(ev)                  # the UI card is clean too


def test_rows_pasted_outside_the_json_object_are_caught() -> None:
    # The parent receives the WHOLE message. A run shipped 31 rows to the orchestrator as
    # text AFTER a clean JSON object — the object passed, the message did not get checked.
    rows = "\n".join(f"2026-08-{d:02d}, {63000 + d * 137.5}" for d in range(10, 22))
    msg = AIMessage("All points retrieved.\n\n" + VALID + "\n\nFULL DAILY SERIES:\n" + rows)
    work = [HumanMessage("unit: BTC"), ToolMessage("rows", tool_call_id="1")]
    mw = SubagentFindingsMiddleware()

    update = mw.after_model(make_state(*work, msg), None)
    assert update and update.get("jump_to") == "model"
    assert "text outside the JSON object" in update["messages"][0].content

    # Nudge spent: the handoff is normalized to the bare object — the trailing rows are gone.
    nudge = HumanMessage("fix", name=FINDINGS_NUDGE_NAME)
    again = mw.after_model(make_state(*work, nudge, msg), None)
    assert again and "jump_to" not in again
    text = again["messages"][0].content
    assert "2026-08-15" not in text and "FULL DAILY SERIES" not in text
    assert "Active addresses up 12% w/w" in text


def test_prose_around_a_clean_object_is_trimmed_on_accept() -> None:
    # Harmless preamble is not a bounce, but the parent still gets only the object.
    work = [HumanMessage("unit: BTC"), ToolMessage("rows", tool_call_id="1")]
    update = SubagentFindingsMiddleware().after_model(
        make_state(*work, AIMessage("Here are my findings:\n\n" + VALID)), None)
    assert update and "jump_to" not in update
    assert update["messages"][0].content.startswith("```json\n{")
    assert "Here are my findings" not in update["messages"][0].content


def test_a_dump_inside_a_field_is_reported_once_not_also_as_outside_text() -> None:
    import json

    rows = "\n".join(f"2026-08-{d:02d}, {63000 + d}" for d in range(10, 22))
    obj = {"summary": "ok", "findings": [{"finding": "x", "evidence": rows, "source": "S"}]}
    work = [HumanMessage("u"), ToolMessage("r", tool_call_id="1")]
    update = SubagentFindingsMiddleware().after_model(
        make_state(*work, AIMessage("```json\n" + json.dumps(obj) + "\n```")), None)
    text = update["messages"][0].content
    assert text.count("transcribes a time series") == 1
    assert "outside the JSON object" not in text


def test_a_fence_without_a_newline_still_counts_as_the_bare_object() -> None:
    work = [HumanMessage("u"), ToolMessage("r", tool_call_id="1")]
    assert SubagentFindingsMiddleware().after_model(
        make_state(*work, AIMessage("```json" + VALID + "```")), None) is None


# --- batched briefs: two or three closely related questions, each answered or gapped by tag ----

def test_brief_questions_reads_the_numbered_tags_from_the_brief_only() -> None:
    from deep_research_agent.findings_gate import brief_questions

    brief = HumanMessage("FILE: /workspace/x.json\nQUESTIONS:\n  Q1: themes\n  Q2: claims\n Q3) split\n")
    assert brief_questions([brief, ToolMessage("rows", tool_call_id="1")]) == ["Q1", "Q2", "Q3"]
    assert brief_questions([HumanMessage("QUESTION: themes only")]) == []
    nudge = HumanMessage("Q1: this is a nudge, not the brief", name=FINDINGS_NUDGE_NAME)
    assert brief_questions([HumanMessage("Q1: real\nQ2: brief"), nudge]) == ["Q1", "Q2"]
    assert brief_questions([HumanMessage("Q1: a\nQ4: beyond the batch cap\nQ1: again")]) == []  # one tag: no batch
    assert brief_questions([HumanMessage("- **Q1**: bold\n(Q2) bracketed\nsee Q3 mid-sentence")]) == ["Q1", "Q2"]
    assert brief_questions([HumanMessage("q1: lower\nq2: case")]) == ["Q1", "Q2"]


def test_quarters_in_a_brief_are_periods_not_questions() -> None:
    # A research unit is often a reporting period: "Q3 2026" must never read as a question tag.
    from deep_research_agent.findings_gate import brief_questions

    periods = HumanMessage("Research Coinbase quarterly results.\nPERIODS:\n- Q2 2026 revenue\n- Q3 2026 revenue")
    assert brief_questions([periods]) == []
    gapped = HumanMessage("Q2: $1.2B revenue\nQ3: $1.5B revenue")          # delimited, but not from Q1
    assert brief_questions([gapped]) == []


def test_the_brief_survives_compaction() -> None:
    # Compaction puts its summary FIRST; the brief is the turn's real human message.
    from deep_research_agent.findings_gate import brief_questions
    from deep_research_agent.turn import COMPACTION_SUMMARY_NAME

    summary = HumanMessage("Summary of earlier work: Q3 2026 data read.", name=COMPACTION_SUMMARY_NAME)
    assert brief_questions([summary, HumanMessage("Q1: themes\nQ2: claims")]) == ["Q1", "Q2"]


def test_one_finding_may_answer_several_questions() -> None:
    from deep_research_agent.findings_gate import coverage_problems

    obj = {"summary": "s", "findings": [{"finding": "Q1/Q2: ETF theme, BlackRock flows", "source": "S"}],
           "gaps": "Q3: not determinable — no rows"}
    assert coverage_problems(obj, ["Q1", "Q2", "Q3"]) == []
    assert coverage_problems({"summary": "s", "findings": [{"finding": "Q3 2026 revenue rose", "source": "S"}]},
                             ["Q1", "Q2", "Q3"])                           # a period is not a tag


def test_research_subagent_never_checks_question_tags() -> None:
    mw = SubagentFindingsMiddleware()                                     # the research-subagent's gate
    brief = HumanMessage("Q1: themes\nQ2: claims")
    ok = AIMessage('{"summary": "s", "findings": [{"finding": "ETF talk (41 msgs)",'
                   ' "evidence": "41 of 200", "source": "S"}], "gaps": []}')
    assert mw.after_model(make_state(brief, ToolMessage("rows", tool_call_id="1"), ok), None) is None


def test_coverage_needs_a_tagged_finding_or_gap_per_question() -> None:
    from deep_research_agent.findings_gate import coverage_problems

    obj = {"summary": "s", "findings": [{"finding": "Q1: ETF talk dominates (41 msgs)", "source": "S"}],
           "gaps": ["Q2: not determinable — no price targets named"]}
    assert coverage_problems(obj, ["Q1", "Q2"]) == []
    problems = coverage_problems(obj, ["Q1", "Q2", "Q3"])
    assert len(problems) == 1 and problems[0].startswith("Q3:")
    assert coverage_problems(obj, []) == []                       # single-question brief: no rule
    assert coverage_problems(None, ["Q1"]) == []                  # shape problems are reported elsewhere
    untagged = {"summary": "s", "findings": [{"finding": "ETF talk dominates", "source": "S"}], "gaps": []}
    assert [p[:6] for p in coverage_problems(untagged, ["Q1", "Q2"])] == ["Q1, Q2"]
    styled = {"summary": "s", "findings": [{"finding": "**Q1** — ETF talk", "source": "S"}],
              "gaps": ["(Q2) not determinable"]}
    assert coverage_problems(styled, ["Q1", "Q2"]) == []


def test_batched_brief_bounces_a_skipped_question_once_then_accepts_when_gapped() -> None:
    mw = SubagentFindingsMiddleware(batched_questions=True)
    brief = HumanMessage("FILE: /workspace/x.json\nQUESTIONS:\nQ1: themes\nQ2: claims\nSOURCE LABEL: S")
    work = [brief, ToolMessage("rows", tool_call_id="1")]
    partial = AIMessage('{"summary": "themes read", "findings": [{"finding": "Q1: ETF talk (41 msgs)",'
                        ' "evidence": "41 of 200 random", "source": "S"}], "gaps": []}')
    update = mw.after_model(make_state(*work, partial), None)
    assert update and update.get("jump_to") == "model"
    assert "Q2" in update["messages"][0].content and "Q1" not in update["messages"][0].content.split("Q2")[0][-4:]

    complete = AIMessage('{"summary": "themes read", "findings": [{"finding": "Q1: ETF talk (41 msgs)",'
                         ' "evidence": "41 of 200 random", "source": "S"}],'
                         ' "gaps": ["Q2: not determinable — no checkable claims in the sample"]}')
    with capture_events_cm() as events:
        accepted = mw.after_model(make_state(*work, complete), None)
    assert accepted is None                                             # already the bare object: accepted as-is
    assert [e["type"] for e in events] == ["subagent_findings"]


def test_prompts_and_skill_batch_only_closely_related_questions() -> None:
    from pathlib import Path
    from deep_research_agent import prompts

    assert "NUMBERED QUESTIONS (`Q1:`–`Q3:` in your task)" in prompts.extract_prompt("")
    sub = prompts.subagent_prompt("", "")
    assert "ONE question per task" in sub and "Never more than three per task" in sub
    assert "two or three CLOSELY RELATED" in prompts.orchestrator_prompt("", "")
    skill = Path(__file__).resolve().parents[1].joinpath("skills/crowd-positioning/signals.md").read_text()
    assert "go in ONE\n`task(" in skill and "Q1: <Themes" in skill and "one task each" not in skill


def test_after_the_nudge_a_still_skipped_question_becomes_an_explicit_gap() -> None:
    mw = SubagentFindingsMiddleware(batched_questions=True)
    brief = HumanMessage("QUESTIONS:\nQ1: themes\nQ2: claims\nQ3: split\nSOURCE LABEL: S")
    nudge = HumanMessage("fix it", name=FINDINGS_NUDGE_NAME)
    still_partial = AIMessage('{"summary": "s", "findings": [{"finding": "Q1: ETF talk (41 msgs)",'
                              ' "source": "S"}], "gaps": ["Q3: not determinable — too few messages"]}')
    with capture_events_cm() as events:
        update = mw.after_model(make_state(brief, ToolMessage("rows", tool_call_id="1"), nudge, still_partial), None)
    assert update and update.get("jump_to") is None                      # accepted, not bounced again
    handed = json.loads(update["messages"][0].content.strip("`json\n"))
    assert handed["gaps"] == ["Q3: not determinable — too few messages",
                              "Q2: not answered — ask it again in its own task"]
    assert events[0]["type"] == "subagent_findings" and events[0]["gaps"] == handed["gaps"]
