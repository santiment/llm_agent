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

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research_agent.findings_gate import (
    SubagentFindingsMiddleware,
    extract_findings,
    findings_problems,
)
from deep_research_agent.turn import FINDINGS_NUDGE_NAME, current_turn

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

def _state(*messages) -> dict:
    return {"messages": list(messages)}


def test_bounces_prose_once_then_accepts() -> None:
    mw = SubagentFindingsMiddleware()
    work = [HumanMessage("unit: BTC"), ToolMessage("rows", tool_call_id="1")]
    bad = AIMessage("I looked at BTC and it seems fine.")

    update = mw.after_model(_state(*work, bad), None)
    assert update and update.get("jump_to") == "model", update
    nudge = update["messages"][0]
    assert getattr(nudge, "name", None) == FINDINGS_NUDGE_NAME
    assert "RETURN FORMAT" in nudge.content  # points at the prompt, no schema restated

    # Same instance, nudge now present in state -> accepted as-is (cap is in state,
    # not on the instance).
    again = mw.after_model(_state(*work, nudge, bad), None)
    assert again is None


def test_valid_json_and_tool_calls_pass_through() -> None:
    mw = SubagentFindingsMiddleware()
    did_work = ToolMessage("rows", tool_call_id="1")
    assert mw.after_model(_state(did_work, AIMessage(VALID)), None) is None
    working = AIMessage("", tool_calls=[
        {"name": "web_search", "args": {"query": "x"}, "id": "1"}])
    assert mw.after_model(_state(working), None) is None
    assert mw.after_model(_state(AIMessage("")), None) is None        # empty content
    assert mw.after_model(_state(HumanMessage("hi")), None) is None   # not an AIMessage
    assert mw.after_model(_state(), None) is None                     # no messages


def test_provenance_findings_without_tools_bounce() -> None:
    mw = SubagentFindingsMiddleware()
    # Non-empty findings with ZERO tool calls in state -> fabricated from memory -> bounce.
    update = mw.after_model(_state(HumanMessage("unit: BTC"), AIMessage(VALID)), None)
    assert update and update.get("jump_to") == "model", update
    assert "tool" in update["messages"][0].content
    # Honest empty findings with no tool calls is legitimate -> accepted.
    empty = '{"summary": "no data available for this unit", "findings": []}'
    assert mw.after_model(_state(HumanMessage("unit: X"), AIMessage(empty)), None) is None


def test_findings_nudge_is_not_a_turn_boundary() -> None:
    nudge = HumanMessage("fix the format", name=FINDINGS_NUDGE_NAME)
    msgs = [HumanMessage("real user turn"), AIMessage("prose"), nudge, AIMessage(VALID)]
    turn = current_turn(msgs)
    assert turn[0] is msgs[0], "findings nudge must not start a new turn"


def test_accepted_findings_emit_structured_event(monkeypatch=None) -> None:
    # On clean accept, a `subagent_findings` event fires carrying the parsed object +
    # a unit label (the sub-agent's task assignment) — that's what the UI renders.
    import deep_research_agent.findings_gate as fg

    captured = []
    orig = fg.emit
    fg.emit = lambda ev: captured.append(ev)
    try:
        mw = SubagentFindingsMiddleware()
        state = _state(HumanMessage("Research BTC on-chain activity"),
                       ToolMessage("rows", tool_call_id="1"), AIMessage(VALID))
        assert mw.after_model(state, None) is None  # accepted
    finally:
        fg.emit = orig

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
