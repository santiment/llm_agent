"""Report delivery channel — the prose-report seam found by a live extra-low run.

A cheap orchestrator did the research, then wrote the whole summary as a plain chat
message instead of calling ``submit_report``. The old behavior accepted it silently
(``_looks_delivered`` suppressed the nudge to avoid apology loops) and the citations
salvage delivered it with an error-toned status. Pins the new contract:

  - a delivered-looking prose ending after research gets ONE mechanical
    resubmit-verbatim nudge (``RESUBMIT_NUDGE_NAME``) — routing the text through
    ``submit_report`` and its quality gate when the model complies;
  - if prose persists after the nudge, it is accepted (the salvage delivers it) —
    never a second nag;
  - the salvage end-state is ``done`` / ``report_salvaged`` — a recovery, not an
    error — and the resubmit nudge is not a turn boundary.

Runs with plain Python (``python tests/test_report_delivery.py``) — no pytest needed —
and is also pytest-discoverable.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research_agent.citations import ResearchOutputMiddleware
from deep_research_agent.completion import ForceCompletionMiddleware
from deep_research_agent.turn import NUDGE_NAME, RESUBMIT_NUDGE_NAME, current_turn

PROSE_REPORT = (
    "Here's the summary of the 5 Santiment MCP metric calls:\n\n"
    "Call 1 — price_usd (Bitcoin): success, range $60,866 → $63,808.\n"
    "Call 2 — marketcap_usd (Ethereum): success, ~$189B → ~$213B.\n"
    "Call 3 — dev_activity: failed with a did-you-mean suggestion (dev_activity_1d).\n"
    "Call 4 — daily_active_addresses: success, ~220K to ~724K.\n"
    "Call 5 — nvtstr: the intentionally wrong one; failed cleanly with guidance.\n\n"
    "Bottom line: 3 of 5 calls succeeded; the MCP validates names and returns "
    "descriptive, actionable errors instead of crashing the run."
)

_WORK = [HumanMessage("test the MCP"), ToolMessage("rows", tool_call_id="1")]


def _state(*messages) -> dict:
    return {"messages": list(messages)}


def test_prose_report_gets_one_resubmit_nudge() -> None:
    mw = ForceCompletionMiddleware()
    update = mw.after_model(_state(*_WORK, AIMessage(PROSE_REPORT)), None)
    assert update and update.get("jump_to") == "model", update
    nudge = update["messages"][0]
    assert getattr(nudge, "name", None) == RESUBMIT_NUDGE_NAME
    assert "EXACTLY" in nudge.content and "verbatim" in nudge.content
    assert "submit_report" in nudge.content


def test_persistent_prose_is_accepted_after_the_nudge() -> None:
    mw = ForceCompletionMiddleware()
    nudge = HumanMessage("resubmit", name=RESUBMIT_NUDGE_NAME)
    again = mw.after_model(
        _state(*_WORK, AIMessage(PROSE_REPORT), nudge, AIMessage(PROSE_REPORT)), None)
    assert again is None  # salvage delivers; no second nag


def test_short_intent_stub_still_gets_the_regular_nudge() -> None:
    mw = ForceCompletionMiddleware()
    update = mw.after_model(_state(*_WORK, AIMessage("Now I will compare the data.")), None)
    assert update and update.get("jump_to") == "model"
    assert getattr(update["messages"][0], "name", None) == NUDGE_NAME


def test_direct_answer_without_research_is_untouched() -> None:
    mw = ForceCompletionMiddleware()
    state = _state(HumanMessage("what is the capital of Bulgaria?"),
                   AIMessage("Sofia. " * 80))  # long, but no research this turn
    assert mw.after_model(state, None) is None


def test_resubmit_nudge_is_not_a_turn_boundary() -> None:
    nudge = HumanMessage("resubmit", name=RESUBMIT_NUDGE_NAME)
    msgs = [HumanMessage("real turn"), AIMessage(PROSE_REPORT), nudge, AIMessage("x")]
    assert current_turn(msgs)[0] is msgs[0]


def test_salvage_classifies_as_done_not_error() -> None:
    mw = ResearchOutputMiddleware(max_tool_calls=80, max_total_tokens=1_000_000)
    state, reason, detail = mw._classify(
        via_tool=False, researched=True, salvaged=True, clarified=False,
        calls=7, tokens=10_000, nudges=0)
    assert state == "done" and reason == "report_salvaged", (state, reason)
    assert "recovered" in detail
    # The no-report-at-all ending stays an error.
    state, reason, _ = mw._classify(
        via_tool=False, researched=True, salvaged=False, clarified=False,
        calls=7, tokens=10_000, nudges=0)
    assert state == "error" and reason == "ended_without_report"


if __name__ == "__main__":
    test_prose_report_gets_one_resubmit_nudge()
    test_persistent_prose_is_accepted_after_the_nudge()
    test_short_intent_stub_still_gets_the_regular_nudge()
    test_direct_answer_without_research_is_untouched()
    test_resubmit_nudge_is_not_a_turn_boundary()
    test_salvage_classifies_as_done_not_error()
    print("OK — prose reports get one verbatim resubmit nudge, salvage is a recovery.")
