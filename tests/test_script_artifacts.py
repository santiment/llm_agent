"""Scripts surface as an artifact event, never as a path in agent prose.

The leak this pins, seen live in a delivered answer:

    STATUS: ok SCRIPT: /workspace/mvrv_corr.py OUTPUT:

A sandbox path is a dead end — the reader cannot open it and the sandbox is torn down
with the run — and once the orchestrator reads one it repeats it into the report. So the
coding worker's handoff is scrubbed of file machinery before it becomes the parent's tool
result, and the CODE travels to the UI on the ``script`` event instead: app-code channel,
never model context.

Runs with plain Python (``python tests/test_script_artifacts.py``) — no pytest needed —
and is also pytest-discoverable. No network, no API keys.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research_agent.events import EVENT_SCHEMAS
from deep_research_agent.script_artifacts import (MAX_CODE_CHARS, ScriptArtifactsMiddleware,
                                                  replay_scripts, scrub_handoff)
from deep_research_agent.turn import FINDINGS_NUDGE_NAME

from conftest import capture_events_cm, make_state

LEAK = """STATUS: ok
SCRIPT: /workspace/mvrv_corr.py
OUTPUT: correlation 0.62 over 90 days
NOTES: wrote /workspace/mvrv_corr.py, then fixed mvrv_corr.py after a KeyError."""


def _write(path: str, content: str, call_id: str = "1") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": "write_file", "id": call_id,
                                      "args": {"file_path": path, "content": content}}])


def _edit(path: str, old: str, new: str, call_id: str = "2") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": "edit_file", "id": call_id,
                                      "args": {"file_path": path, "old_string": old,
                                               "new_string": new}}])


def test_scrub_strips_the_script_line_and_every_path() -> None:
    out = scrub_handoff(LEAK)
    assert "SCRIPT:" not in out
    assert "/workspace" not in out and "mvrv_corr.py" not in out
    assert "STATUS: ok" in out and "correlation 0.62 over 90 days" in out  # content survives
    assert "wrote the script, then fixed the script after a KeyError." in out


def test_scrub_keeps_the_one_path_the_caller_needs() -> None:
    # A result too large to print is handed over as a LABELED field — the caller passes it
    # to the extract worker. Structured (its own line), so a UI can hide it; prose paths
    # cannot be told apart from content and are always scrubbed.
    handoff = ("STATUS: ok\nOUTPUT: 4,812 rows scored\n"
               "RESULT FILE: /workspace/scored.json\n"
               "NOTES: results went to /workspace/scored.json via scorer.py.")
    out = scrub_handoff(handoff)
    assert "RESULT FILE: /workspace/scored.json" in out
    assert out.count("/workspace/scored.json") == 1   # the prose copy is gone
    assert "scorer.py" not in out


def test_scrub_is_a_no_op_on_a_clean_handoff() -> None:
    clean = "STATUS: ok\nOUTPUT: mean 41.2, peak 88.0 on 2026-08-14\nNOTES: assumed UTC days."
    assert scrub_handoff(clean) == clean
    assert scrub_handoff("") == ""


def test_replay_returns_the_code_that_actually_ran() -> None:
    # write -> failing run -> edit: the tab must show the FIXED script, not the first draft.
    msgs = [_write("/workspace/corr.py", "import pandas\nprint(df.mvrv)"),
            ToolMessage("KeyError: 'mvrv'", tool_call_id="1"),
            _edit("/workspace/corr.py", "df.mvrv", "df['mvrv']")]
    assert replay_scripts(msgs) == {"/workspace/corr.py": "import pandas\nprint(df['mvrv'])"}


def test_replay_ignores_data_files() -> None:
    # Only source files are artifacts; an offloaded JSON blob is data the report summarizes.
    assert replay_scripts([_write("/workspace/data/rows.json", '{"a": 1}')]) == {}


def test_handoff_emits_the_code_and_loses_the_path() -> None:
    mw = ScriptArtifactsMiddleware("coding-subagent", scrub=True)
    state = make_state(HumanMessage("Correlate MVRV with price"),
                       _write("/workspace/mvrv_corr.py", "print('corr 0.62')"),
                       ToolMessage("corr 0.62", tool_call_id="1"),
                       AIMessage(LEAK, id="final"))
    with capture_events_cm() as captured:
        update = mw.after_model(state, None)

    ev = next(e for e in captured if e.get("type") == "script")
    assert ev["code"] == "print('corr 0.62')"      # the UI tab renders THIS
    assert ev["language"] == "python" and ev["name"] == "mvrv_corr.py"  # basename, no directory
    assert ev["agent"] == "coding-subagent" and not ev["truncated"]
    assert "/workspace" not in ev["name"]
    assert EVENT_SCHEMAS["script"] <= ev.keys()

    # Same id => the message REPLACES itself in state, so deepagents forwards the scrubbed
    # text as the parent's tool result and the orchestrator never sees a path at all.
    msg = update["messages"][0]
    assert msg.id == "final"
    assert "/workspace" not in msg.content and "SCRIPT:" not in msg.content
    assert "correlation 0.62" in msg.content


def test_capture_only_mode_never_touches_the_handoff() -> None:
    # The research sub-agent writes analysis scripts too, but its handoff is findings JSON
    # guarded by its own gate — rewriting it here could break the parse.
    findings = '{"summary": "s", "findings": [], "gaps": []}'
    state = make_state(HumanMessage("unit"), _write("/workspace/a.py", "x = 1"),
                       AIMessage(findings, id="final"))
    with capture_events_cm() as captured:
        assert ScriptArtifactsMiddleware("research-subagent").after_model(state, None) is None
    assert [e["type"] for e in captured] == ["script"]


def test_nothing_fires_mid_iteration_or_without_a_script() -> None:
    mw = ScriptArtifactsMiddleware("coding-subagent", scrub=True)
    working = make_state(HumanMessage("go"), _write("/workspace/a.py", "x = 1"))
    with capture_events_cm() as captured:
        assert mw.after_model(working, None) is None   # still holding tool calls
    assert captured == []                              # one event per script, at handoff

    plain = make_state(HumanMessage("go"), AIMessage("STATUS: failed\nNOTES: no input.",
                                                     id="final"))
    with capture_events_cm() as captured:
        assert mw.after_model(plain, None) is None     # clean handoff, nothing to rewrite
    assert captured == []


def test_a_bounced_handoff_does_not_double_the_tab() -> None:
    # The findings gate bounces a sub-agent's handoff back once, so the model stops TWICE
    # in one turn. The unchanged script must not open a second tab; a script edited after
    # the bounce must, since the tab has to show the code that ran.
    mw = ScriptArtifactsMiddleware("research-subagent")
    first = [HumanMessage("unit"), _write("/workspace/a.py", "x = 1"),
             ToolMessage("ok", tool_call_id="1"), AIMessage("prose handoff", id="s1")]
    with capture_events_cm() as captured:
        mw.after_model(make_state(*first), None)
    assert len(captured) == 1

    bounced = [*first, HumanMessage("return findings JSON", name=FINDINGS_NUDGE_NAME),
               AIMessage('{"summary": "s", "findings": [], "gaps": []}', id="s2")]
    with capture_events_cm() as captured:
        mw.after_model(make_state(*bounced), None)
    assert captured == []

    fixed = [*first, HumanMessage("fix it", name=FINDINGS_NUDGE_NAME),
             _edit("/workspace/a.py", "x = 1", "x = 2"), ToolMessage("ok", tool_call_id="2"),
             AIMessage("second handoff", id="s3")]
    with capture_events_cm() as captured:
        mw.after_model(make_state(*fixed), None)
    assert [e["code"] for e in captured] == ["x = 2"]


def test_oversized_script_is_capped_and_flagged() -> None:
    big = "# pad\n" * (MAX_CODE_CHARS // 3)
    state = make_state(HumanMessage("go"), _write("/workspace/big.py", big),
                       AIMessage("STATUS: ok\nOUTPUT: done", id="final"))
    with capture_events_cm() as captured:
        ScriptArtifactsMiddleware("coding-subagent", scrub=True).after_model(state, None)
    ev = next(e for e in captured if e["type"] == "script")
    assert len(ev["code"]) == MAX_CODE_CHARS and ev["truncated"]


if __name__ == "__main__":
    test_scrub_strips_the_script_line_and_every_path()
    test_scrub_keeps_the_one_path_the_caller_needs()
    test_scrub_is_a_no_op_on_a_clean_handoff()
    test_replay_returns_the_code_that_actually_ran()
    test_replay_ignores_data_files()
    test_handoff_emits_the_code_and_loses_the_path()
    test_capture_only_mode_never_touches_the_handoff()
    test_nothing_fires_mid_iteration_or_without_a_script()
    test_a_bounced_handoff_does_not_double_the_tab()
    test_oversized_script_is_capped_and_flagged()
    print("OK — scripts ship as artifacts, not paths.")
