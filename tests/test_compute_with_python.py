"""Computation goes through Python — never through the model's head.

Found live: "compute square root of 9238123123123123123123123" was classified SIMPLE by the
pre-triage router, whose answer path holds NO tools, so the agent replied with a hand-waved
"approximately 3,039,500,000,000" (the exact value is 3,039,428,091,454.56 — 71.9M off). The
contract pinned here:

  - the router never keeps a calculation on the tool-less SIMPLE path;
  - the orchestrator and the sub-agents are told, in their system prompts, that every number
    they report is printed by Python, not estimated;
  - a turn whose ONLY tool was `execute` is a COMPUTE turn: it ends as plain text with the
    number, with no report nudge and no "ended without report" error;
  - every role that can run code carries `ExecuteArtifactsMiddleware`, so the code it ran
    reaches the UI (the run above computed the exact value and showed the user nothing).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research_agent import prompts
from deep_research_agent.citations import ResearchOutputMiddleware
from deep_research_agent.completion import ForceCompletionMiddleware
from deep_research_agent.script_artifacts import ExecuteArtifactsMiddleware
from deep_research_agent.triage import ROUTER_PROMPT
from deep_research_agent.turn import did_research_work

from conftest import make_graph_capture, make_state


def _calc_turn(answer: str) -> dict:
    """An orchestrator COMPUTE turn: one `execute` call, its output, a plain-text answer."""
    call = {"name": "execute", "args": {"command": "python3 - <<'PY'\nprint(1)\nPY"}, "id": "c1"}
    return make_state(
        HumanMessage("compute square root of 9238123123123123123123123"),
        AIMessage("", tool_calls=[call]),
        ToolMessage("3039428091454", tool_call_id="c1", name="execute"),
        AIMessage(answer),
    )


def test_router_never_keeps_a_calculation_on_the_simple_path() -> None:
    assert "NO CALCULATION" in ROUTER_PROMPT
    assert "ANY COMPUTATION IS RESEARCH" in ROUTER_PROMPT
    assert "never estimate a number" in ROUTER_PROMPT
    assert "unit conversions" not in ROUTER_PROMPT  # was listed as SIMPLE; it needs Python


def test_orchestrator_prompt_has_a_compute_class_and_the_arithmetic_rule() -> None:
    p = prompts.orchestrator_prompt("", "")
    assert "- COMPUTE:" in p
    assert "PYTHON DOES THE ARITHMETIC, ALWAYS" in p
    assert "never by mental arithmetic" in p
    assert "math.isqrt" in p  # exact big-integer math, not float sqrt
    assert "no todos, no sub-agents, no `submit_report`" in p


def test_subagent_and_extract_prompts_forbid_estimating_a_figure() -> None:
    assert "COMPUTE, NEVER ESTIMATE" in prompts.subagent_prompt("", "")
    assert "Never eyeball, estimate or round a count" in prompts.extract_prompt("")


def test_an_execute_only_turn_is_a_direct_answer_not_research() -> None:
    turn = _calc_turn("The square root is 3,039,428,091,454.56.")["messages"]
    assert did_research_work(turn) is False
    # Any gather/plan tool alongside the calculation makes it a research turn again.
    with_task = turn + [AIMessage("", tool_calls=[{"name": "task", "args": {}, "id": "t1"}])]
    assert did_research_work(with_task) is True


def test_a_computed_answer_is_not_nudged_into_a_report() -> None:
    mw = ForceCompletionMiddleware()
    assert mw.after_model(_calc_turn("The square root is 3,039,428,091,454.56."), None) is None


def test_every_role_that_can_run_code_emits_its_script(monkeypatch) -> None:
    """The other half of the live gap: the number was computed for real and the user saw no
    sign of it. `ExecuteArtifactsMiddleware` must be mounted on the orchestrator (which
    computes with `execute` and holds no file tools) and on every sub-agent that runs code."""
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    captured = make_graph_capture(monkeypatch, {"configurable": {
        "openai_api_key": "test-key", "mcp_servers": [],
        "sandbox_url": "http://sandbox.invalid:8080"}})

    def labels(middleware) -> set[str]:
        return {m.agent for m in middleware if isinstance(m, ExecuteArtifactsMiddleware)}

    assert labels(captured["middleware"]) == {"orchestrator"}
    for spec in captured["subagents"]:
        assert labels(spec["middleware"]) == {spec["name"]}, spec["name"]


def test_a_computed_answer_ends_the_run_as_done() -> None:
    mw = ResearchOutputMiddleware(max_tool_calls=80, max_total_tokens=1_000_000)
    state, reason, _ = mw._classify(
        via_tool=False, researched=False, salvaged=False, clarified=False,
        calls=1, tokens=3_000, nudges=0)
    assert (state, reason) == ("done", "direct_answer")


if __name__ == "__main__":
    test_router_never_keeps_a_calculation_on_the_simple_path()
    test_orchestrator_prompt_has_a_compute_class_and_the_arithmetic_rule()
    test_subagent_and_extract_prompts_forbid_estimating_a_figure()
    test_an_execute_only_turn_is_a_direct_answer_not_research()
    test_a_computed_answer_is_not_nudged_into_a_report()
    test_a_computed_answer_ends_the_run_as_done()
    # test_every_role_that_can_run_code_emits_its_script needs pytest's monkeypatch fixture.
    print("ok")
