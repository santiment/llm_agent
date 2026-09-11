"""coding-subagent wiring: a dedicated coder that writes/fixes ONE script per task.

Registered whenever a sandbox is configured (it needs only `execute` + the file tools, not
offloading), runs on cfg.coding_model, holds no data tools, keeps write/edit/read but loses
grep + write_todos, carries no findings gate (its handoff is output + notes), and is
nested into the research-subagent's `task` tool next to the extract-subagent. Its scripts
reach the user as `script` EVENTS (see test_script_artifacts.py), never as a path in prose.
The prompt work that sends failed `execute` calls to it — instead of narrated `python -c`
quoting retries — is pinned here too.
"""

from __future__ import annotations

import asyncio

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware

import deep_research_agent.agent as agent_mod
from deep_research_agent.config import MODEL_TIERS
from deep_research_agent.findings_gate import SubagentFindingsMiddleware
from deep_research_agent.metering import SubagentUsageMiddleware
from deep_research_agent.model_errors import ModelBackoffMiddleware
from deep_research_agent.prompts import (coding_prompt, extract_prompt, orchestrator_prompt,
                                         subagent_prompt)
from deep_research_agent.script_artifacts import ScriptArtifactsMiddleware
from deep_research_agent.tool_filter import CODING_EXCLUDED_TOOLS, ExcludeToolsMiddleware
from conftest import make_graph_capture


def _sandbox_config(**extra) -> dict:
    return {"configurable": {"openai_api_key": "test-key", "mcp_servers": [],
                             "sandbox_url": "http://sandbox.invalid:8080", **extra}}


def _coding_spec(captured: dict) -> dict:
    return next(s for s in captured["subagents"] if s["name"] == "coding-subagent")


def test_no_sandbox_no_coding_subagent(monkeypatch) -> None:
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    captured = make_graph_capture(monkeypatch, {"configurable": {"openai_api_key": "k",
                                                                  "mcp_servers": []}})
    assert [s["name"] for s in captured["subagents"]] == ["research-subagent"]


def test_coding_subagent_needs_only_the_sandbox(monkeypatch) -> None:
    # Offloading off -> no extract-subagent, but the coder is still there: it fixes scripts,
    # it does not read offloaded files.
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    captured = make_graph_capture(monkeypatch, _sandbox_config(offload_results=False))
    assert [s["name"] for s in captured["subagents"]] == ["research-subagent", "coding-subagent"]


def test_coding_subagent_runs_on_the_tier_coding_model(monkeypatch) -> None:
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    for tier, package in MODEL_TIERS.items():
        captured = make_graph_capture(monkeypatch, _sandbox_config(model_tier=tier))
        spec = _coding_spec(captured)
        assert spec["model"].model_name == package["coding_model"], tier
        assert list(spec["tools"]) == [], tier
        usage = next(m for m in spec["middleware"] if isinstance(m, SubagentUsageMiddleware))
        assert (usage.role, usage.model) == ("coding-subagent", package["coding_model"]), tier


def test_coding_subagent_keeps_file_tools_but_not_grep_or_todos(monkeypatch) -> None:
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    spec = _coding_spec(make_graph_capture(monkeypatch, _sandbox_config()))
    filters = [m for m in spec["middleware"] if isinstance(m, ExcludeToolsMiddleware)]
    assert len(filters) == 1 and filters[0].excluded == CODING_EXCLUDED_TOOLS
    assert {"grep", "write_todos"} == set(CODING_EXCLUDED_TOOLS)
    assert not {"execute", "write_file", "edit_file", "read_file"} & CODING_EXCLUDED_TOOLS
    # After tool injection; only the model-call backoff (innermost) sits inside it.
    assert isinstance(spec["middleware"][-2], ExcludeToolsMiddleware)
    assert isinstance(spec["middleware"][-1], ModelBackoffMiddleware)
    # Not a findings producer: the gate would bounce its STATUS/OUTPUT/NOTES handoff.
    assert not any(isinstance(m, SubagentFindingsMiddleware) for m in spec["middleware"])


def test_research_subagent_can_delegate_to_the_coder(monkeypatch) -> None:
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    captured = make_graph_capture(monkeypatch, _sandbox_config())
    research = next(s for s in captured["subagents"] if s["name"] == "research-subagent")
    task_mw = next(m for m in research["middleware"] if isinstance(m, SubAgentMiddleware))
    assert "coding-subagent" in task_mw.subagent_names
    nested = next(s for s in task_mw._subagents if s["name"] == "coding-subagent")
    # Own filesystem stack (this path skips deepagents' default stack), filter last.
    fs_idx = next(i for i, m in enumerate(nested["middleware"])
                  if isinstance(m, FilesystemMiddleware))
    filt_idx = next(i for i, m in enumerate(nested["middleware"])
                    if isinstance(m, ExcludeToolsMiddleware))
    assert fs_idx < filt_idx
    # The nested filesystem prompt tells it the file rules the filter leaves standing.
    assert "write_file" in nested["middleware"][fs_idx]._custom_system_prompt


def test_scripts_are_captured_and_the_handoff_is_scrubbed(monkeypatch) -> None:
    # The coder emits its code as an artifact event AND has its handoff scrubbed (free
    # prose the parent reads verbatim); the research sub-agent only emits (its handoff is
    # findings JSON with its own gate).
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    captured = make_graph_capture(monkeypatch, _sandbox_config())
    coding = next(m for m in _coding_spec(captured)["middleware"]
                  if isinstance(m, ScriptArtifactsMiddleware))
    assert (coding.agent, coding.scrub) == ("coding-subagent", True)
    research = next(s for s in captured["subagents"] if s["name"] == "research-subagent")
    capture_only = next(m for m in research["middleware"]
                        if isinstance(m, ScriptArtifactsMiddleware))
    assert (capture_only.agent, capture_only.scrub) == ("research-subagent", False)


def test_real_graph_compiles_with_both_nested_workers(monkeypatch) -> None:
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    graph = asyncio.run(agent_mod.make_graph(_sandbox_config()))
    assert graph is not None


def test_coding_prompt_contract() -> None:
    prompt = coding_prompt()
    assert "<<MCP_TOOLS>>" not in prompt and "<<DOMAIN>>" not in prompt  # no unfilled slots
    assert 'NEVER `python -c "..."`' in prompt
    assert "`write_file` the script to /workspace/<name>.py" in prompt
    assert "NO data tools" in prompt
    assert "Do NOT narrate" in prompt
    for field in ("STATUS:", "OUTPUT:", "NOTES:", "RESULT FILE:"):
        assert field in prompt
    # No SCRIPT: field, and an explicit ban: a path is a dead end for the reader and gets
    # copied into the report. The code goes out as a `script` event instead.
    assert "SCRIPT:" not in prompt
    assert "NEVER name the script anywhere in that handoff" in prompt


def test_planner_and_fleet_are_told_to_delegate_failed_scripts() -> None:
    # The narration loop this fixes: a failed `python -c` retried in ever-new quoting
    # variants with an "I will now…" paragraph between each. Both tool-loop roles get the
    # same two rules: scripts are files, and a failure goes to the coder, silently.
    for prompt in (orchestrator_prompt(""), subagent_prompt("")):
        assert "`task` with `coding-subagent`" in prompt or "`coding-subagent` via `task`" in prompt
        assert "python -c" in prompt
    assert "/workspace/<name>.py" in subagent_prompt("")   # the fleet holds write_file
    assert "You hold no file tools" in orchestrator_prompt("")  # the planner does not
    # The extract worker has no coder to call (and no file tools): heredoc, fix once, no narration.
    extract = extract_prompt()
    assert "heredoc" in extract and "no narration" in extract
    assert "coding-subagent" not in extract
