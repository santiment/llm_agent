"""The orchestrator's harness is trimmed: a triage router runs first, it holds no file tools
and no skills (those live with the research sub-agent), and its prompt lists the skills by
name and description instead."""

from __future__ import annotations

from pathlib import Path

from deep_research_agent.agent import describe_skills
from deep_research_agent.skill_usage import SkillUsageMiddleware
from deep_research_agent.tool_filter import ORCHESTRATOR_EXCLUDED_TOOLS, ExcludeToolsMiddleware
from deep_research_agent.triage import TriageRouterMiddleware
from conftest import make_graph_capture

REPO = Path(__file__).resolve().parents[1]


def _captured(monkeypatch, **extra):
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    return make_graph_capture(monkeypatch, {"configurable": {"openai_api_key": "k",
                                                              "mcp_servers": [], **extra}})


def test_triage_router_runs_first(monkeypatch):
    captured = _captured(monkeypatch)
    assert isinstance(captured["middleware"][0], TriageRouterMiddleware)


def test_orchestrator_hides_file_tools_but_keeps_execute(monkeypatch):
    captured = _captured(monkeypatch)
    filters = [m for m in captured["middleware"] if isinstance(m, ExcludeToolsMiddleware)]
    assert len(filters) == 1 and filters[0].excluded == ORCHESTRATOR_EXCLUDED_TOOLS
    assert {"ls", "read_file", "write_file", "edit_file", "glob", "grep"} == set(ORCHESTRATOR_EXCLUDED_TOOLS)
    assert "execute" not in ORCHESTRATOR_EXCLUDED_TOOLS and "task" not in ORCHESTRATOR_EXCLUDED_TOOLS
    assert isinstance(captured["middleware"][-1], ExcludeToolsMiddleware)


def test_skills_live_with_the_research_subagent_only(monkeypatch):
    captured = _captured(monkeypatch, skills_dir=str(REPO / "skills"))
    assert captured["skills"] is None
    research = next(s for s in captured["subagents"] if s["name"] == "research-subagent")
    assert research["skills"] == ["/skills/"]
    assert any(isinstance(m, SkillUsageMiddleware) for m in research["middleware"])
    assert not any(isinstance(m, SkillUsageMiddleware) for m in captured["middleware"])
    # The planner sees the skill by name + description, not the file.
    prompt = captured["system_prompt"]
    assert "SKILLS (playbooks held by the sub-agents" in prompt
    assert "`crowd-positioning`" in prompt and "POSITIONED" in prompt
    assert "You hold no file tools" in prompt


def test_describe_skills_reads_front_matter(tmp_path, caplog):
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "SKILL.md").write_text("---\nname: alpha\ndescription: >-\n  Does\n  things.\n---\n# body\n")
    (tmp_path / "beta").mkdir()
    (tmp_path / "beta" / "SKILL.md").write_text("---\nname: [unclosed\n---\n")
    (tmp_path / "gamma").mkdir()                     # no SKILL.md → not a skill
    with caplog.at_level("WARNING"):
        out = describe_skills(str(tmp_path))
    assert out == [("alpha", "Does things."), ("beta", "")]
    assert "beta" in caplog.text
    assert describe_skills("") == [] and describe_skills(str(tmp_path / "missing")) == []
