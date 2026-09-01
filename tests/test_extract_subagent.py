"""extract-subagent wiring: registered only when offloading is live, runs on
cfg.utility_model with no data tools, and returns the gated findings format."""

from __future__ import annotations

import asyncio

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware

import deep_research_agent.agent as agent_mod
from deep_research_agent.findings_gate import SubagentFindingsMiddleware
from deep_research_agent.prompts import extract_prompt


class _StubGraph:
    def with_config(self, *args, **kwargs):
        return self


def _make_graph(monkeypatch, config: dict) -> dict:
    """Run make_graph with create_deep_agent stubbed out; return its captured kwargs."""
    captured: dict = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return _StubGraph()

    monkeypatch.setattr(agent_mod, "create_deep_agent", fake_create_deep_agent)
    asyncio.run(agent_mod.make_graph(config))
    return captured


def _base_config() -> dict:
    return {"configurable": {"openai_api_key": "test-key", "mcp_servers": []}}


def test_no_sandbox_no_extract_subagent(monkeypatch) -> None:
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    captured = _make_graph(monkeypatch, _base_config())
    names = [s["name"] for s in captured["subagents"]]
    assert names == ["research-subagent"]


def test_extract_subagent_wired_when_offloading(monkeypatch) -> None:
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    config = _base_config()
    config["configurable"]["sandbox_url"] = "http://sandbox.invalid:8080"
    captured = _make_graph(monkeypatch, config)

    specs = {s["name"]: s for s in captured["subagents"]}
    assert set(specs) == {"research-subagent", "extract-subagent"}

    extract = specs["extract-subagent"]
    assert extract["model"].model_name != specs["research-subagent"]["model"].model_name
    assert list(extract["tools"]) == []
    assert any(isinstance(m, SubagentFindingsMiddleware) for m in extract["middleware"])
    assert "RETURN FORMAT" in extract["system_prompt"]


def test_extract_subagent_uses_utility_model(monkeypatch) -> None:
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    config = _base_config()
    config["configurable"]["sandbox_url"] = "http://sandbox.invalid:8080"
    config["configurable"]["model_tier"] = "extra-low"
    captured = _make_graph(monkeypatch, config)

    extract = next(s for s in captured["subagents"] if s["name"] == "extract-subagent")
    assert extract["model"].model_name == "qwen/qwen3-30b-a3b-instruct-2507"


def test_extract_subagent_disabled_when_offload_off(monkeypatch) -> None:
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    config = _base_config()
    config["configurable"]["sandbox_url"] = "http://sandbox.invalid:8080"
    config["configurable"]["offload_results"] = False
    captured = _make_graph(monkeypatch, config)
    names = [s["name"] for s in captured["subagents"]]
    assert names == ["research-subagent"]


def test_research_subagent_delegates_to_extract(monkeypatch) -> None:
    """Research-subagent gets a nested task tool restricted to extract-subagent."""
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    config = _base_config()
    config["configurable"]["sandbox_url"] = "http://sandbox.invalid:8080"
    captured = _make_graph(monkeypatch, config)

    research = next(s for s in captured["subagents"] if s["name"] == "research-subagent")
    task_mws = [m for m in research["middleware"] if isinstance(m, SubAgentMiddleware)]
    assert len(task_mws) == 1
    assert task_mws[0].subagent_names == frozenset({"extract-subagent"})
    # Nested spec must carry its own filesystem stack (no default stack on this path).
    nested = task_mws[0]._subagents[0]
    assert any(isinstance(m, FilesystemMiddleware) for m in nested["middleware"])
    assert any(isinstance(m, SubagentFindingsMiddleware) for m in nested["middleware"])


def test_real_graph_compiles_with_nested_extract(monkeypatch) -> None:
    """Real graph build — the nested spec compiles eagerly, so wiring errors surface here."""
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    config = _base_config()
    config["configurable"]["sandbox_url"] = "http://sandbox.invalid:8080"
    graph = asyncio.run(agent_mod.make_graph(config))
    assert graph is not None


def test_no_nested_task_without_offloading(monkeypatch) -> None:
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    captured = _make_graph(monkeypatch, _base_config())
    research = next(s for s in captured["subagents"] if s["name"] == "research-subagent")
    assert not any(isinstance(m, SubAgentMiddleware) for m in research["middleware"])


def test_orchestrator_holds_no_data_tools(monkeypatch) -> None:
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    captured = _make_graph(monkeypatch, _base_config())

    orchestrator_tools = {t.name for t in captured["tools"]}
    assert orchestrator_tools == {"request_clarification", "submit_report"}
    subagent = next(s for s in captured["subagents"] if s["name"] == "research-subagent")
    assert "web_search" in {t.name for t in subagent["tools"]}


def test_extract_prompt_contract() -> None:
    prompt = extract_prompt()
    assert "RETURN FORMAT" in prompt
    assert "<<" not in prompt
    assert "NO data tools" in prompt


def test_extract_prompt_renders_domain_block() -> None:
    prompt = extract_prompt("Focus on crypto assets.")
    assert "DOMAIN CONTEXT" in prompt
    assert "Focus on crypto assets." in prompt
    assert "DOMAIN CONTEXT" not in extract_prompt("")
