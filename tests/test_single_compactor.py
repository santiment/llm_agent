"""Compaction is ours alone: deepagents' built-in SummarizationMiddleware must be absent
from every assembled stack.

Our OpenRouter models carry no profile, so deepagents' summarizer defaults to a fixed
170k-token trigger and runs on each role's own model. It never fired while our compaction
triggered at 100k; with ContextCompactionMiddleware's sub-agent trigger at 600k it
would take over — on the expensive model, without the provider-routing fallback, writing
conversation-history files into the sandbox. Builds the real graph (create_agent spied),
offline.
"""

from __future__ import annotations

import asyncio

import deepagents.graph as dg

from deep_research_agent import agent as agent_mod
from deep_research_agent.compaction import ContextCompactionMiddleware


def _stacks(monkeypatch, **configurable) -> list[list]:
    stacks: list[list] = []
    real = dg.create_agent

    def spy(*args, **kwargs):
        stacks.append(list(kwargs.get("middleware") or []))
        return real(*args, **kwargs)

    monkeypatch.setattr(dg, "create_agent", spy)
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    asyncio.run(agent_mod.make_graph({"configurable": {"openai_api_key": "k", "mcp_servers": [],
                                                       **configurable}}))
    return stacks


def _names(stack: list) -> set[str]:
    return {type(m).__name__ for m in stack}


def test_no_deepagents_summarizer_and_our_compaction_on_every_role(monkeypatch):
    for extra in ({}, {"sandbox_url": "http://sandbox.invalid:8080"}):
        stacks = _stacks(monkeypatch, **extra)
        assert stacks, "create_agent was never called"
        for stack in stacks:
            assert "_DeepAgentsSummarizationMiddleware" not in _names(stack)
            assert any(isinstance(m, ContextCompactionMiddleware) for m in stack)
