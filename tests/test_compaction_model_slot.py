"""The compaction summarizer runs on the tier's compaction_model, not the utility model."""

from __future__ import annotations

from deep_research_agent.compaction import ContextCompactionMiddleware
from deep_research_agent.config import MODEL_TIERS
from conftest import make_graph_capture


def test_compaction_uses_the_compaction_slot(monkeypatch) -> None:
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    for tier, package in MODEL_TIERS.items():
        captured = make_graph_capture(monkeypatch, {"configurable": {
            "openai_api_key": "k", "mcp_servers": [], "model_tier": tier}})
        compactors = [m for m in captured["middleware"]
                      if isinstance(m, ContextCompactionMiddleware)]
        assert len(compactors) == 1, tier
        assert compactors[0].model.model_name == package["compaction_model"], tier
