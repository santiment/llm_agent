"""deepagents' auto-added `general-purpose` sub-agent is opted out for the whole fleet."""

from __future__ import annotations

import deep_research_agent.agent  # noqa: F401  (registers the harness profile on import)
from deep_research_agent.config import MODEL_TIERS, ResearchConfig
from deep_research_agent.models import build_chat_model
from deepagents.profiles.harness.harness_profiles import _harness_profile_for_model


def test_every_tier_model_resolves_to_a_profile_without_general_purpose():
    for tier, package in MODEL_TIERS.items():
        cfg = ResearchConfig.from_runnable_config({"configurable": {"openai_api_key": "k", "model_tier": tier}})
        for slot in ("research_model", "subagent_model"):
            model = build_chat_model(getattr(cfg, slot), cfg)
            profile = _harness_profile_for_model(model, None)
            gp = profile.general_purpose_subagent
            assert gp is not None and gp.enabled is False, (tier, slot)
