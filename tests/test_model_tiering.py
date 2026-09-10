"""Model tiering — named price packages are the ONLY way models are chosen.

Pins the tier-only contract: callers and the environment select a package by NAME
(``configurable.model_tier`` / ``DRA_MODEL_TIER``); every model that can run is
named in code (``MODEL_TIERS``). Legacy per-model keys (``research_model``,
``subagent_model``, ``compression_model``, …) are ignored with a warning.
``DEFAULT_MODEL_TIER`` (the cheapest package) applies when no tier is chosen, so a
bare config can never silently pick an expensive model. The findings gate that
guards the cheap tiers is covered separately in ``test_findings_gate.py``.

Runs with plain Python (``python tests/test_model_tiering.py``) — no pytest needed —
and is also pytest-discoverable. No network, no API keys.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from deep_research_agent import config
from deep_research_agent.config import (DEFAULT_MODEL_TIER, MODEL_CACHING, MODEL_REASONING,
                                         MODEL_TIERS, ResearchConfig)

_ENV_KEYS = ("DRA_MODEL_TIER",)


def _cfg(env: dict[str, str] | None = None, **configurable) -> ResearchConfig:
    """Build a config with the tier env var masked (so the ambient value can't leak
    into the assertions), optionally setting a controlled one via `env`."""
    saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
    os.environ.update(env or {})
    try:
        return ResearchConfig.from_runnable_config({"configurable": configurable})
    finally:
        for k in env or {}:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_model_tier_package_selects_all_three() -> None:
    for name, package in MODEL_TIERS.items():
        cfg = _cfg(model_tier=name)
        assert cfg.research_model == package["research_model"], name
        assert cfg.subagent_model == package["subagent_model"], name
        assert cfg.utility_model == package["utility_model"], name
        assert cfg.coding_model == package["coding_model"], name
        assert cfg.compaction_model == package["compaction_model"], name
    # The rule the "high" package encodes: its top model ORCHESTRATES ONLY — never
    # handed to the sub-agent fleet or the utility slot, which would defeat the
    # tiering. Stated against the slot, not a model name, so swapping the orchestrator
    # (opus-4.8 -> kimi-k3 -> claude-sonnet-5, …) can't quietly turn this vacuous.
    high = MODEL_TIERS["high"]
    assert high["research_model"] not in (high["subagent_model"], high["utility_model"])


def test_bare_config_defaults_to_cheapest_tier() -> None:
    # Nothing configured at all -> the DEFAULT_MODEL_TIER package, never an
    # expensive surprise. Production callers opt UP explicitly.
    assert DEFAULT_MODEL_TIER == "extra-low"
    cfg = _cfg()
    package = MODEL_TIERS[DEFAULT_MODEL_TIER]
    assert cfg.research_model == package["research_model"]
    assert cfg.subagent_model == package["subagent_model"]
    assert cfg.utility_model == package["utility_model"]
    assert cfg.coding_model == package["coding_model"]
    assert cfg.compaction_model == package["compaction_model"]


def test_configurable_tier_beats_env_and_unknown_falls_back() -> None:
    cfg = _cfg(env={"DRA_MODEL_TIER": "low"}, model_tier="high")
    assert cfg.research_model == MODEL_TIERS["high"]["research_model"]  # cfg beats env
    cfg = _cfg(env={"DRA_MODEL_TIER": "low"})
    assert cfg.research_model == MODEL_TIERS["low"]["research_model"]   # env honored
    cfg = _cfg(model_tier="no-such-tier")    # falls back to the default tier, warns
    assert cfg.research_model == MODEL_TIERS[DEFAULT_MODEL_TIER]["research_model"]


def test_per_model_keys_are_ignored() -> None:
    # Tier-only contract: a caller cannot smuggle in a specific model. Legacy keys
    # (incl. the old sanbase aliases) are ignored — the tier package wins.
    package = MODEL_TIERS[DEFAULT_MODEL_TIER]
    cfg = _cfg(research_model="anthropic/claude-opus-4.8",
               subagent_model="my/custom-model",
               utility_model="my/other-model",
               coding_model="my/coder-model",
               compaction_model="my/summary-model",
               final_report_model="my/report-model",
               compression_model="my/compression-model")
    assert cfg.research_model == package["research_model"]
    assert cfg.subagent_model == package["subagent_model"]
    assert cfg.utility_model == package["utility_model"]
    assert cfg.coding_model == package["coding_model"]
    assert cfg.compaction_model == package["compaction_model"]
    assert cfg.report_model == package["research_model"]  # report = research, reserved


def test_report_model_follows_tier_research_slot() -> None:
    cfg = _cfg(model_tier="high")
    assert cfg.report_model == MODEL_TIERS["high"]["research_model"]


def test_budget_fallbacks_match_dataclass_defaults() -> None:
    # The from_runnable_config fallbacks must be the documented dataclass defaults —
    # they diverged once (80 vs 200, 2M vs 4M) and the README lied about behavior.
    saved = {k: os.environ.pop(k, None)
             for k in ("DRA_MAX_TOOL_CALLS", "DRA_MAX_TOTAL_TOKENS")}
    try:
        cfg = _cfg()
        assert cfg.max_tool_calls == ResearchConfig.max_tool_calls
        assert cfg.max_total_tokens == ResearchConfig.max_total_tokens
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _tier_prices() -> dict[str, dict[str, tuple[float, float]]]:
    """``{tier: {slot: ($in, $out)}}`` parsed from the ``# $x / $y`` comments beside each
    slug in ``MODEL_TIERS``. Those comments are the only price data in the repo, so this
    parse is what makes them load-bearing instead of decorative — an edit that changes a
    model without its price, or breaks the cost ordering below, fails here."""
    src = Path(config.__file__).read_text(encoding="utf-8")
    body = src.split("MODEL_TIERS: dict[str, dict[str, str]] = {", 1)[1].split("\n}\n", 1)[0]
    out: dict[str, dict[str, tuple[float, float]]] = {}
    tier = ""
    for line in body.splitlines():
        opens = re.match(r'\s*"([\w-]+)":\s*\{', line)
        if opens:
            tier = opens.group(1)
            out[tier] = {}
            continue
        slot = re.match(r'\s*"(\w+_model)":\s*"[^"]+",\s*#\s*\$([\d.]+)\s*/\s*\$([\d.]+)', line)
        if slot and tier:
            out[tier][slot.group(1)] = (float(slot.group(2)), float(slot.group(3)))
    return out


def test_every_tier_slot_documents_its_price() -> None:
    prices = _tier_prices()
    assert set(prices) == set(MODEL_TIERS), (set(prices), set(MODEL_TIERS))
    for tier, package in MODEL_TIERS.items():
        assert set(prices[tier]) == set(package), tier


def test_coding_slot_is_a_dedicated_coder() -> None:
    # The coding slot exists to put a model GOOD AT CODE behind script writing/fixing, on a
    # small input — so it is exempt from the fleet-price invariant below but must never be
    # a silent alias of the fleet model (which would make the slot decorative).
    for tier, package in MODEL_TIERS.items():
        coder = package["coding_model"]
        assert coder != package["subagent_model"], f"{tier}: coding slot just mirrors the fleet"
        assert coder != package["utility_model"], f"{tier}: coding slot just mirrors utility"


def test_subagent_is_never_pricier_than_the_orchestrator() -> None:
    # The tiering invariant. The sub-agent fleet makes MOST of a run's tool calls, each
    # worker burning raw tool output in its own context, so a fleet at planner prices
    # makes the tier's cost the FLEET's cost and the tiering stops meaning anything.
    # This has been violated in practice (a `high` tier once ran an opus planner over a
    # sonnet fleet priced identically to it), which is why it is asserted, not just noted.
    for tier, slots in _tier_prices().items():
        research, subagent = slots["research_model"], slots["subagent_model"]
        assert subagent[0] <= research[0], f"{tier}: sub-agent input ${subagent[0]} > ${research[0]}"
        assert subagent[1] <= research[1], f"{tier}: sub-agent output ${subagent[1]} > ${research[1]}"


def test_tiers_cost_more_as_they_go_up() -> None:
    # `extra-low` < `low` < `mid` < `high` on the research slot, both axes. A "higher"
    # tier that is cheaper than the one below it means the names lie to the caller.
    prices = _tier_prices()
    ladder = ["extra-low", "low", "mid", "high"]
    assert set(ladder) == set(MODEL_TIERS), "ladder must cover every tier"
    for lower, higher in zip(ladder, ladder[1:]):
        lo, hi = prices[lower]["research_model"], prices[higher]["research_model"]
        assert lo[0] <= hi[0] and lo[1] <= hi[1], f"{higher} is not pricier than {lower}"


def test_every_tier_model_carries_a_reasoning_flag() -> None:
    # A missing flag is not a crash — models.py just omits the parameter — so this is the
    # only thing that notices a model quietly stopped reasoning.
    cfg = _cfg()
    for tier, package in MODEL_TIERS.items():
        for slot, slug in package.items():
            assert slug in MODEL_REASONING, (
                f"{tier}.{slot} = {slug} has no flag in config.MODEL_REASONING, so it will "
                "run WITHOUT the reasoning parameter. Check the slug against OpenRouter's "
                'model index (`supported_parameters` must contain "reasoning") and add '
                "True, or add False to record that the model rejects it."
            )
            assert cfg.supports_reasoning(slug) is MODEL_REASONING[slug], (tier, slot, slug)


def test_reasoning_is_withheld_from_models_flagged_false() -> None:
    # `-instruct-2507` rejects the parameter, its `-thinking-2507` sibling accepts it: flags
    # are per slug, never per family.
    cfg = _cfg()
    assert MODEL_REASONING["qwen/qwen3-30b-a3b-instruct-2507"] is False
    assert not cfg.supports_reasoning("qwen/qwen3-30b-a3b-instruct-2507")
    assert cfg.supports_reasoning("qwen/qwen3-30b-a3b-thinking-2507")
    # An unflagged model is treated as rejecting: forfeit the parameter, never risk a 400.
    assert "some/unreleased-model-v9" not in MODEL_REASONING
    assert not cfg.supports_reasoning("some/unreleased-model-v9")
    # The `openai:` prefix form resolves to the same bare slug (see _strip_provider).
    assert cfg.supports_reasoning("openai:qwen/qwen3.8-27b")


def test_reasoning_param_reaches_only_capable_models() -> None:
    # Through the real model builder: present for a listed slug, absent for an unlisted one.
    from deep_research_agent.models import build_chat_model

    cfg = _cfg()
    capable = build_chat_model(MODEL_TIERS["mid"]["research_model"], cfg)
    assert (capable.extra_body or {}).get("reasoning") == {"effort": cfg.reasoning_effort}
    rejecting = build_chat_model("qwen/qwen3-30b-a3b-instruct-2507", cfg)
    assert "reasoning" not in (rejecting.extra_body or {})



def test_no_tier_may_name_a_model_that_cannot_cache() -> None:
    # Every ReAct step re-sends the growing prefix; a model without a cache pays full price
    # to re-read its own context.
    cfg = _cfg()
    for tier, package in MODEL_TIERS.items():
        for slot, slug in package.items():
            assert slug in MODEL_CACHING, (
                f"{tier}.{slot} = {slug} has no flag in config.MODEL_CACHING. Check "
                "OpenRouter's model index for `pricing.input_cache_read` and add True, or "
                "add False and pick a different model — a tier may not name one."
            )
            assert cfg.caches_prompts(slug), (
                f"{tier}.{slot} = {slug} prices no cache read: an input-heavy role on it "
                "pays full price for every re-sent prefix. Pick a caching model."
            )


def test_known_non_caching_models_are_recorded_and_rejected() -> None:
    # Recorded as False, not omitted, so a tier edit cannot pick one up believing it caches.
    cfg = _cfg()
    for slug in ("qwen/qwen3-30b-a3b-instruct-2507", "qwen/qwen3-30b-a3b-thinking-2507",
                 "qwen/qwen3-30b-a3b"):
        assert MODEL_CACHING[slug] is False
        assert not cfg.caches_prompts(slug)
    # Unlisted is treated as non-caching: unknown is not a promise.
    assert not cfg.caches_prompts("some/unreleased-model-v9")
    assert cfg.caches_prompts("deepseek/deepseek-v4-flash-0731")

if __name__ == "__main__":
    test_model_tier_package_selects_all_three()
    test_bare_config_defaults_to_cheapest_tier()
    test_configurable_tier_beats_env_and_unknown_falls_back()
    test_per_model_keys_are_ignored()
    test_report_model_follows_tier_research_slot()
    test_budget_fallbacks_match_dataclass_defaults()
    test_every_tier_slot_documents_its_price()
    test_subagent_is_never_pricier_than_the_orchestrator()
    test_tiers_cost_more_as_they_go_up()
    test_every_tier_model_carries_a_reasoning_flag()
    test_reasoning_is_withheld_from_models_flagged_false()
    test_reasoning_param_reaches_only_capable_models()
    test_no_tier_may_name_a_model_that_cannot_cache()
    test_known_non_caching_models_are_recorded_and_rejected()
    print("OK — tier-only model selection verified.")
