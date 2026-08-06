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
from deep_research_agent.config import DEFAULT_MODEL_TIER, MODEL_TIERS, ResearchConfig

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
               final_report_model="my/report-model",
               compression_model="my/compression-model")
    assert cfg.research_model == package["research_model"]
    assert cfg.subagent_model == package["subagent_model"]
    assert cfg.utility_model == package["utility_model"]
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
    print("OK — tier-only model selection verified.")
