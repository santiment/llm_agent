"""Every OpenRouter model call carries a provider-routing object: a soft throughput
preference, a hard price cap at a factor of the cheapest healthy endpoint, and an ignore
list of unstable providers — the latter two computed per model from OpenRouter's live
endpoint feed at graph build (``provider_routing.py``).

OpenRouter's default routing is price-weighted, and within one model the cheapest
providers are the slowest (on 2026-09-11 the fleet model's $0.05–0.07 endpoints ran
16–55 tok/s at p50 while its $0.10–0.13 ones ran 115–117; the cheapest V4.1 Flash
reseller ran 12 tok/s against 127 first-party). Every ReAct step pays that. Pins:

  - the three knobs parse like the rest (defaults, env, configurable-beats-env, clamps,
    unknown sort falls back to load balancing);
  - build_chat_model sends OpenRouter's `provider` object with the documented field
    names and the p50 percentile form, only when something is set, and never off
    OpenRouter;
  - routing(): the cap is factor x the cheapest HEALTHY endpoint per axis in $/M; a
    provider goes on `ignore` only when none of its endpoints is healthy; a missing uptime
    figure counts as healthy; no healthy endpoint or no feed -> soft preferences only;
  - resolve() / make_graph: the per-model object reaches the model that slot builds.

Runs with plain Python (``python tests/test_provider_routing.py``) — no pytest needed.
"""

from __future__ import annotations

import os

import asyncio

import deep_research_agent.agent as agent_mod
import deep_research_agent.provider_routing as pr
from deep_research_agent.config import ResearchConfig
from deep_research_agent.models import build_chat_model, provider_preferences

_ENV_KEYS = ("DRA_PROVIDER_MIN_THROUGHPUT", "DRA_PROVIDER_MAX_LATENCY", "DRA_PROVIDER_SORT",
             "DRA_PROVIDER_MAX_PRICE_FACTOR", "DRA_PROVIDER_MIN_UPTIME",
             "DRA_PROVIDER_ROUTING_TTL", "OPENAI_BASE_URL")


def _cfg(env: dict[str, str] | None = None, **configurable) -> ResearchConfig:
    saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
    os.environ.update(env or {})
    try:
        return ResearchConfig.from_runnable_config({"configurable": configurable})
    finally:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update({k: v for k, v in saved.items() if v is not None})


def test_defaults_prefer_throughput_softly() -> None:
    cfg = _cfg()
    assert cfg.provider_min_throughput == 50.0 == ResearchConfig.provider_min_throughput
    assert cfg.provider_max_latency == 0.0
    assert cfg.provider_sort == ""
    assert (cfg.provider_max_price_factor, cfg.provider_min_uptime, cfg.provider_routing_ttl) == (1.25, 97.0, 300.0)
    assert provider_preferences(cfg) == {"preferred_min_throughput": {"p50": 50.0}}


def test_live_rule_knobs_parse_and_clamp() -> None:
    cfg = _cfg({"DRA_PROVIDER_MAX_PRICE_FACTOR": "1.5", "DRA_PROVIDER_MIN_UPTIME": "95",
                "DRA_PROVIDER_ROUTING_TTL": "60"})
    assert (cfg.provider_max_price_factor, cfg.provider_min_uptime, cfg.provider_routing_ttl) == (1.5, 95.0, 60.0)
    cfg = _cfg({"DRA_PROVIDER_MAX_PRICE_FACTOR": "1.5"}, provider_max_price_factor=2)
    assert cfg.provider_max_price_factor == 2.0  # configurable beats env
    assert _cfg(provider_max_price_factor=0).provider_max_price_factor == 0.0  # off
    assert _cfg(provider_max_price_factor=0.5).provider_max_price_factor == 1.0  # never below the cheapest
    assert _cfg(provider_min_uptime=150).provider_min_uptime == 100.0
    assert _cfg(provider_min_uptime=-5).provider_min_uptime == 0.0
    assert _cfg(provider_routing_ttl=-1).provider_routing_ttl == 0.0


def test_env_and_configurable() -> None:
    cfg = _cfg({"DRA_PROVIDER_MIN_THROUGHPUT": "90", "DRA_PROVIDER_MAX_LATENCY": "2.5",
                "DRA_PROVIDER_SORT": "Throughput"})
    assert (cfg.provider_min_throughput, cfg.provider_max_latency, cfg.provider_sort) == (90.0, 2.5, "throughput")
    cfg = _cfg({"DRA_PROVIDER_MIN_THROUGHPUT": "90"}, provider_min_throughput=120,
               provider_max_latency=1, provider_sort="latency")
    assert (cfg.provider_min_throughput, cfg.provider_max_latency, cfg.provider_sort) == (120.0, 1.0, "latency")
    assert provider_preferences(cfg) == {
        "preferred_min_throughput": {"p50": 120.0},
        "preferred_max_latency": {"p50": 1.0},
        "sort": "latency",
    }


def test_zero_disables_and_negatives_clamp() -> None:
    cfg = _cfg(provider_min_throughput=0, provider_max_latency=-3)
    assert cfg.provider_min_throughput == 0.0 and cfg.provider_max_latency == 0.0
    assert provider_preferences(cfg) == {}
    assert _cfg({"DRA_PROVIDER_MIN_THROUGHPUT": "0"}).provider_min_throughput == 0.0


def test_unknown_sort_keeps_load_balancing() -> None:
    assert _cfg(provider_sort="fastest").provider_sort == ""
    assert _cfg({"DRA_PROVIDER_SORT": "cheap"}).provider_sort == ""


def test_preference_reaches_the_chat_model_on_openrouter() -> None:
    cfg = _cfg(openai_api_key="k")
    assert cfg.is_openrouter
    model = build_chat_model("deepseek/deepseek-v4-flash-0731", cfg)
    assert model.extra_body["provider"] == {"preferred_min_throughput": {"p50": 50.0}}
    # The other OpenRouter extras are untouched by it.
    assert model.extra_body["max_tokens"] == cfg.max_output_tokens
    assert model.extra_body["reasoning"] == {"effort": cfg.reasoning_effort}


def test_nothing_set_sends_no_provider_object() -> None:
    cfg = _cfg(openai_api_key="k", provider_min_throughput=0)
    model = build_chat_model("deepseek/deepseek-v4-flash-0731", cfg)
    assert "provider" not in (model.extra_body or {})


def test_explicit_provider_object_wins() -> None:
    cfg = _cfg(openai_api_key="k")
    routed = {"preferred_min_throughput": {"p50": 50.0}, "max_price": {"prompt": 0.0625, "completion": 0.2}}
    assert build_chat_model("m", cfg, routed).extra_body["provider"] == routed
    assert "provider" not in (build_chat_model("m", cfg, {}).extra_body or {})


# --- the live rules, on a fake feed -------------------------------------------------------

def _ep(provider, tag, prompt, completion, uptime=99.5, status=0):
    return {"provider_name": provider, "tag": tag, "status": status, "uptime_last_30m": uptime,
            "pricing": {"prompt": str(prompt / 1e6), "completion": str(completion / 1e6)}}


_FLEET = [
    _ep("OpenInference", "openinference/fp8", 0.05, 0.16),
    _ep("DeepInfra", "deepinfra/fp8", 0.06, 0.18),
    _ep("Wafer", "wafer", 0.10, 0.25),
    _ep("BaseTen", "baseten/fp8", 0.13, 0.26, uptime=69.4, status=-5),
    _ep("BaseTen", "baseten/fp4", 0.13, 0.26, uptime=99.9),      # a healthy sibling: provider stays
    _ep("Fireworks", "fireworks", 0.22, 0.66, uptime=94.7, status=-2),
    _ep("Novita", "novita/fp8", 0.41, 1.23, uptime=None),         # no figure yet: counts as ok
]


def test_cap_is_factor_times_cheapest_healthy_and_unstable_providers_are_ignored() -> None:
    out = pr.routing(_cfg(), _FLEET, "fleet")
    assert out["preferred_min_throughput"] == {"p50": 50.0}
    assert out["max_price"] == {"prompt": 0.0625, "completion": 0.2}  # 1.25 x $0.05 / $0.16
    assert out["ignore"] == ["fireworks"]


def test_cap_anchors_on_healthy_endpoints_only() -> None:
    feed = [_ep("Cheap", "cheap", 0.04, 0.10, uptime=80.0, status=-2)] + _FLEET[:3]
    out = pr.routing(_cfg(), feed)
    assert out["max_price"] == {"prompt": 0.0625, "completion": 0.2}  # the $0.04 one is down
    assert out["ignore"] == ["cheap"]


def test_rules_switch_off_independently() -> None:
    no_cap = pr.routing(_cfg(provider_max_price_factor=0), _FLEET)
    assert "max_price" not in no_cap and no_cap["ignore"] == ["fireworks"]
    no_uptime = pr.routing(_cfg(provider_min_uptime=0), _FLEET)
    assert "ignore" not in no_uptime
    assert no_uptime["max_price"] == {"prompt": 0.0625, "completion": 0.2}


def test_no_feed_or_no_healthy_endpoint_means_soft_preferences_only() -> None:
    soft = {"preferred_min_throughput": {"p50": 50.0}}
    assert pr.routing(_cfg(), None) == soft
    assert pr.routing(_cfg(), []) == soft
    down = [_ep("A", "a", 0.05, 0.16, uptime=50.0, status=-5), _ep("B", "b", 0.06, 0.18, uptime=90.0)]
    assert pr.routing(_cfg(), down, "x") == soft


def test_resolve_keys_every_slot_and_skips_the_feed_off_openrouter(monkeypatch) -> None:
    feeds = {"a/fleet": _FLEET, "b/planner": None}
    calls = []

    async def fake_fetch(slug, ttl, timeout=5.0):
        calls.append(slug)
        return feeds[slug]

    monkeypatch.setattr(pr, "fetch_endpoints", fake_fetch)
    cfg = _cfg(openai_api_key="k")
    out = asyncio.run(pr.resolve(cfg, ["a/fleet", "b/planner", "a/fleet"]))
    assert sorted(calls) == ["a/fleet", "b/planner"]  # deduplicated
    assert out["a/fleet"]["max_price"] == {"prompt": 0.0625, "completion": 0.2}
    assert out["b/planner"] == {"preferred_min_throughput": {"p50": 50.0}}
    calls.clear()
    local = _cfg({"OPENAI_BASE_URL": "http://localhost:11434/v1"}, openai_api_key="k")
    assert asyncio.run(pr.resolve(local, ["a/fleet"])) == {"a/fleet": {}} and calls == []


def test_make_graph_routes_every_slot_model(monkeypatch) -> None:
    from conftest import make_graph_capture

    routed: dict = {}

    async def fake_resolve(cfg, slugs):
        routed.update({s: {"max_price": {"prompt": 1.0, "completion": 2.0}, "tag": s} for s in slugs})
        return routed

    monkeypatch.setattr(agent_mod, "resolve_routing", fake_resolve)
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    config = {"configurable": {"openai_api_key": "k", "mcp_servers": []}}
    captured = make_graph_capture(monkeypatch, config)
    cfg = ResearchConfig.from_runnable_config(config)
    assert captured["model"].extra_body["provider"] == routed[cfg.research_model]
    fleet = next(s for s in captured["subagents"] if s["name"] == "research-subagent")["model"]
    assert fleet.extra_body["provider"] == routed[cfg.subagent_model]


def test_not_sent_off_openrouter() -> None:
    cfg = _cfg({"OPENAI_BASE_URL": "http://localhost:11434/v1"}, openai_api_key="k")
    assert not cfg.is_openrouter
    model = build_chat_model("some/local-model", cfg)
    assert "provider" not in (model.extra_body or {})


if __name__ == "__main__":
    test_defaults_prefer_throughput_softly()
    test_live_rule_knobs_parse_and_clamp()
    test_explicit_provider_object_wins()
    test_cap_is_factor_times_cheapest_healthy_and_unstable_providers_are_ignored()
    test_cap_anchors_on_healthy_endpoints_only()
    test_rules_switch_off_independently()
    test_no_feed_or_no_healthy_endpoint_means_soft_preferences_only()
    test_env_and_configurable()
    test_zero_disables_and_negatives_clamp()
    test_unknown_sort_keeps_load_balancing()
    test_preference_reaches_the_chat_model_on_openrouter()
    test_nothing_set_sends_no_provider_object()
    test_not_sent_off_openrouter()
    print("OK — provider routing preference verified.")
