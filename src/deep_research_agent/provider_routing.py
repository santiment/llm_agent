"""Per-model OpenRouter provider routing from the live endpoint feed.

Rules, all config knobs: among a model's HEALTHY endpoints (status ok, ``uptime_last_30m``
>= ``provider_min_uptime``) the cheapest sets the price baseline; anything above
``provider_max_price_factor`` x that is refused with ``max_price`` (hard — OpenRouter
errors rather than falls back, which is why the baseline is re-read every
``provider_routing_ttl`` seconds), providers with no healthy endpoint go on ``ignore``, and
the soft ``preferred_*`` thresholds order what is left. The feed
(``GET /api/v1/models/{slug}/endpoints``, public) unreachable or empty -> soft preferences
only, logged; nothing here can fail a graph build.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Iterable

from .config import ResearchConfig
from .models import provider_preferences

log = logging.getLogger("deep_research_agent.provider_routing")

ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{slug}/endpoints"

# slug -> (expires_at, endpoints). Module-level like the MCP listing cache: make_graph runs
# per research run, and the feed changes on the minute, not per run.
_CACHE: dict[str, tuple[float, list[dict]]] = {}


def clear_cache() -> None:
    _CACHE.clear()


async def fetch_endpoints(slug: str, ttl: float, timeout: float = 5.0) -> list[dict] | None:
    """The model's endpoint list from OpenRouter, cached ``ttl`` seconds; None if unreachable."""
    now = time.monotonic()
    hit = _CACHE.get(slug) if ttl > 0 else None
    if hit and hit[0] > now:
        return hit[1]
    import httpx  # the OpenAI SDK's client; imported here so the module stays cheap to import

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(ENDPOINTS_URL.format(slug=slug),
                                 headers={"User-Agent": "deep-research-agent"})
            r.raise_for_status()
            eps = (r.json().get("data") or {}).get("endpoints") or []
    except Exception as exc:  # never fail a graph build over routing metadata
        log.warning("PROVIDER ROUTING %s: endpoint feed unavailable (%s) — soft preferences only",
                    slug, exc)
        return None
    if ttl > 0:
        _CACHE[slug] = (now + ttl, eps)
    return eps


def _price(ep: dict, key: str) -> float | None:
    """$/M tokens, the unit ``max_price`` takes; the feed quotes $/token as a string."""
    try:
        return float(ep["pricing"][key]) * 1e6
    except (KeyError, TypeError, ValueError):
        return None


def provider_slug(ep: dict) -> str:
    """The slug ``ignore`` / ``order`` take: the endpoint tag's provider part ("deepinfra/fp8"
    -> "deepinfra"), else the display name lowercased."""
    tag = str(ep.get("tag") or "")
    if tag:
        return tag.split("/")[0]
    return str(ep.get("provider_name") or "").strip().lower().replace(" ", "-")


def is_healthy(ep: dict, min_uptime: float) -> bool:
    """Status ok and 30-minute uptime at or above the floor; no uptime figure counts as ok."""
    if ep.get("status", 0) != 0:
        return False
    up = ep.get("uptime_last_30m")
    return up is None or min_uptime <= 0 or float(up) >= min_uptime


def routing(cfg: ResearchConfig, endpoints: list[dict] | None, slug: str = "") -> dict[str, Any]:
    """One model's ``provider`` object: soft preferences always; price cap and ignore list
    when the feed has at least one healthy endpoint to anchor them."""
    out = provider_preferences(cfg)
    if not endpoints:
        return out
    healthy = [e for e in endpoints if is_healthy(e, cfg.provider_min_uptime)]
    if not healthy:
        log.warning("PROVIDER ROUTING %s: no endpoint at >= %.0f%% uptime — no price cap, "
                    "no ignore list", slug, cfg.provider_min_uptime)
        return out
    if cfg.provider_max_price_factor > 0:
        prompts = [p for p in (_price(e, "prompt") for e in healthy) if p is not None]
        completions = [p for p in (_price(e, "completion") for e in healthy) if p is not None]
        if prompts and completions:
            f = cfg.provider_max_price_factor
            out["max_price"] = {"prompt": round(min(prompts) * f, 4),
                                "completion": round(min(completions) * f, 4)}
    if cfg.provider_min_uptime > 0:
        ok = {provider_slug(e) for e in healthy}
        bad = sorted({provider_slug(e) for e in endpoints} - ok - {""})
        if bad:
            out["ignore"] = bad
    return out


async def resolve(cfg: ResearchConfig, slugs: Iterable[str]) -> dict[str, dict[str, Any]]:
    """``provider`` objects for the run's models, keyed by slug. Off OpenRouter nothing is
    sent; with the live rules off, the feed is not read."""
    slugs = sorted(set(slugs))
    if not cfg.is_openrouter:
        return {s: {} for s in slugs}
    live = cfg.provider_max_price_factor > 0 or cfg.provider_min_uptime > 0
    feeds = (await asyncio.gather(*(fetch_endpoints(s, cfg.provider_routing_ttl) for s in slugs))
             if live else [None] * len(slugs))
    out: dict[str, dict[str, Any]] = {}
    for s, eps in zip(slugs, feeds):
        out[s] = routing(cfg, eps, s)
        if eps is not None:
            healthy = sum(is_healthy(e, cfg.provider_min_uptime) for e in eps)
            log.info("PROVIDER ROUTING %s: %d/%d endpoints healthy; max_price=%s ignore=%s",
                     s, healthy, len(eps), out[s].get("max_price"), out[s].get("ignore") or [])
    return out
