"""Per-model OpenRouter provider routing from the live endpoint feed.

Rules, all config knobs: a model's providers with no HEALTHY endpoint (status ok,
``uptime_last_30m`` >= ``provider_min_uptime``) go on ``ignore``, and the soft
``preferred_*`` thresholds order what is left. The feed (``GET
/api/v1/models/{slug}/endpoints``, public) is re-read every ``provider_routing_ttl``
seconds; unreachable or empty -> soft preferences only, logged; nothing here can fail a
graph build.

No price cap. One was here (1.25x the cheapest healthy endpoint) until a near-free FP4
listing ($0.008 / $0.079 against ~$0.13 / $0.26 for the rest) became the baseline and
pinned every call of the model to that single endpoint. Price stays OpenRouter's job: its
default balancing is price-weighted already.

An ignore list OpenRouter cannot satisfy is a 404 on the call, not a fallback: its
account-side filters (tier, data policy) run first and the public feed knows nothing of
them. ``relax`` is the ladder the model-call fallback climbs on that 404
(``model_errors.ProviderRoutingFallbackMiddleware``): drop the provider lists, then the
whole object. ``remember_relaxed`` keeps the level that worked per model for
``provider_routing_ttl`` seconds, so every role sharing the model and the next graph build
start there instead of paying a refused call each.
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
    _RELAXED.clear()


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
    """One model's ``provider`` object: soft preferences always; the ignore list when the
    feed has at least one healthy endpoint (ignoring every provider would leave nothing)."""
    out = provider_preferences(cfg)
    if not endpoints:
        return out
    healthy = [e for e in endpoints if is_healthy(e, cfg.provider_min_uptime)]
    if not healthy:
        log.warning("PROVIDER ROUTING %s: no endpoint at >= %.0f%% uptime — no ignore list",
                    slug, cfg.provider_min_uptime)
        return out
    if cfg.provider_min_uptime > 0:
        ok = {provider_slug(e) for e in healthy}
        bad = sorted({provider_slug(e) for e in endpoints} - ok - {""})
        if bad:
            out["ignore"] = bad
    return out


# Hard constraints in the order the fallback gives them up: the provider lists, then the
# whole object (OpenRouter's own routing).
_RELAX_STEPS: tuple[tuple[str, ...], ...] = (("ignore", "order", "only"),)
MAX_RELAX_LEVEL = len(_RELAX_STEPS) + 1


def relax(provider: dict[str, Any] | None, level: int) -> dict[str, Any]:
    """The routing object with its hard constraints relaxed ``level`` steps: 1 drops the
    ignore/order/only lists, ``MAX_RELAX_LEVEL`` everything — the bare model, routed by
    OpenRouter alone. 0 is a copy, unchanged."""
    if level >= MAX_RELAX_LEVEL:
        return {}
    out = dict(provider or {})
    for keys in _RELAX_STEPS[:max(0, level)]:
        for k in keys:
            out.pop(k, None)
    return out


def next_relax_level(provider: dict[str, Any] | None, level: int) -> int | None:
    """The first level above ``level`` that changes what is sent (``provider`` is the
    ORIGINAL object), or None when nothing is left to give up."""
    current = relax(provider, level)
    for lvl in range(level + 1, MAX_RELAX_LEVEL + 1):
        if relax(provider, lvl) != current:
            return lvl
    return None


# slug -> (expires_at, level): the relaxation a model turned out to need. Module-level like
# the feed cache and for the same TTL: the next graph build re-reads the feed AND starts at
# this level, and every parallel role on the model skips the refused call.
_RELAXED: dict[str, tuple[float, int]] = {}


def remember_relaxed(slug: str, level: int, ttl: float) -> None:
    if slug and ttl > 0 and level > 0:
        _RELAXED[slug] = (time.monotonic() + ttl, max(level, relaxed_level(slug)))


def relaxed_level(slug: str) -> int:
    """The remembered level for ``slug``, 0 when none or expired."""
    hit = _RELAXED.get(slug)
    if not hit:
        return 0
    if hit[0] <= time.monotonic():
        _RELAXED.pop(slug, None)
        return 0
    return hit[1]


def admitted(cfg: ResearchConfig, endpoints: list[dict] | None,
             provider: dict[str, Any] | None) -> list[dict]:
    """The endpoints a call may land on under ``provider``: healthy and not on ``ignore``.
    When that admits none, every healthy endpoint — the call goes wherever OpenRouter
    sends it."""
    healthy = [e for e in endpoints or [] if is_healthy(e, cfg.provider_min_uptime)]
    ignore = set((provider or {}).get("ignore") or [])
    out = [e for e in healthy if provider_slug(e) not in ignore]
    return out or healthy


def context_window(cfg: ResearchConfig, endpoints: list[dict] | None,
                   provider: dict[str, Any] | None) -> int | None:
    """The context window (tokens) a model can count on: the SMALLEST among the endpoints the
    routing admits, so it holds whichever of them serves the call. None without feed data."""
    windows = []
    for e in admitted(cfg, endpoints, provider):
        try:
            if int(e.get("context_length") or 0) > 0:
                windows.append(int(e["context_length"]))
        except (TypeError, ValueError):
            continue
    return min(windows) if windows else None


async def context_windows(cfg: ResearchConfig, slugs: Iterable[str],
                          routing: dict[str, dict[str, Any]] | None = None) -> dict[str, int | None]:
    """Per model, the window ``context_window`` derives — from the same cached feed
    ``resolve`` read, so no second request within the TTL. Every slug None off OpenRouter
    (no feed) or when the feed is unreachable: callers fall back to absolute limits."""
    slugs = sorted(set(slugs))
    if not cfg.is_openrouter:
        return {s: None for s in slugs}
    feeds = await asyncio.gather(*(fetch_endpoints(s, cfg.provider_routing_ttl) for s in slugs))
    return {s: context_window(cfg, eps, (routing or {}).get(s)) for s, eps in zip(slugs, feeds)}


async def resolve(cfg: ResearchConfig, slugs: Iterable[str]) -> dict[str, dict[str, Any]]:
    """``provider`` objects for the run's models, keyed by slug. Off OpenRouter nothing is
    sent; with the live rules off, the feed is not read."""
    slugs = sorted(set(slugs))
    if not cfg.is_openrouter:
        return {s: {} for s in slugs}
    live = cfg.provider_min_uptime > 0
    feeds = (await asyncio.gather(*(fetch_endpoints(s, cfg.provider_routing_ttl) for s in slugs))
             if live else [None] * len(slugs))
    out: dict[str, dict[str, Any]] = {}
    for s, eps in zip(slugs, feeds):
        out[s] = routing(cfg, eps, s)
        if eps is not None:
            healthy = sum(is_healthy(e, cfg.provider_min_uptime) for e in eps)
            log.info("PROVIDER ROUTING %s: %d/%d endpoints healthy; ignore=%s",
                     s, healthy, len(eps), out[s].get("ignore") or [])
        level = relaxed_level(s)
        if level and (relaxed := relax(out[s], level)) != out[s]:
            log.warning("PROVIDER ROUTING %s: starting at relax level %d/%d (%s) — OpenRouter "
                        "refused the full routing object within the last %.0fs",
                        s, level, MAX_RELAX_LEVEL, relaxed or "bare", cfg.provider_routing_ttl)
            out[s] = relaxed
    return out
