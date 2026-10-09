"""``./run.sh tiers`` — which models each tier runs, and how fast they answer right now.

Without a flag it reads only OpenRouter's public endpoint feed (no key, no cost): every tier's
models, each model's price range and healthy endpoints, and the ``provider`` object the agent
would send with it (``provider_routing.resolve``). ``--measure N`` also times N short requests
per distinct (model, reasoning) pair the tiers use, routed exactly like the agent's own calls,
and reports which provider served them, the time to the first token and the output speed.
That spends real tokens on ``OPENAI_API_KEY`` — a few cents at N=3.

The feed carries no speed data (``throughput_last_30m`` is null), so timing our own calls is
the only throughput figure that reflects our routing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from . import provider_routing
from .config import MODEL_TIERS, ResearchConfig
from .models import reasoning_param

# The role each slot's model is built for (agent.py), which sets its reasoning effort.
SLOT_ROLES = {
    "research_model": "orchestrator",
    "subagent_model": "research-subagent",
    "utility_model": "extract-subagent",
    "compaction_model": "compaction",
    "coding_model": "coding-subagent",
}
TIER_ORDER = ["extra-low", "low", "mid", "high"]

# Fixed-length prose, so output speed compares across models; no tools, so every model can
# answer it as is.
PROMPT = ("Write about 250 words on the history of the printing press, as plain prose with no "
          "headings or lists.")
MAX_TOKENS = 600


@dataclass
class Sample:
    """One timed call: seconds to the first streamed token and to the end, the output tokens
    billed (reasoning included), and the provider OpenRouter routed it to."""

    provider: str
    ttft: float | None
    seconds: float
    output_tokens: int | None
    error: str | None = None

    @property
    def tokens_per_second(self) -> float | None:
        if not self.output_tokens or self.ttft is None or self.seconds <= self.ttft:
            return None
        return self.output_tokens / (self.seconds - self.ttft)


def tier_rows(tiers: dict[str, dict[str, str]] = MODEL_TIERS) -> list[tuple[str, str, str]]:
    """``(tier, slot, model)`` for every slot, tiers cheapest first."""
    order = [t for t in TIER_ORDER if t in tiers] + [t for t in tiers if t not in TIER_ORDER]
    return [(tier, slot, tiers[tier][slot]) for tier in order for slot in SLOT_ROLES
            if slot in tiers[tier]]


def endpoint_summary(cfg: ResearchConfig, endpoints: list[dict] | None) -> str:
    """``healthy/total endpoints, $in range / $out range`` from the feed; "no feed" without one."""
    if not endpoints:
        return "no feed"
    healthy = sum(provider_routing.is_healthy(e, cfg.provider_min_uptime) for e in endpoints)

    def span(key: str) -> str:
        prices = []
        for e in endpoints:
            try:
                prices.append(float(e["pricing"][key]) * 1e6)
            except (KeyError, TypeError, ValueError):
                continue
        if not prices:
            return "?"
        lo, hi = min(prices), max(prices)
        return f"${lo:.3g}" if lo == hi else f"${lo:.3g}–{hi:.3g}"

    return f"{healthy}/{len(endpoints)} healthy, in {span('prompt')} / out {span('completion')}"


class StreamTimer:
    """Reads an OpenRouter SSE stream line by line AS IT ARRIVES: the first delta with text
    or reasoning stops the time-to-first-token clock, the final chunk's usage gives the
    output tokens, and every chunk names the provider. Comments
    (``: OPENROUTER PROCESSING``) are keep-alives."""

    def __init__(self, started: float) -> None:
        self.started = started
        self.provider = "?"
        self.ttft: float | None = None
        self.tokens: int | None = None
        self.error: str | None = None

    def feed(self, line: str, now: float) -> None:
        if not line.startswith("data:"):
            return
        raw = line[5:].strip()
        if raw == "[DONE]":
            return
        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            return
        self.provider = chunk.get("provider") or self.provider
        err = chunk.get("error")
        if err:
            self.error = str(err.get("message") if isinstance(err, dict) else err)[:200]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if self.ttft is None and (delta.get("content") or delta.get("reasoning")):
                self.ttft = now - self.started
        usage = chunk.get("usage") or {}
        if usage.get("completion_tokens") is not None:
            self.tokens = int(usage["completion_tokens"])

    def sample(self, now: float) -> Sample:
        return Sample(provider=self.provider, ttft=self.ttft, seconds=now - self.started,
                      output_tokens=self.tokens, error=self.error)


async def measure_once(client, cfg: ResearchConfig, model: str, role: str,
                       provider: dict[str, Any]) -> Sample:
    body: dict[str, Any] = {"model": model, "stream": True, "max_tokens": MAX_TOKENS,
                            "messages": [{"role": "user", "content": PROMPT}],
                            "usage": {"include": True}}
    reasoning = reasoning_param(cfg, model, role)
    if reasoning is not None:
        body["reasoning"] = reasoning
    if provider:
        body["provider"] = provider
    started = time.monotonic()
    try:
        async with client.stream("POST", cfg.base_url.rstrip("/") + "/chat/completions",
                                 json=body,
                                 headers={"Authorization": f"Bearer {cfg.openai_api_key}"}) as r:
            if r.status_code != 200:
                text = (await r.aread()).decode("utf-8", "replace")[:200]
                return Sample("?", None, time.monotonic() - started, None,
                              f"HTTP {r.status_code}: {text}")
            timer = StreamTimer(started)
            async for line in r.aiter_lines():
                timer.feed(line, time.monotonic())
    except Exception as exc:  # a timeout or a dropped connection is a measurement, not a crash
        return Sample("?", None, time.monotonic() - started, None, f"{type(exc).__name__}: {exc}")
    return timer.sample(time.monotonic())


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(samples: list[Sample]) -> str:
    ok = [s for s in samples if s.error is None]
    errors = len(samples) - len(ok)
    ttft = _mean([s.ttft for s in ok if s.ttft is not None])
    tps = _mean([s.tokens_per_second for s in ok if s.tokens_per_second is not None])
    providers = Counter(s.provider for s in ok)
    served = ", ".join(f"{p}×{n}" for p, n in providers.most_common()) or "-"
    parts = [f"first token {ttft:.1f}s" if ttft is not None else "first token ?",
             f"{tps:.0f} tok/s" if tps is not None else "? tok/s", f"via {served}"]
    if errors:
        first = next(s.error for s in samples if s.error)
        parts.append(f"{errors} failed ({first})")
    return " · ".join(parts)


async def _report(n: int) -> int:
    cfg = ResearchConfig.from_runnable_config({})
    rows = tier_rows()
    models = sorted({m for _, _, m in rows})
    feeds: dict[str, list[dict] | None] = {m: None for m in models}
    routing: dict[str, dict] = {m: {} for m in models}
    if cfg.is_openrouter:
        fetched = await asyncio.gather(*(provider_routing.fetch_endpoints(
            m, cfg.provider_routing_ttl) for m in models))
        feeds = dict(zip(models, fetched))
        routing = await provider_routing.resolve(cfg, models)
    else:
        print(f"base_url {cfg.base_url} is not OpenRouter — no endpoint feed or routing object.")

    current = None
    for tier, slot, model in rows:
        if tier != current:
            current = tier
            print(f"\n{tier}")
        role = SLOT_ROLES[slot]
        effort = cfg.reasoning_effort_for(role) or "provider default"
        print(f"  {slot:<17} {model:<34} reasoning {effort:<8} {endpoint_summary(cfg, feeds[model])}")
    print("\nrouting sent per model:")
    for m in models:
        print(f"  {m:<34} {json.dumps(routing.get(m) or {})}")

    if n <= 0:
        print("\n(add --measure N to time N calls per model; spends OPENAI_API_KEY tokens)")
        return 0
    if not cfg.openai_api_key:
        print("\n--measure needs OPENAI_API_KEY (.env).")
        return 1

    import httpx

    pairs = sorted({(m, SLOT_ROLES[s]) for _, s, m in rows},
                   key=lambda p: (p[0], cfg.reasoning_effort_for(p[1])))
    seen: set[tuple[str, dict | None]] = set()
    print(f"\nmeasured ({n} call{'s' if n != 1 else ''} each, {MAX_TOKENS} max output tokens):")
    async with httpx.AsyncClient(timeout=cfg.request_timeout) as client:
        for model, role in pairs:
            key = (model, json.dumps(reasoning_param(cfg, model, role)))
            if key in seen:  # two roles with the same effort are the same request
                continue
            seen.add(key)
            samples = [await measure_once(client, cfg, model, role, routing.get(model) or {})
                       for _ in range(n)]
            effort = cfg.reasoning_effort_for(role) or "default"
            print(f"  {model:<34} reasoning {effort:<8} {summarize(samples)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="./run.sh tiers", description=__doc__.split("\n\n")[0])
    parser.add_argument("--measure", type=int, default=0, metavar="N",
                        help="time N calls per model through OpenRouter (spends tokens)")
    args = parser.parse_args(argv)
    return asyncio.run(_report(args.measure))


if __name__ == "__main__":
    raise SystemExit(main())
