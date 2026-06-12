"""Custom (deployment-specific) tool: Santiment social messages.

Drop-in plugin auto-loaded by ``deep_research_agent.tools.custom`` — this file is
NOT committed (custom_tools/ is gitignored), so the agent codebase stays generic.

Exposes a `social_messages` tool that POSTs to metrics-hub-server's
``/sample_documents`` for a stratified sample of raw social posts about a coin
plus a full-population stats block. Reaches metrics-hub directly over VPN (no
auth), so only enable where the agent itself is access-controlled.

Config: set ``DRA_METRICS_HUB_URL`` (or ``METRICS_HUB_URL``), e.g.
``http://metrics-hub-server:3000``. Unset → the tool is simply not loaded.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.request

from langchain_core.tools import StructuredTool

_TIMEOUT = 60
_SLUG_RE = re.compile(r"^[a-z0-9-]+$")

_DESCRIPTION = (
    "What the crowd is saying about a coin: a stratified sample of raw social posts "
    "(telegram, reddit, twitter, 4chan, bitcointalk, farcaster) plus a FULL-POPULATION "
    "stats block (total vs sampled, volume curve, sentiment balance, trend words, top "
    "channels) — internal Santiment social data; cite it as 'Santiment social messages'. "
    "Returns JSON {stats, messages}; each message is tagged with a `stratum`: `head` (top "
    "by engagement), `random` (unbiased base), `poles` (oversampled bull/bear extremes). "
    "Judge prevalence and mood ONLY from the `random` stratum and the stats block; use "
    "head/poles for what spread and where the disagreement is. Large results are saved to "
    "a file — load it with `execute` (pandas over messages, cite numbers from stats)."
)


def build_tools(cfg) -> list:
    """Loader entrypoint. Returns [social_messages] when a metrics-hub URL is set, else []."""
    base = (os.environ.get("DRA_METRICS_HUB_URL") or os.environ.get("METRICS_HUB_URL") or "").rstrip("/")
    if not base:
        return []
    url = f"{base}/sample_documents"
    # Default sample size (words). Override with DRA_SAMPLE_MAX_WORDS; the agent can also
    # pass max_words per call. The metrics-hub server clamps to its own ceiling (400k).
    default_max_words = int(os.environ.get("DRA_SAMPLE_MAX_WORDS") or 100_000)

    async def social_messages(
        asset: str,
        from_timestamp: str = "",
        to_timestamp: str = "",
        sources: str = "",
        max_words: int = default_max_words,
    ) -> str:
        """Sample raw social messages for a coin with a full-population stats block.

        Args:
            asset: coin slug (e.g. 'bitcoin') or a free search word.
            from_timestamp: window start, ISO-8601 or ES date math ('now-24h'); default 24h ago.
            to_timestamp: window end, ISO-8601 or ES date math ('now'); default now.
            sources: comma-separated sources; default all crowd sources.
            max_words: max total words across the sampled message texts (default 100k; the
                server clamps to its own ceiling). Bigger = more raw posts, more cost/latency.
        """
        body: dict = {"max_words": int(max_words)}
        if asset and _SLUG_RE.match(asset):
            body["slug"] = asset
        else:
            body["search_text"] = asset
        # The model often emits Santiment-style date math (`utc_now-24h`); ES wants
        # `now-24h`. Normalize so either form (and ISO) works.
        if from_timestamp:
            body["from_timestamp"] = from_timestamp.replace("utc_now", "now")
        if to_timestamp:
            body["to_timestamp"] = to_timestamp.replace("utc_now", "now")
        if sources:
            body["sources"] = sources

        try:
            payload = await asyncio.to_thread(_post_json, url, body)
        except Exception as exc:
            return f"social_messages request failed: {exc}"

        if not isinstance(payload, dict):
            return f"social_messages: unexpected response: {str(payload)[:300]}"
        if payload.get("error"):
            return f"social_messages service error: {payload.get('error')}"
        data = payload.get("data")
        if not isinstance(data, dict):
            return f"social_messages: unexpected response shape: {str(payload)[:300]}"
        return json.dumps({"stats": data.get("stats", {}), "messages": data.get("messages", [])},
                          default=str)

    return [StructuredTool.from_function(
        coroutine=social_messages, name="social_messages", description=_DESCRIPTION)]


def _post_json(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310 (trusted internal URL)
        return json.loads(resp.read().decode("utf-8"))
