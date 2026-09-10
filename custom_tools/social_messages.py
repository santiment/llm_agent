"""Custom (deployment-specific) tool: Santiment social messages.

Drop-in plugin auto-loaded by ``deep_research_agent.tools.custom``. It lives here
rather than in the agent's source tree so the codebase itself stays generic: nothing
under ``src/`` knows this tool exists, and a deployment without ``DRA_METRICS_HUB_URL``
never loads it.

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
import logging
import os
import re
import urllib.error
import urllib.request

from langchain_core.tools import StructuredTool

log = logging.getLogger("deep_research_agent.custom_tools")

_TIMEOUT = 60
_SLUG_RE = re.compile(r"^[a-z0-9-]+$")
# Source ids metrics-hub accepts; anything else is a server 400, so reject up front.
VALID_SOURCES = ("telegram", "reddit", "twitter_crypto", "4chan", "bitcointalk", "farcaster")

_DESCRIPTION = (
    "What the crowd is saying about a coin: a stratified sample of raw social posts plus a "
    "FULL-POPULATION stats block (total vs sampled, volume curve, sentiment balance, trend "
    "words, top channels) — internal Santiment social data; cite it as 'Santiment social "
    "messages'. `sources` accepts ONLY these ids: " + ", ".join(VALID_SOURCES) + " (default: "
    "all of them). Returns JSON {stats, messages}; each message is tagged with a `stratum`: "
    "`head` (top by engagement), `random` (unbiased base), `poles` (oversampled bull/bear "
    "extremes). Judge prevalence and mood ONLY from the `random` stratum and the stats "
    "block; use head/poles for what spread and where the disagreement is. If the stats say "
    "zero messages, there is NO crowd data for that asset and window: report that plainly "
    "and do NOT substitute web search results for it. Large results are saved to a file: "
    "compute NUMBERS with `execute` (pandas over messages; cite from stats), but READ the "
    "message text (topics, narratives, claims) by handing the file to `extract-subagent` "
    "via the `task` tool — never by printing it into your own context."
)

_NO_DATA_NOTE = (
    "No social messages matched this asset in this window (checked as project slug and as "
    "free text). This is the answer: report 'no crowd data', do not fill the gap with web "
    "search results."
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
            sources: comma-separated source ids from: telegram, reddit, twitter_crypto, 4chan,
                bitcointalk, farcaster (exact ids); default all of them.
            max_words: max total words across the sampled message texts (default 100k; the
                server clamps to its own ceiling). Bigger = more raw posts, more cost/latency.
        """
        asset = (asset or "").strip()
        if not asset:
            return ("social_messages: `asset` is required — a project slug (e.g. 'bitcoin') or a "
                    "search word. Nothing was fetched.")
        body: dict = {"max_words": int(max_words)}
        # Slugs are lowercase; a model that writes 'Bitcoin' means the slug, not a text search.
        by_slug = _SLUG_RE.match(asset.lower()) is not None
        body["slug" if by_slug else "search_text"] = asset.lower() if by_slug else asset
        # The model often emits Santiment-style date math (`utc_now-24h`); ES wants
        # `now-24h`. Normalize so either form (and ISO) works.
        if from_timestamp:
            body["from_timestamp"] = from_timestamp.replace("utc_now", "now")
        if to_timestamp:
            body["to_timestamp"] = to_timestamp.replace("utc_now", "now")
        if sources:
            wanted = [s.strip().lower() for s in sources.split(",") if s.strip()]
            bad = [s for s in wanted if s not in VALID_SOURCES]
            if bad:
                return (f"social_messages: unknown source(s) {', '.join(bad)}. Valid sources: "
                        f"{', '.join(VALID_SOURCES)}. Retry with valid ids or omit `sources`.")
            body["sources"] = ",".join(wanted)

        data = await _call(url, body)
        if isinstance(data, str):
            return data
        if not isinstance(data.get("stats"), dict):
            # Without a stats block there is no population count, so "no crowd data" would
            # be a guess — say the response was malformed instead.
            return f"social_messages: response carries no stats block: {str(data)[:300]}"
        # A slug's curated query can match nothing while the name itself is discussed;
        # fall back to a free-text search before declaring the crowd silent.
        if by_slug and _total(data) == 0:
            text_body = {k: v for k, v in body.items() if k != "slug"}
            text_body["search_text"] = asset
            text_data = await _call(url, text_body)
            if isinstance(text_data, str):
                # The slug matched nothing and the retry never answered, so whether the
                # crowd is silent is unknown — _NO_DATA_NOTE would assert it as fact.
                return (f"social_messages: slug {asset!r} matched no messages and the free-text "
                        f"retry failed, so it is UNKNOWN whether crowd data exists for it. Do "
                        f"not report 'no crowd data'. {text_data}")
            if _total(text_data) > 0:
                data = text_data
                data.setdefault("stats", {})["query_mode"] = "search_text"
                data["stats"]["note"] = (
                    f"The project query for slug {asset!r} matched no messages; these results "
                    f"are a free-text search for {asset!r} instead.")
        out = {"stats": data.get("stats", {}), "messages": data.get("messages", [])}
        if _total(data) == 0 and not out["messages"]:
            out["note"] = _NO_DATA_NOTE
        return json.dumps(out, default=str)

    return [StructuredTool.from_function(
        coroutine=social_messages, name="social_messages", description=_DESCRIPTION)]


def _total(data: dict) -> int:
    """Full-population match count from the stats block (0 when absent/unparseable)."""
    try:
        return int((data.get("stats") or {}).get("total_matching") or 0)
    except (TypeError, ValueError, AttributeError):
        return 0


async def _call(url: str, body: dict) -> dict | str:
    """One POST; the ``data`` dict on success, else a plain error string for the model."""
    try:
        payload = await asyncio.to_thread(_post_json, url, body)
    except Exception as exc:
        log.warning("social_messages request failed: %s", exc)
        return f"social_messages request failed: {exc}"
    if not isinstance(payload, dict):
        return f"social_messages: unexpected response: {str(payload)[:300]}"
    if payload.get("error"):
        return f"social_messages service error: {payload.get('error')}"
    data = payload.get("data")
    if not isinstance(data, dict):
        return f"social_messages: unexpected response shape: {str(payload)[:300]}"
    return data


def _post_json(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310 (trusted internal URL)
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        # urlopen raises on 4xx/5xx and the exception IS the response: metrics-hub puts the
        # real cause in its body, so unread it leaves only "HTTP Error 500: INTERNAL SERVER
        # ERROR". HTTPError subclasses URLError — this clause must stay first.
        raise RuntimeError(f"metrics-hub {url} -> HTTP {exc.code}: {_error_detail(exc)}") from exc
    except urllib.error.URLError as exc:
        # No response at all: DNS, refused, TLS, connect timeout (VPN down). str(exc) omits
        # the host, and the host is the question when the tool is misconfigured.
        raise RuntimeError(f"metrics-hub unreachable at {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        # Timeout while reading the body arrives bare, not wrapped in URLError.
        raise RuntimeError(f"metrics-hub timed out after {_TIMEOUT}s at {url}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # A proxy or a wrong port answers 200 with HTML; show what actually came back.
        raise RuntimeError(f"metrics-hub {url} -> non-JSON response: {raw[:300]!r}") from exc


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """Human-sized cause from an error body: metrics-hub's ``error`` field when it sends
    JSON ({error, trace} — the trace is skipped, it belongs in metrics-hub's own logs and
    would only burn model context), else the raw body, else the status reason."""
    try:
        raw = exc.read().decode("utf-8", "replace").strip()
    except OSError:
        return str(exc.reason or "")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:300] or str(exc.reason or "")
    if isinstance(payload, dict) and (payload.get("error") or payload.get("message")):
        return str(payload.get("error") or payload.get("message"))[:300]
    return raw[:300] or str(exc.reason or "")
