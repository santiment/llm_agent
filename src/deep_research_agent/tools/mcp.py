"""MCP tool loading via langchain-mcp-adapters, with diagnostics.

MCP connection errors are never swallowed: every failure is surfaced as a
``status`` event AND raised-visible in logs, and ``0.0.0.0`` hosts are rewritten
to loopback in config (see ``_normalize_mcp_url``) — the two things that most
commonly make MCP connections flaky.

Tool LISTINGS are cached per server connection for ``cfg.mcp_tools_ttl`` seconds:
``make_graph`` runs once per research run and the listing is a network round-trip per
server — the single biggest cost of a graph build. The adapter's tools are stateless
(each call opens its own session), so a listed tool is safe to reuse; only the
per-run instrumentation (meter, offload sink, semaphore) is rebuilt every time.
Failures are never cached.
"""

from __future__ import annotations

import asyncio
import logging
import time

from langchain_core.tools import BaseTool

from ..config import ResearchConfig
from ..events import emit, instrument_tool, result_handling

log = logging.getLogger("deep_research_agent.mcp")

# {connection key: (expires_at_monotonic, [raw adapter tools])}
_LISTING_CACHE: dict[tuple, tuple[float, list[BaseTool]]] = {}


def _cache_key(name: str, conn: dict) -> tuple:
    headers = conn.get("headers") or {}
    return (name, conn.get("url"), conn.get("transport"),
            tuple(sorted((str(k), str(v)) for k, v in headers.items())))


def clear_listing_cache() -> None:
    _LISTING_CACHE.clear()


async def _list_tools(client, name: str, conn: dict, ttl: float) -> tuple[list[BaseTool], bool]:
    """``(tools, from_cache)`` — the server's tool listing, reused within ``ttl`` seconds."""
    key = _cache_key(name, conn)
    now = time.monotonic()
    hit = _LISTING_CACHE.get(key) if ttl > 0 else None
    if hit and hit[0] > now:
        return list(hit[1]), True
    tools = await client.get_tools(server_name=name)
    if ttl > 0:
        _LISTING_CACHE[key] = (now + ttl, list(tools))
    return list(tools), False


async def load_mcp_tools(
    cfg: ResearchConfig,
    meter: object | None = None,
    offload_sink: object | None = None,
) -> list[BaseTool]:
    """Connect to each configured MCP server and return its instrumented tools.

    Tools are loaded PER server so we can (a) attribute each tool to its friendly
    source label — the adapter does not prefix tool names, so a flat load loses
    which server a tool came from — and (b) isolate failures: one unreachable
    server no longer takes down the others. Each server dict is tagged in place
    with ``tool_names`` for the citation guidance the agent builds afterwards.
    """
    if not cfg.mcp_servers:
        return []

    from langchain_mcp_adapters.client import MultiServerMCPClient

    connections: dict[str, dict] = {}
    for s in cfg.mcp_servers:
        url = s.get("url")
        if not url:
            continue
        conn: dict = {"url": url, "transport": s.get("transport", "streamable_http")}
        if s.get("headers"):
            conn["headers"] = s["headers"]
        connections[s["name"]] = conn

    if not connections:
        return []

    log.info("MCP connecting: %s", {k: v["url"] for k, v in connections.items()})
    client = MultiServerMCPClient(connections)

    # ONE bounded queue shared by every MCP tool. langchain-mcp-adapters opens a
    # fresh connection per call (no session reuse), so the orchestrator plus all
    # parallel sub-researchers would otherwise open as many simultaneous
    # connections as the model fans out — enough to exhaust the MCP server's file
    # descriptors and trip its rate limiter. The semaphore admits at most N at a
    # time; the rest await a slot.
    gate = asyncio.Semaphore(cfg.mcp_max_concurrency)
    log.info("MCP call concurrency capped at %d", cfg.mcp_max_concurrency)

    out: list[BaseTool] = []
    for s in cfg.mcp_servers:
        name = s.get("name")
        if name not in connections:
            continue
        try:
            tools, cached = await _list_tools(client, name, connections[name], cfg.mcp_tools_ttl)
        except Exception as exc:
            log.exception("MCP connection failed: %s", name)
            emit({"type": "status", "state": "mcp_error", "detail": str(exc),
                  "server": name, "label": s.get("label")})
            s["tool_names"] = []
            continue

        allow = set(s.get("tools") or [])
        if allow:
            tools = [t for t in tools if t.name in allow]

        s["tool_names"] = [t.name for t in tools]
        if not (connections[name].get("headers") or {}).get("Authorization"):
            # Listing needs no credentials on most servers, so a missing token shows up
            # only later, as every tools/call failing — say it now instead.
            log.warning("MCP %s: no Authorization header configured (DRA_MCP_BEARER / "
                        "headers) — if the server requires auth, tool calls will fail "
                        "with HTTP 401 although the listing succeeded", name)
        out.extend(
            instrument_tool(
                t, kind="mcp", semaphore=gate,
                rate_limit_max_wait=cfg.mcp_rate_limit_max_wait,
                **result_handling(cfg, meter, offload_sink),
            )
            for t in tools
        )
        log.info("MCP %s (%s): %d tools%s: %s", name, s.get("label"), len(tools),
                 " (listing cached)" if cached else "", s["tool_names"])

    emit({"type": "status", "state": "mcp_ready", "tool_count": len(out),
          "tools": [t.name for t in out]})
    return out
