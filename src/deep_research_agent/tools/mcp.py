"""MCP tool loading via langchain-mcp-adapters, with diagnostics.

MCP connection errors are never swallowed: every failure is surfaced as a
``status`` event AND raised-visible in logs, and ``0.0.0.0`` hosts are rewritten
to loopback in config (see ``_normalize_mcp_url``) — the two things that most
commonly make MCP connections flaky.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import BaseTool

from ..config import ResearchConfig
from ..events import emit, instrument_tool

log = logging.getLogger("deep_research_agent.mcp")


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
            tools = await client.get_tools(server_name=name)
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
        out.extend(
            instrument_tool(
                t, kind="mcp", semaphore=gate,
                rate_limit_max_wait=cfg.mcp_rate_limit_max_wait,
                max_result_chars=cfg.max_result_chars,
                max_result_rows=cfg.max_result_rows,
                meter=meter,
                offload_sink=offload_sink,
                offload_dir=cfg.offload_dir,
            )
            for t in tools
        )
        log.info("MCP %s (%s): %d tools: %s",
                 name, s.get("label"), len(tools), s["tool_names"])

    emit({"type": "status", "state": "mcp_ready", "tool_count": len(out),
          "tools": [t.name for t in out]})
    return out
