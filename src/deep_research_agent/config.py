"""Self-contained run configuration. No host-app imports — this is the seam
that keeps the agent portable.

Resolution order for every field: per-run ``configurable`` override  ->  env var
->  default. The ``configurable`` keys accept BOTH this package's native names
AND a set of compatibility aliases (``apiKeys``, ``mcp_config``, ``mcp_prompt``)
so an existing caller can adopt the agent with zero backend changes.

Exception — models: chosen by tier NAME only (``model_tier`` / ``DRA_MODEL_TIER``);
the models behind each name live in ``MODEL_TIERS`` (code). Per-model keys and env
vars are deliberately not honored (legacy ones are ignored with a warning).
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger("deep_research_agent.config")


def _default_skills_dir() -> str:
    """The repo's ``./skills`` directory. This file lives at
    ``src/deep_research_agent/config.py`` -> project root is two parents up."""
    return str(Path(__file__).resolve().parents[2] / "skills")

# Hostnames that resolve to cloud-metadata endpoints — never a legitimate MCP target.
_BLOCKED_MCP_HOSTNAMES = {"metadata", "metadata.google.internal"}

# Named model packages ("price tiers") — THE ONLY place models are chosen. Callers
# and the environment select a package by NAME (configurable ``model_tier`` /
# env ``DRA_MODEL_TIER``); individual models are not settable per run or per env, so
# every model that can ever run is named here, in one reviewed place.
# DEFAULT_MODEL_TIER applies when nothing is configured — the cheapest package, so a
# bare checkout can't silently burn money; production callers opt UP explicitly.
# To add a packaging: add an entry here, pick a name, document it in the README's
# Model tiers table — callers then just set model_tier=<name>.
# OpenRouter slugs. The inline ($in/$out per 1M tokens) figures were verified live on
# 2026-06-11 — they DRIFT; re-check on OpenRouter before relying on them or editing.
MODEL_TIERS: dict[str, dict[str, str]] = {
    # Rock bottom: deepseek-v4-flash for both tool-loop roles — it's already this tier's
    # orchestrator AND the `low` tier's sub-agent, so it's proven reliable at the job
    # here (unlike gpt-oss, which gave up and returned empty findings). Same model for
    # both means the sub-agent is never pricier than the orchestrator, and delegation
    # still pays off via context isolation (raw data stays in sub-agent contexts).
    # Utility is a cheaper Qwen for input-heavy map/extract. The orchestrator still plans
    # worse + quits earlier than higher tiers — the force-completion / findings-gate /
    # budget backstops keep runs honest, not great. Demos, smoke tests, high-volume
    # low-stakes ticks; not for decisions.
    "extra-low": {
        "research_model": "deepseek/deepseek-v4-flash",            # $0.10 / $0.20
        "subagent_model": "deepseek/deepseek-v4-flash",            # $0.10 / $0.20
        "utility_model": "qwen/qwen3-30b-a3b-instruct-2507",       # $0.05 / $0.19
    },
    # Cheapest sane agent: v4-pro orchestrator over v4-flash workers. deepseek-v4-flash
    # streaming is force-disabled via streaming_denylist (known off-spec chunks).
    "low": {
        "research_model": "deepseek/deepseek-v4-pro",              # $0.44 / $0.87
        "subagent_model": "deepseek/deepseek-v4-flash",            # $0.10 / $0.20
        "utility_model": "deepseek/deepseek-v4-flash",             # $0.10 / $0.20
    },
    # The value sweet spot: current-gen flash orchestrator, cheaper flash workers.
    "mid": {
        "research_model": "google/gemini-3.5-flash",               # $1.50 / $9.00
        "subagent_model": "google/gemini-2.5-flash",               # $0.30 / $2.50
        "utility_model": "deepseek/deepseek-v4-flash",             # $0.10 / $0.20
    },
    # Best research quality. Opus plans and synthesizes ONLY — sub-agent/utility stay
    # sonnet/haiku tier on purpose (an Opus sub-agent fleet defeats the tiering).
    "high": {
        "research_model": "anthropic/claude-opus-4.8",             # $5.00 / $25.00
        "subagent_model": "anthropic/claude-sonnet-4.6",           # $3.00 / $15.00
        "utility_model": "anthropic/claude-haiku-4.5",             # $1.00 / $5.00
    },
}

DEFAULT_MODEL_TIER = "extra-low"


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def _allowed_base_urls() -> set[str]:
    """Trusted OpenAI-compatible endpoints. The server-side env/default is always
    allowed; operators may add more via ``DRA_ALLOWED_BASE_URLS`` (comma-separated)."""
    allowed = {_env("OPENAI_BASE_URL", default="https://openrouter.ai/api/v1").rstrip("/")}
    for u in _env("DRA_ALLOWED_BASE_URLS").split(","):
        u = u.strip().rstrip("/")
        if u:
            allowed.add(u)
    return allowed


def _mcp_url_blocked(url: str) -> str | None:
    """SSRF defense-in-depth for a per-run MCP URL. Returns a reason string if the URL
    must be refused, else ``None``. Loopback/private hosts are ALLOWED (the internal
    gateway legitimately uses them); only non-http(s) schemes and link-local / cloud-
    metadata targets (169.254.0.0/16, fe80::/10, ``metadata.*``) are blocked."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"scheme {parsed.scheme!r} not allowed"
    host = (parsed.hostname or "").lower()
    if not host:
        return "missing host"
    if host in _BLOCKED_MCP_HOSTNAMES:
        return f"blocked metadata host {host!r}"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None  # a DNS name (e.g. host.docker.internal) — not an IP literal to vet
    if ip.is_link_local:
        return f"link-local address {host} blocked"
    return None


def _slug_from_url(url: str) -> str:
    """Stable connection key (and tool-name prefix) from an MCP URL's last path
    segment: ``.../mcp/data-provider`` -> ``data_provider``."""
    seg = (urlparse(url).path.rstrip("/").rsplit("/", 1)[-1] or "").lower()
    return re.sub(r"[^a-z0-9]+", "_", seg).strip("_")


def _strip_provider(model_id: str) -> str:
    """``openai:anthropic/claude`` -> ``anthropic/claude`` (OpenRouter wants the bare slug)."""
    return model_id.split(":", 1)[1] if model_id.startswith("openai:") else model_id


def _normalize_mcp_url(url: str) -> str:
    """Make an MCP URL dialable and well-formed.

    - ``0.0.0.0`` is a *bind* address; dialing it fails on many stacks -> loopback.
    - Append ``/mcp`` ONLY for a bare host (no path). If a path is already present
      (e.g. ``/mcp/data-provider``) leave it untouched — appending would 404.
    """
    url = url.strip().rstrip("/")
    url = url.replace("://0.0.0.0", "://127.0.0.1")
    path = urlparse(url).path
    if not path or path == "/":
        url = url + "/mcp"
    return url


@dataclass
class ResearchConfig:
    openai_api_key: str
    base_url: str
    tavily_api_key: str
    research_model: str
    report_model: str
    # Model tiering — smart orchestrator, cheap sub-agents. The orchestrator plans,
    # delegates and synthesizes on research_model; sub-agents run their tool loops on
    # subagent_model (typically a tier down); utility_model is the floor (flash-class)
    # for pure map/extract/verify work that needs no tool-loop judgment. All three are
    # filled from the selected MODEL_TIERS package (DEFAULT_MODEL_TIER when none
    # chosen) — never settable individually by env or caller. utility_model has no
    # consumer yet — plumbed now so the verifier / compaction / map-worker features
    # configure against a stable key.
    subagent_model: str
    utility_model: str
    temperature: float = 0.0
    search_max_results: int = 6
    max_concurrent_units: int = 3
    # Hard ceiling on SIMULTANEOUS MCP tool calls across the whole run (orchestrator +
    # all parallel sub-researchers share it). langchain-mcp-adapters opens a NEW
    # streamable_http connection per call, so without this the agent's fan-out can open
    # hundreds of sockets at once and exhaust the MCP server's file descriptors / trip
    # its rate limiter. 10 keeps throughput high while staying well under the limit;
    # lower it if the server still strains.
    mcp_max_concurrency: int = 10
    # Per-call rate-limit backoff budget (seconds): on a 429 the MCP tool wrapper waits
    # and retries until cumulative backoff would exceed this, then fails — bounded so a
    # permanently-throttled server can't hang a run forever. Same altitude as
    # mcp_max_concurrency so both throttle knobs are operator-tunable.
    mcp_rate_limit_max_wait: float = 120.0
    # Each server dict carries a human-friendly ``label`` (e.g. "Data Provider
    # MCP") used in the report's Sources, plus the connection ``name``
    # (tool-name prefix), ``url``, ``headers`` and optional ``tools`` allow-list.
    mcp_servers: list[dict] = field(default_factory=list)
    mcp_prompt: str = ""
    # Directory of agent skills (folders each containing a SKILL.md). Loaded read-only
    # at startup. For now a single local dir; a future loader will layer system-wide +
    # per-user skills here.
    skills_dir: str = ""
    # Stream model output token-by-token (drives the live "thinking" UI). Set
    # DRA_STREAMING=false to fetch full responses in one shot — a workaround for models
    # whose off-spec streaming chunks merge into doubled/dropped metadata.
    streaming: bool = True
    # Substrings of model ids whose OpenRouter streaming is off-spec: chunks merge into
    # DOUBLED metadata (finish_reason "stopstop", doubled model_name) and tool_calls get
    # dropped, stalling the loop. Streaming is force-disabled for matches (models.py).
    # Override via DRA_STREAMING_DENYLIST (comma-separated substrings).
    streaming_denylist: list[str] = field(default_factory=lambda: ["deepseek-v4-flash"])
    # LangGraph super-step ceiling for the orchestrator (agent.py clamps deepagents' 9_999).
    # ~7 super-steps per ReAct loop here, so this caps loops, not tool calls. Must stay ABOVE
    # max_tool_calls × steps-per-loop so BudgetMiddleware — the real runaway guard — binds
    # first.
    recursion_limit: int = 4500
    # Cumulative ceilings per run (BudgetMiddleware), soft wrap-up nudge at 75%. Large results
    # OFFLOAD to the sandbox instead of piling into context (see offload_results): a many-call
    # scan (e.g. a large cross-entity sweep) doesn't grow the token footprint per call, so the
    # call ceiling can be generous. Without a sandbox these still backstop runaway runs.
    max_tool_calls: int = 200
    max_total_tokens: int = 4_000_000
    # Per-call MCP result threshold (events.py). With a sandbox, a result over EITHER bound is
    # written to a file under offload_dir and only a compact stub (path, row count, columns,
    # head) enters context — the model then processes the file with the `execute` tool. Without
    # a sandbox these are hard truncation caps. Keep them modest on purpose:
    # the point is to keep context lean, not to fit more rows in it.
    max_result_chars: int = 60_000
    max_result_rows: int = 1000
    # Offload large MCP results to the sandbox filesystem rather than truncating them. No-op
    # when no sandbox is configured (falls back to truncation). offload_dir is inside the
    # container's persistent /workspace, so a later `execute` call can read the files back.
    offload_results: bool = True
    offload_dir: str = "/workspace/data"
    # Code-execution sandbox sidecar (projects/llm_sandbox). When sandbox_url is set, the
    # agent's DEFAULT filesystem backend becomes the sandbox and deepagents' `execute` tool is
    # enabled — the model runs REAL shell/python/js in the container. Empty → in-memory
    # StateBackend (no execution). sandbox_token must match the service's LLM_SANDBOX_TOKEN.
    sandbox_url: str = ""
    sandbox_token: str = ""
    sandbox_network: bool = False
    sandbox_session_timeout: int = 900

    @classmethod
    def from_runnable_config(cls, config: dict | None) -> "ResearchConfig":
        c = (config or {}).get("configurable", {}) or {}
        keys = c.get("apiKeys") or {}

        openai_key = (keys.get("OPENAI_API_KEY") or c.get("openai_api_key")
                      or _env("OPENAI_API_KEY", "OPENROUTER_API_KEY"))
        tavily_key = (keys.get("TAVILY_API_KEY") or c.get("tavily_api_key")
                      or _env("TAVILY_API_KEY"))
        # base_url allowlist: a hostile `configurable.base_url` would receive the server's
        # API key as a Bearer token (key exfiltration). Honor an override only if it is on
        # the allowlist; otherwise fall back to the trusted env/default.
        trusted_base = _env("OPENAI_BASE_URL", default="https://openrouter.ai/api/v1")
        requested_base = (c.get("base_url") or "").rstrip("/")
        if requested_base and requested_base in _allowed_base_urls():
            base_url = requested_base
        else:
            if requested_base:
                log.warning("ignoring non-allowlisted base_url override: %s", requested_base)
            base_url = trusted_base

        # Named package (see MODEL_TIERS); the cheapest one when nothing is configured.
        # Per-model keys/env still win slot-by-slot.
        tier_name = (c.get("model_tier")
                     or _env("DRA_MODEL_TIER", default=DEFAULT_MODEL_TIER)).strip().lower()
        tier = MODEL_TIERS.get(tier_name)
        if tier is None:
            log.warning("unknown model_tier %r — using %r (known tiers: %s)",
                        tier_name, DEFAULT_MODEL_TIER, ", ".join(sorted(MODEL_TIERS)))
            tier = MODEL_TIERS[DEFAULT_MODEL_TIER]

        # Models come ONLY from the tier package — deliberately no per-model env vars
        # and no per-model configurable keys. The env and the caller pick a NAME
        # (DRA_MODEL_TIER / configurable.model_tier); which models that name means is
        # decided in code (MODEL_TIERS), in one reviewed place. Callers still sending
        # the legacy per-model keys get a warning, not silent ignoring.
        _ignored = [k for k in ("research_model", "subagent_model", "utility_model",
                                "final_report_model", "report_model", "compression_model")
                    if c.get(k)]
        if _ignored:
            log.warning("per-run model selection is disabled — ignoring %s; "
                        "pick a package via configurable.model_tier instead", _ignored)
        research_model = _strip_provider(tier["research_model"])
        report_model = research_model  # reserved for a future dedicated synthesis step
        subagent_model = _strip_provider(tier.get("subagent_model") or research_model)
        utility_model = _strip_provider(tier.get("utility_model") or subagent_model)

        # MCP servers, in precedence order: native `mcp_servers`, compat single
        # `mcp_config`, `DRA_MCP_SERVERS` (JSON list of {label,url,...}), or a single
        # `DRA_MCP_URL` (+ `DRA_MCP_LABEL`). Each entry may carry a friendly `label`.
        mcp_servers = c.get("mcp_servers") or []
        if not mcp_servers and c.get("mcp_config"):
            mc = c["mcp_config"]
            # Compat contract: url is a BASE and the client appends "/mcp"
            # (the URL may already carry a path like /threads/<id>, so we append
            # explicitly here rather than relying on the bare-host rule below).
            base = (mc.get("url") or "").rstrip("/")
            mcp_servers = [{
                "name": mc.get("name", "mcp"),
                "label": mc.get("label", ""),
                "url": (base + "/mcp") if base else "",
                "tools": mc.get("tools") or [],
                "headers": mc.get("headers") or {},
            }]
        if not mcp_servers and _env("DRA_MCP_SERVERS"):
            try:
                parsed = json.loads(_env("DRA_MCP_SERVERS"))
                mcp_servers = parsed if isinstance(parsed, list) else []
            except (ValueError, TypeError):
                mcp_servers = []
        if not mcp_servers and _env("DRA_MCP_URL"):
            mcp_servers = [{"url": _env("DRA_MCP_URL"), "label": _env("DRA_MCP_LABEL")}]

        # Normalize URLs, drop SSRF-unsafe targets, derive a connection key + friendly
        # label, attach bearer auth.
        bearer = _env("DRA_MCP_BEARER")
        safe_servers: list[dict] = []
        for s in mcp_servers:
            if s.get("url"):
                s["url"] = _normalize_mcp_url(s["url"])
            blocked = _mcp_url_blocked(s.get("url", "")) if s.get("url") else "missing url"
            if blocked:
                log.warning("refusing MCP server %s: %s", s.get("url") or "(none)", blocked)
                continue
            if not s.get("name"):
                s["name"] = _slug_from_url(s.get("url", "")) or "mcp"
            if not (s.get("label") or "").strip():
                # No explicit label → derive a readable one from the slug name
                # ("data_provider" -> "Data Provider"), never the generic placeholder.
                s["label"] = s["name"].replace("_", " ").replace("-", " ").title()
            if bearer and not (s.get("headers") or {}).get("Authorization"):
                s.setdefault("headers", {})["Authorization"] = f"Bearer {bearer}"
            safe_servers.append(s)
        mcp_servers = safe_servers

        return cls(
            openai_api_key=openai_key,
            base_url=base_url,
            tavily_api_key=tavily_key,
            research_model=research_model,
            report_model=report_model,
            subagent_model=subagent_model,
            utility_model=utility_model,
            temperature=float(c.get("temperature", 0.0) or 0.0),
            search_max_results=int(c.get("search_max_results", 6) or 6),
            max_concurrent_units=int(
                c.get("max_concurrent_research_units")
                or c.get("max_concurrent_units")
                or 3),
            mcp_max_concurrency=max(1, int(
                c.get("mcp_max_concurrency")
                or _env("DRA_MCP_MAX_CONCURRENCY")
                or 10)),
            mcp_rate_limit_max_wait=float(
                c.get("mcp_rate_limit_max_wait")
                or _env("DRA_MCP_RATE_LIMIT_MAX_WAIT")
                or 120.0),
            mcp_servers=mcp_servers,
            mcp_prompt=c.get("mcp_prompt") or "",
            skills_dir=c.get("skills_dir") or _env("DRA_SKILLS_DIR") or _default_skills_dir(),
            streaming=(
                bool(c["streaming"]) if "streaming" in c
                else _env("DRA_STREAMING", default="true").strip().lower()
                not in ("0", "false", "no", "off")
            ),
            streaming_denylist=(
                [s.strip().lower() for s in c["streaming_denylist"] if str(s).strip()]
                if isinstance(c.get("streaming_denylist"), list)
                else [s.strip().lower() for s in
                      _env("DRA_STREAMING_DENYLIST", default="deepseek-v4-flash").split(",")
                      if s.strip()]
            ),
            recursion_limit=int(
                c.get("recursion_limit") or _env("DRA_RECURSION_LIMIT") or 4500),
            max_tool_calls=int(
                c.get("max_react_tool_calls")
                or c.get("max_tool_calls")
                or _env("DRA_MAX_TOOL_CALLS")
                or cls.max_tool_calls),
            max_total_tokens=int(
                c.get("max_total_tokens") or _env("DRA_MAX_TOTAL_TOKENS")
                or cls.max_total_tokens),
            max_result_chars=int(
                c.get("max_result_chars") or _env("DRA_MAX_RESULT_CHARS") or 60_000),
            max_result_rows=int(
                c.get("max_result_rows") or _env("DRA_MAX_RESULT_ROWS") or 1000),
            offload_results=(
                bool(c["offload_results"]) if "offload_results" in c
                else _env("DRA_OFFLOAD_RESULTS", default="true").strip().lower()
                not in ("0", "false", "no", "off")
            ),
            offload_dir=c.get("offload_dir") or _env("DRA_OFFLOAD_DIR") or "/workspace/data",
            sandbox_url=(c.get("sandbox_url") or _env("LLM_SANDBOX_URL") or "").rstrip("/"),
            sandbox_token=c.get("sandbox_token") or _env("LLM_SANDBOX_TOKEN") or "",
            sandbox_network=(
                bool(c["sandbox_network"]) if "sandbox_network" in c
                else _env("LLM_SANDBOX_NETWORK", default="false").strip().lower()
                in ("1", "true", "yes", "on")
            ),
            sandbox_session_timeout=int(
                c.get("sandbox_session_timeout") or _env("LLM_SANDBOX_SESSION_TIMEOUT") or 900),
        )
