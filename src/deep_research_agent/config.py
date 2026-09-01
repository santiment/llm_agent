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
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger("deep_research_agent.config")


def _repo_dir(name: str) -> str:
    """Path to a repo-root content dir (``skills/``, ``custom_tools/``) when running
    from a checkout — this file lives at ``src/deep_research_agent/config.py``, so the
    project root is two parents up. When the package is INSTALLED as a dependency,
    ``parents[2]`` is site-packages and no such dir exists — return "" (feature off)
    rather than a garbage path; deployments point ``DRA_SKILLS_DIR`` /
    ``DRA_CUSTOM_TOOLS_DIR`` at their own content explicitly."""
    d = Path(__file__).resolve().parents[2] / name
    return str(d) if d.is_dir() else ""


def _read_prompt_file(path: str) -> str:
    """Contents of an operator-supplied prompt file (``DRA_DOMAIN_PROMPT_FILE``), or
    "" when unset or unreadable — a missing file must degrade to 'no domain prompt',
    never take the server down at import/config time."""
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("cannot read prompt file %s: %s", path, exc)
        return ""

# Hostnames that resolve to cloud-metadata endpoints — never a legitimate target for
# ANY outbound URL (MCP config or model-supplied web_fetch).
_BLOCKED_HOSTNAMES = {"metadata", "metadata.google.internal"}

# Default for the ``streaming_denylist`` field — a module constant so the field default
# and the resolver share ONE list instead of repeating the model substring.
_DEFAULT_STREAMING_DENYLIST = ["deepseek-v4-flash"]

# Named model packages ("price tiers") — the ONLY place models are chosen; callers pick a
# package by name (``model_tier`` / ``DRA_MODEL_TIER``), never a model. See README for what
# each tier is for. Inline $in/$out per 1M tokens, verified 2026-07-30 (they drift);
# tests/test_model_tiering.py parses them and fails if a fleet outprices its planner, if a
# tier undercuts the one below it, or if a slug carries no price.
MODEL_TIERS: dict[str, dict[str, str]] = {
    # mimo-v2.5 is absent from _DEFAULT_STREAMING_DENYLIST, so this default tier's planner
    # streams — add it there if tool_calls come back doubled or dropped.
    "extra-low": {
        "research_model": "xiaomi/mimo-v2.5",                      # $0.14 / $0.28
        "subagent_model": "deepseek/deepseek-v4-flash",            # $0.14 / $0.28
        "utility_model": "qwen/qwen3-30b-a3b-instruct-2507",       # $0.05 / $0.19
    },
    "low": {
        "research_model": "deepseek/deepseek-v4-pro",              # $0.43 / $0.87
        "subagent_model": "deepseek/deepseek-v4-flash",            # $0.14 / $0.28
        "utility_model": "deepseek/deepseek-v4-flash",             # $0.14 / $0.28
    },
    "mid": {
        "research_model": "google/gemini-3.6-flash",               # $1.50 / $7.50
        "subagent_model": "xiaomi/mimo-v2.5",                      # $0.14 / $0.28
        "utility_model": "deepseek/deepseek-v4-flash",             # $0.14 / $0.28
    },
    # Utility is long-context by design: that slot maps/extracts, so context binds, not depth.
    "high": {
        "research_model": "anthropic/claude-sonnet-5",             # $2.00 / $10.00
        "subagent_model": "moonshotai/kimi-k2.6",                  # $0.65 / $2.72
        "utility_model": "google/gemini-3.5-flash-lite",           # $0.30 / $2.50
    },
}

DEFAULT_MODEL_TIER = "extra-low"

# Valid values for reasoning_effort ("" = provider default; "none" = disable thinking).
_REASONING_EFFORTS = frozenset({"", "none", "minimal", "low", "medium", "high"})


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def _pick(c: dict, *keys: str, env: str = "", default: Any = None) -> Any:
    """Resolve ONE field: first ``configurable`` key that was supplied (listed in
    precedence order, so a compat alias can precede the native name), then ``env``,
    then ``default`` — which is always the dataclass field default, so a knob's
    default is written exactly once.

    "Supplied" means not ``None`` and not "" (an unset env var reads as ""). Unlike
    the ``or`` chain this replaces, an explicit ``0`` / ``False`` therefore WINS
    instead of silently falling back — ``max_retries=0`` means retries off."""
    for k in keys:
        v = c.get(k)
        if v is not None and v != "":
            return v
    v = os.environ.get(env) if env else None
    return default if v in (None, "") else v


_FLAG_ON = ("1", "true", "yes", "on")
_FLAG_OFF = ("0", "false", "no", "off")


def _flag(c: dict, *keys: str, env: str = "", default: bool) -> bool:
    """A boolean knob, resolved by ``_pick``. One parse for every flag, so
    ``DRA_STREAMING`` and ``LLM_SANDBOX_NETWORK`` can't disagree on what "no" means.

    Only an explicit on/off spelling flips the value; anything else (a typo like
    ``flase``) keeps the DEFAULT, with a warning. This matters most for a
    security-relevant flag whose default is off — ``LLM_SANDBOX_NETWORK=flase``
    must not silently open the sandbox's network."""
    v = _pick(c, *keys, env=env, default=default)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _FLAG_ON:
            return True
        if s in _FLAG_OFF:
            return False
        log.warning("unrecognized boolean %r for %s — keeping default %s",
                    v, keys[0] if keys else env, default)
        return default
    return bool(v)


def _allowed_base_urls() -> set[str]:
    """Trusted OpenAI-compatible endpoints. The server-side env/default is always
    allowed; operators may add more via ``DRA_ALLOWED_BASE_URLS`` (comma-separated)."""
    allowed = {_env("OPENAI_BASE_URL", default="https://openrouter.ai/api/v1").rstrip("/")}
    for u in _env("DRA_ALLOWED_BASE_URLS").split(","):
        u = u.strip().rstrip("/")
        if u:
            allowed.add(u)
    return allowed


def url_blocked(url: str, *, allow_private: bool) -> str | None:
    """SSRF vetting shared by every outbound-URL consumer. Returns a reason string if
    the URL must be refused, else ``None``. One skeleton (scheme allowlist, host
    presence, cloud-metadata denylist, IP-literal properties) so hardening it — e.g.
    adding a metadata alias — protects every caller at once; only the PRIVATE-address
    policy diverges per caller:

      - ``allow_private=True`` (operator-supplied MCP URLs): loopback/private hosts are
        legitimate (the internal gateway uses them); only link-local / metadata targets
        (169.254.0.0/16, fe80::/10, ``metadata.*``) are blocked.
      - ``allow_private=False`` (MODEL-supplied web_fetch URLs): localhost and any
        non-public IP literal are refused too.

    A DNS name that RESOLVES to a private address is a known residual gap for both
    (no resolve-and-pin here); the network posture is the second fence."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"scheme {parsed.scheme!r} not allowed"
    host = (parsed.hostname or "").lower()
    if not host:
        return "missing host"
    if host in _BLOCKED_HOSTNAMES:
        return f"blocked metadata host {host!r}"
    if not allow_private and host == "localhost":
        return "localhost blocked"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None  # a DNS name (e.g. host.docker.internal) — not an IP literal to vet
    if ip.is_link_local:
        return f"link-local address {host} blocked"
    if not allow_private and (ip.is_private or ip.is_loopback or ip.is_reserved
                              or ip.is_multicast or ip.is_unspecified):
        return f"non-public address {host} blocked"
    return None


def _mcp_url_blocked(url: str) -> str | None:
    """The MCP-config policy: private/loopback allowed (see ``url_blocked``)."""
    return url_blocked(url, allow_private=True)


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
    # chosen) — never settable individually by env or caller. utility_model's first
    # consumer is the extract-subagent (agent.py): the map/extract worker that reads
    # offloaded result files. Future verifier / compaction features share this key.
    subagent_model: str
    utility_model: str
    temperature: float = 0.0
    # OpenRouter unified `reasoning` effort sent with every model call: none/minimal/
    # low/medium/high, or "" for the provider default. Thinking bills as OUTPUT tokens
    # and provider defaults run medium/dynamic, so capped to "low" by default; "none"
    # disables thinking where supported. Unsupported models ignore the parameter.
    # DRA_REASONING_EFFORT.
    reasoning_effort: str = "low"
    # Per-HTTP-request ceiling (seconds) and retry count on every model call. Without an
    # explicit timeout the OpenAI client's default applies to a request that a proxied
    # provider can stall far longer on — one hung call would otherwise pin a research unit
    # (and its concurrency slot) for the rest of the run. Retries cover transient 429/5xx
    # and are what the SDK already does between attempts with backoff; 0 disables them.
    # Override via DRA_REQUEST_TIMEOUT / DRA_MAX_RETRIES.
    request_timeout: float = 180.0
    max_retries: int = 3
    search_max_results: int = 6
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
    # Deployment-specific guidance injected into BOTH system prompts at the <<DOMAIN>>
    # slot (prompts.py wraps it in a labeled DOMAIN CONTEXT block): the domain's
    # analytical dimensions, terminology, example asks, report register. This EXTENDS
    # the base prompt; the engine contracts the middleware enforces (findings format,
    # submit_report protocol, clarification protocol) are not replaceable. Resolution:
    # configurable ``domain_prompt`` -> env ``DRA_DOMAIN_PROMPT`` (inline text) ->
    # env ``DRA_DOMAIN_PROMPT_FILE`` (path, read at config time).
    domain_prompt: str = ""
    # Drop-in directory of deployment-specific tools. Each ``*.py`` file that subclasses
    # ``CustomTool`` — or defines ``build_tools(cfg)`` / ``build_tool(cfg)`` returning
    # LangChain tool(s) — is auto-loaded into the agent's tool list, with no edits to this
    # generic codebase. Absent dir = no custom tools. Point ``DRA_CUSTOM_TOOLS_DIR`` at a
    # deployment's own directory to load tools that live outside this repo.
    custom_tools_dir: str = ""
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
    streaming_denylist: list[str] = field(
        default_factory=lambda: list(_DEFAULT_STREAMING_DENYLIST))
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
    # In-flight context compaction (compaction.py): when an agent's estimated context
    # crosses this many tokens, older messages are summarized on utility_model and
    # replaced with the summary. The knob is ABSOLUTE (not a window fraction) because an
    # OpenRouter slug does not expose its context size; sized for the ~256k-window models
    # the default tiers use, with headroom for the response. 0 disables compaction
    # entirely (a run that outgrows the window then dies on the provider error, as
    # before). DRA_COMPACTION_TOKENS.
    compaction_tokens: int = 100_000
    # Inject Anthropic-style prompt-cache breakpoints (cache_control) into every model
    # request (caching.py). Applied only when base_url is OpenRouter, which forwards the
    # markers to caching providers and strips them elsewhere. DRA_PROMPT_CACHING=false
    # is the kill switch.
    prompt_caching: bool = True
    # Full-page reader tool (tools/fetch.py): web_fetch returns a page's complete
    # readable text so sub-agents can cite substance, not snippets. Oversized pages
    # offload to the sandbox like any big tool result. DRA_WEB_FETCH=false disables.
    web_fetch: bool = True
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
    # Code-execution sandbox sidecar (the separate llm-sandbox service). When sandbox_url is
    # set, the agent's DEFAULT filesystem backend becomes the sandbox and deepagents' `execute`
    # tool is enabled — the model runs REAL shell/python in the container. Empty → in-memory
    # StateBackend (no execution). sandbox_token must match the service's LLM_SANDBOX_TOKEN.
    sandbox_url: str = ""
    sandbox_token: str = ""
    sandbox_network: bool = False
    sandbox_session_timeout: int = 900

    @property
    def is_openrouter(self) -> bool:
        """The ONE provider-detection gate for OpenRouter-only behaviors (cost
        reporting in models.py, cache_control breakpoints in agent.py). A self-hosted
        OpenRouter-compatible gateway whose URL lacks the substring silently loses
        those features — extend here, not at the call sites."""
        return "openrouter" in self.base_url.lower()

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
        # A name is the ONLY model input there is — see the note below on legacy keys.
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

        # Every remaining field resolves through _pick / _flag: configurable key(s) ->
        # env var -> the dataclass default (`cls.<field>`, so no default is written twice).
        # A denylist supplied by the caller is a list; from the env it is comma-separated.
        denylist = _pick(c, "streaming_denylist", env="DRA_STREAMING_DENYLIST",
                         default=_DEFAULT_STREAMING_DENYLIST)
        if isinstance(denylist, str):
            denylist = denylist.split(",")

        reasoning_effort = str(_pick(
            c, "reasoning_effort", env="DRA_REASONING_EFFORT",
            default=cls.reasoning_effort)).strip().lower()
        if reasoning_effort not in _REASONING_EFFORTS:
            log.warning("unknown reasoning_effort %r — using %r (allowed: %s)",
                        reasoning_effort, cls.reasoning_effort,
                        ", ".join(sorted(v or "\"\"" for v in _REASONING_EFFORTS)))
            reasoning_effort = cls.reasoning_effort

        return cls(
            openai_api_key=openai_key,
            base_url=base_url,
            tavily_api_key=tavily_key,
            research_model=research_model,
            report_model=report_model,
            subagent_model=subagent_model,
            utility_model=utility_model,
            temperature=float(_pick(c, "temperature", default=cls.temperature)),
            reasoning_effort=reasoning_effort,
            request_timeout=float(_pick(
                c, "request_timeout", env="DRA_REQUEST_TIMEOUT",
                default=cls.request_timeout)),
            max_retries=max(0, int(_pick(
                c, "max_retries", env="DRA_MAX_RETRIES", default=cls.max_retries))),
            search_max_results=int(_pick(
                c, "search_max_results", default=cls.search_max_results)),
            mcp_max_concurrency=max(1, int(_pick(
                c, "mcp_max_concurrency", env="DRA_MCP_MAX_CONCURRENCY",
                default=cls.mcp_max_concurrency))),
            mcp_rate_limit_max_wait=float(_pick(
                c, "mcp_rate_limit_max_wait", env="DRA_MCP_RATE_LIMIT_MAX_WAIT",
                default=cls.mcp_rate_limit_max_wait)),
            mcp_servers=mcp_servers,
            mcp_prompt=c.get("mcp_prompt") or "",
            domain_prompt=(_pick(c, "domain_prompt", env="DRA_DOMAIN_PROMPT", default="")
                           or _read_prompt_file(_env("DRA_DOMAIN_PROMPT_FILE"))),
            custom_tools_dir=_pick(c, "custom_tools_dir", env="DRA_CUSTOM_TOOLS_DIR",
                                   default=_repo_dir("custom_tools")),
            skills_dir=_pick(c, "skills_dir", env="DRA_SKILLS_DIR",
                             default=_repo_dir("skills")),
            streaming=_flag(c, "streaming", env="DRA_STREAMING", default=cls.streaming),
            streaming_denylist=[s.strip().lower() for s in denylist if str(s).strip()],
            recursion_limit=int(_pick(
                c, "recursion_limit", env="DRA_RECURSION_LIMIT",
                default=cls.recursion_limit)),
            max_tool_calls=int(_pick(
                c, "max_react_tool_calls", "max_tool_calls", env="DRA_MAX_TOOL_CALLS",
                default=cls.max_tool_calls)),
            max_total_tokens=int(_pick(
                c, "max_total_tokens", env="DRA_MAX_TOTAL_TOKENS",
                default=cls.max_total_tokens)),
            compaction_tokens=int(_pick(
                c, "compaction_tokens", env="DRA_COMPACTION_TOKENS",
                default=cls.compaction_tokens)),
            prompt_caching=_flag(c, "prompt_caching", env="DRA_PROMPT_CACHING",
                                 default=cls.prompt_caching),
            web_fetch=_flag(c, "web_fetch", env="DRA_WEB_FETCH",
                            default=cls.web_fetch),
            max_result_chars=int(_pick(
                c, "max_result_chars", env="DRA_MAX_RESULT_CHARS",
                default=cls.max_result_chars)),
            max_result_rows=int(_pick(
                c, "max_result_rows", env="DRA_MAX_RESULT_ROWS",
                default=cls.max_result_rows)),
            offload_results=_flag(c, "offload_results", env="DRA_OFFLOAD_RESULTS",
                                  default=cls.offload_results),
            offload_dir=_pick(c, "offload_dir", env="DRA_OFFLOAD_DIR",
                              default=cls.offload_dir),
            sandbox_url=str(_pick(c, "sandbox_url", env="LLM_SANDBOX_URL",
                                  default=cls.sandbox_url)).rstrip("/"),
            sandbox_token=_pick(c, "sandbox_token", env="LLM_SANDBOX_TOKEN",
                                default=cls.sandbox_token),
            sandbox_network=_flag(c, "sandbox_network", env="LLM_SANDBOX_NETWORK",
                                  default=cls.sandbox_network),
            sandbox_session_timeout=int(_pick(
                c, "sandbox_session_timeout", env="LLM_SANDBOX_SESSION_TIMEOUT",
                default=cls.sandbox_session_timeout)),
        )
