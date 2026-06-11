"""Graph factory. ``langgraph.json`` points here.

``make_graph`` is an async config-factory: LangGraph calls it per run with the
RunnableConfig, so models / API keys / MCP servers come from the request, not
from import-time globals. That is what lets one deployment serve many apps and
many model choices.
"""

from __future__ import annotations

import logging
import os

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend

from .budget import BudgetMiddleware
from .citations import ResearchOutputMiddleware
from .clarify_fallback import ClarificationFallbackMiddleware
from .completion import ForceCompletionMiddleware
from .config import ResearchConfig
from .findings_gate import SubagentFindingsMiddleware
from .metering import RunMeter, UsageMeterMiddleware
from .models import build_chat_model
from .prompts import describe_mcp_sources, orchestrator_prompt, subagent_prompt
from .report_gate import ReportQualityGateMiddleware
from .skill_usage import SkillUsageMiddleware
from .tools.clarify import build_clarify_tool
from .tools.mcp import load_mcp_tools
from .tools.report import build_submit_report_tool
from .tools.search import build_search_tool

log = logging.getLogger("deep_research_agent.agent")

# Virtual path the on-disk skills directory is mounted at. The agent reads skills
# via `read_file("/skills/<name>/SKILL.md")`; everything else stays in the ephemeral
# StateBackend so the agent's own file ops never touch real disk.
SKILLS_MOUNT = "/skills/"


def build_skills(cfg: ResearchConfig) -> tuple[list[str] | None, object | None]:
    """Mount the skills directory read-only. Returns ``(sources, skills_backend)`` where
    ``skills_backend`` is a read-only FilesystemBackend to route under ``/skills/`` (composed
    in ``make_graph``), or ``(None, None)`` when no skills dir exists."""
    skills_dir = cfg.skills_dir
    if not skills_dir or not os.path.isdir(skills_dir):
        return None, None
    log.info("Skills mounted from %s at %s", skills_dir, SKILLS_MOUNT)
    return [SKILLS_MOUNT], FilesystemBackend(root_dir=skills_dir, virtual_mode=True)


async def make_graph(config: dict | None = None):
    cfg = ResearchConfig.from_runnable_config(config)

    # Per-run usage ledger: shared by the MCP tool wrapper (counts calls + raw result sizes
    # across orchestrator AND sub-agents) and UsageMeterMiddleware (reads it at run end).
    meter = RunMeter()

    # Model tiering — smart orchestrator, cheap sub-agents. The orchestrator plans,
    # delegates and synthesizes on research_model; sub-agents run their tool loops on
    # subagent_model (defaults to research_model, so unset = single-model behavior).
    # Both must be tool-capable. report_model is reserved for a future dedicated
    # synthesis step — using it (often a cheap "nano") for the tool loop makes the
    # agent skip tools and terminate early.
    research_model = build_chat_model(cfg.research_model, cfg)
    # Always a fresh build — never alias the orchestrator's instance on string-equal
    # ids, so future per-tier kwargs (temperature, callbacks) can't be silently shared.
    subagent_model = build_chat_model(cfg.subagent_model, cfg)
    log.info("models: research=%s subagent=%s utility=%s (utility unused until a "
             "consumer lands)", cfg.research_model, cfg.subagent_model, cfg.utility_model)

    tools = []
    search = build_search_tool(cfg)
    if search is not None:
        tools.append(search)

    # Skills (read-only) route under /skills/. The DEFAULT backend is the code sandbox when
    # LLM_SANDBOX_URL is set (enables the `execute` tool + real file ops in the container),
    # else the ephemeral in-memory StateBackend (current behavior, no execution). Build the
    # sandbox BEFORE loading MCP tools so the tool wrapper can offload large results into the
    # SAME /workspace session the `execute` tool reads back from.
    skills, skills_fb = build_skills(cfg)
    routes = {SKILLS_MOUNT: skills_fb} if skills_fb is not None else {}
    sandbox = None
    backend = None
    if cfg.sandbox_url:
        from .sandbox import HttpSandboxBackend, SandboxCompositeBackend
        sandbox = HttpSandboxBackend(
            cfg.sandbox_url, cfg.sandbox_token,
            network=cfg.sandbox_network, session_timeout=cfg.sandbox_session_timeout)
        backend = SandboxCompositeBackend(default=sandbox, routes=routes)
        log.info("Code sandbox enabled at %s (execute tool ON)", cfg.sandbox_url)
    elif routes:
        backend = CompositeBackend(default=StateBackend(), routes=routes)

    # MCP tools. A result too large for context is OFFLOADED to the sandbox filesystem (the
    # same session as `execute`) instead of being truncated, when a sandbox is present and
    # offloading is enabled. load_mcp_tools tags cfg.mcp_servers with each server's
    # `tool_names`, so build the MCP guidance AFTER loading. An app-supplied mcp_prompt wins.
    offload_sink = sandbox if (sandbox is not None and cfg.offload_results) else None
    tools.extend(await load_mcp_tools(cfg, meter, offload_sink=offload_sink))
    mcp_prompt = cfg.mcp_prompt or describe_mcp_sources(cfg.mcp_servers)

    # A sub-agent owns ONE UNIT of research (e.g. a single entity / period / segment): it makes
    # ALL the calls that unit needs in its OWN context and returns only consolidated dense
    # findings. So a large scan's raw output stays isolated per unit instead of piling into
    # the orchestrator's context. It runs on the (typically cheaper) subagent_model; the
    # findings gate bounces a malformed handoff back once so unsourced findings from a weak
    # model don't poison the report's citations.
    subagent_spec = {
        "name": "research-subagent",
        "description": (
            "Researches ONE assigned unit end-to-end — e.g. a single entity, reporting period, "
            "or segment — making ALL the web/MCP calls that unit needs, and returns "
            "consolidated dense findings with sources. Spawn one per unit, in parallel."
        ),
        "system_prompt": subagent_prompt(mcp_prompt),
        "tools": tools,
        "model": subagent_model,
        "middleware": [SubagentFindingsMiddleware()],
    }
    if skills:  # give the sub-agent the same routing skill it needs to execute
        subagent_spec["skills"] = skills
    subagents = [subagent_spec]

    # Orchestrator KEEPS the data tools for small/targeted gathering; the prompt steers it to
    # DELEGATE breadth/scale to sub-agents (partitioned by unit), so large scans isolate their
    # raw output in sub-agent contexts rather than overflowing the orchestrator's.
    middleware = [
        # Hard backstop against runaway runs: cumulative tool-call + token ceilings,
        # soft wrap-up nudge then hard stop.
        BudgetMiddleware(
            max_tool_calls=cfg.max_tool_calls,
            max_total_tokens=cfg.max_total_tokens,
        ),
        ForceCompletionMiddleware(),
        # Bounce a finished report back to the model ONCE if it ships with uncited sources,
        # duplicate source lines, or raw field/tool names — things only the author can fix.
        ReportQualityGateMiddleware(),
        ResearchOutputMiddleware(
            max_tool_calls=cfg.max_tool_calls,
            max_total_tokens=cfg.max_total_tokens,
        ),
        SkillUsageMiddleware(),
        ClarificationFallbackMiddleware(),
        # Per-run usage ledger → `usage` event + "RESEARCH USAGE" log line.
        UsageMeterMiddleware(
            meter,
            max_tool_calls=cfg.max_tool_calls,
            max_total_tokens=cfg.max_total_tokens,
            recursion_limit=cfg.recursion_limit,
        ),
    ]
    if sandbox is not None:
        from .sandbox import SandboxCleanupMiddleware
        middleware.append(SandboxCleanupMiddleware(sandbox))

    agent = create_deep_agent(
        model=research_model,
        tools=[*tools, build_clarify_tool(), build_submit_report_tool()],
        system_prompt=orchestrator_prompt(mcp_prompt),
        subagents=subagents,
        middleware=middleware,
        skills=skills,
        backend=backend,
    )
    # deepagents bakes recursion_limit=9_999 into the graph; clamp it on the returned
    # runnable (LangGraph merges run configs, last wins) so a stuck loop still terminates.
    # Secondary guard — BudgetMiddleware is the primary cap.
    return agent.with_config({"recursion_limit": cfg.recursion_limit})
