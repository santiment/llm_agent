"""Graph factory. ``langgraph.json`` points here.

``make_graph`` is an async config-factory: LangGraph calls it per run with the
RunnableConfig, so models / API keys / MCP servers come from the request, not
from import-time globals. That is what lets one deployment serve many apps and
many model choices.
"""

from __future__ import annotations

import asyncio
import logging
import os

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend

from .budget import BudgetMiddleware
from .caching import PromptCacheMiddleware
from .citations import ResearchOutputMiddleware
from .clarify_fallback import ClarificationFallbackMiddleware, ClarificationGuardMiddleware
from .compaction import ContextCompactionMiddleware
from .completion import ForceCompletionMiddleware
from .config import ResearchConfig
from .findings_gate import SubagentFindingsMiddleware
from .loop_guard import LoopGuardMiddleware
from .metering import RunMeter, SubagentUsageMiddleware, UsageMeterMiddleware
from .models import build_chat_model
from .prompts import (describe_mcp_sources, extract_prompt, orchestrator_prompt,
                      subagent_prompt)
from .report_gate import ReportQualityGateMiddleware
from .skill_usage import SkillUsageMiddleware
from .events import instrument_tool, result_handling
from .tools.clarify import build_clarify_tool
from .tools.custom import load_custom_tools
from .tools.fetch import build_fetch_tool
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


SANDBOX_SEED_DIR = "/workspace"


def skill_seed_files(skills_dir: str) -> list[tuple[str, bytes]]:
    """Python helpers shipped with skills (``skills/<skill>/*.py``) as ``(container_path,
    bytes)`` pairs, seeded into every sandbox session at ``/workspace/<file>`` (the ``execute``
    cwd) — so a skill's recipes are imported (``import recipes as R``) instead of retyped by the
    model out of its markdown. Underscore-prefixed files stay private. Basenames must be unique
    across skills: a later skill's clash is skipped with a warning, never silently overwritten."""
    out: list[tuple[str, bytes]] = []
    owner: dict[str, str] = {}
    if not skills_dir or not os.path.isdir(skills_dir):
        return out
    for skill in sorted(os.listdir(skills_dir)):
        sdir = os.path.join(skills_dir, skill)
        if not os.path.isdir(sdir):
            continue
        for fn in sorted(os.listdir(sdir)):
            if not fn.endswith(".py") or fn.startswith("_"):
                continue
            if fn in owner:
                log.warning("sandbox seed %s from skill %s clashes with skill %s — skipped",
                            fn, skill, owner[fn])
                continue
            owner[fn] = skill
            with open(os.path.join(sdir, fn), "rb") as f:
                out.append((f"{SANDBOX_SEED_DIR}/{fn}", f.read()))
    return out


async def make_graph(config: dict | None = None):
    cfg = ResearchConfig.from_runnable_config(config)

    # Per-run usage ledger: shared by the MCP tool wrapper (counts calls + raw result sizes
    # across orchestrator AND sub-agents) and UsageMeterMiddleware (reads it at run end).
    meter = RunMeter()

    # Model tiering — smart orchestrator, cheap sub-agents. The orchestrator plans,
    # delegates and synthesizes on research_model; sub-agents run their tool loops on
    # subagent_model. Both come from the run's tier package (MODEL_TIERS), which always
    # names both slots, so there is nothing to fall back to here — a tier that omits the
    # sub-agent slot is resolved back to research_model in config.py, not in this file.
    # Both must be tool-capable. report_model is reserved for a future dedicated
    # synthesis step — using it (often a cheap "nano") for the tool loop makes the
    # agent skip tools and terminate early.
    research_model = build_chat_model(cfg.research_model, cfg)
    # Always a fresh build — never alias the orchestrator's instance on string-equal
    # ids, so future per-tier kwargs (temperature, callbacks) can't be silently shared.
    subagent_model = build_chat_model(cfg.subagent_model, cfg)
    # The flash-class floor. This ONE instance is deliberately shared by its two
    # consumers below (extract-subagent's loop + the compaction summarizer).
    utility_model = build_chat_model(cfg.utility_model, cfg)
    log.info("models: research=%s subagent=%s utility=%s",
             cfg.research_model, cfg.subagent_model, cfg.utility_model)

    tools = []
    search = build_search_tool(cfg, meter)
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
        seeds = skill_seed_files(cfg.skills_dir)
        sandbox = HttpSandboxBackend(
            cfg.sandbox_url, cfg.sandbox_token,
            network=cfg.sandbox_network, session_timeout=cfg.sandbox_session_timeout,
            seed_files=seeds)
        backend = SandboxCompositeBackend(default=sandbox, routes=routes)
        log.info("Code sandbox enabled at %s (execute tool ON; %d skill helper(s) seeded per "
                 "session: %s)", cfg.sandbox_url, len(seeds), ", ".join(p for p, _ in seeds) or "-")
    elif routes:
        backend = CompositeBackend(default=StateBackend(), routes=routes)

    # MCP tools. A result too large for context is OFFLOADED to the sandbox filesystem (the
    # same session as `execute`) instead of being truncated, when a sandbox is present and
    # offloading is enabled. load_mcp_tools tags cfg.mcp_servers with each server's
    # `tool_names`, so build the MCP guidance AFTER loading. An app-supplied mcp_prompt wins.
    offload_sink = sandbox if (sandbox is not None and cfg.offload_results) else None
    tools.extend(await load_mcp_tools(cfg, meter, offload_sink=offload_sink))
    mcp_prompt = cfg.mcp_prompt or describe_mcp_sources(cfg.mcp_servers)

    # Deployment-specific tools dropped into the custom_tools/ dir (no edits to this
    # generic codebase). Same instrumentation as MCP tools, so a large result offloads
    # to the sandbox file the `execute` tool reads back. `tools` goes to the
    # research-subagents only — the orchestrator holds no data tools (see below).
    for custom in load_custom_tools(cfg):
        tools.append(instrument_tool(
            custom, kind="tool", **result_handling(cfg, meter, offload_sink)))

    # Full-page reader: search snippets are ~600 chars, and citing a page on its snippet
    # alone is the failure this closes. Same wrapper as MCP/custom tools, so a huge page
    # offloads to the sandbox instead of flooding a sub-agent's context. Its own
    # semaphore bounds concurrent full-page downloads across the whole sub-agent fleet.
    if cfg.web_fetch:
        tools.append(instrument_tool(
            build_fetch_tool(cfg), kind="tool", semaphore=asyncio.Semaphore(4),
            **result_handling(cfg, meter, offload_sink)))

    # Every loaded data-tool name (search + MCP + custom) — the report scrub/lint layer
    # strips exactly THESE names (plus the get_* fallback) when they leak into a report,
    # so hygiene follows the deployment's real tool naming instead of a hardcoded prefix.
    # Deliberately EXCLUDES the deepagents built-ins (task, execute, read_file, …):
    # they are engine machinery, not data-layer names, and words like "task" or
    # "execute" appear in legitimate report prose all the time.
    data_tool_names = tuple(sorted(t.name for t in tools))

    # Middleware shared by the orchestrator AND every sub-agent — all stateless or
    # message-derived, so single instances are safe across the sub-graphs:
    #   - loop guard: break identical-tool-call loops (nudge, then end);
    #   - compaction: summarize-and-shrink when a context nears the window
    #     (0 = disabled; runs the summary on the cheap utility model);
    #   - prompt cache: cache_control breakpoints, OpenRouter-only (it forwards the
    #     markers to caching providers and strips them elsewhere; a plain OpenAI-compat
    #     gateway might reject the unknown block key instead).
    shared_middleware: list = [
        LoopGuardMiddleware(),
        # Self-disables when trigger_tokens <= 0 — no wiring guard needed here.
        ContextCompactionMiddleware(utility_model, trigger_tokens=cfg.compaction_tokens),
    ]
    if cfg.prompt_caching and cfg.is_openrouter:
        shared_middleware.append(PromptCacheMiddleware())

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
        "system_prompt": subagent_prompt(mcp_prompt, cfg.domain_prompt),
        "tools": tools,
        "model": subagent_model,
        # SubagentUsageMiddleware is what makes the fleet's model usage visible in the
        # run's `usage` event — the orchestrator's state never sees these tokens.
        "middleware": [SubagentFindingsMiddleware(),
                       SubagentUsageMiddleware(meter, "research-subagent",
                                               model=cfg.subagent_model),
                       *shared_middleware],
    }
    if skills:  # give the sub-agent the same routing skill it needs to execute
        subagent_spec["skills"] = skills
    subagents = [subagent_spec]

    # Utility-model consumer: reads offloaded /workspace result files on the cheapest
    # model. Registered only when offloading is live. `tools: []` — no data tools to
    # re-fetch with; deepagents still mounts the filesystem + `execute` built-ins, and
    # ExcludeToolsMiddleware then hides every built-in except `execute`: the extractor
    # works ONLY through bounded Python slices (see tool_filter.py for why).
    if offload_sink is not None:
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
        from deepagents.middleware.subagents import SubAgentMiddleware

        from .tool_filter import EXTRACT_EXCLUDED_TOOLS, ExcludeToolsMiddleware

        extract_spec = {
            "name": "extract-subagent",
            "description": (
                "Reads a RESULT FILE saved under /workspace (an offloaded tool result) "
                "and returns consolidated findings: summarize the topics in text, "
                "classify or extract claims, compute aggregates. Runs on the cheapest "
                "model — use it for ALL reading of offloaded files instead of loading "
                "them yourself. Pass the file path(s), the specific question, and the "
                "source label."
            ),
            "system_prompt": extract_prompt(cfg.domain_prompt),
            "tools": [],
            "model": utility_model,
            "middleware": [SubagentFindingsMiddleware(),
                           SubagentUsageMiddleware(meter, "extract-subagent",
                                                   model=cfg.utility_model),
                           *shared_middleware,
                           ExcludeToolsMiddleware(EXTRACT_EXCLUDED_TOOLS)],
        }
        subagents.append(extract_spec)

        # Offloaded files appear in the research-subagents' context (they make the data
        # calls), so nest a `task` tool there, restricted to extract-subagent. This
        # direct path skips deepagents' default stack — carry our own filesystem stack,
        # then the same findings-gate/metering/shared trio as the top-level spec (the
        # nested run has its OWN isolated state, so without its own usage middleware
        # those tokens would never reach the meter).
        nested_extract_spec = {
            **extract_spec,
            "middleware": [
                # Custom prompt: the default one advertises read_file/grep paging, which
                # the tool filter below takes away — don't teach a workflow it can't run.
                FilesystemMiddleware(
                    backend=backend,
                    system_prompt=("Files live under /workspace. You operate on them ONLY "
                                   "through the `execute` tool (Python), in bounded slices."),
                ),
                PatchToolCallsMiddleware(),
                *extract_spec["middleware"],
            ],
        }
        subagent_spec["middleware"] = [
            SubAgentMiddleware(backend=backend, subagents=[nested_extract_spec]),
            *subagent_spec["middleware"],
        ]
        log.info("extract-subagent enabled on %s (reads offloaded files; "
                 "research-subagents delegate to it via task)", cfg.utility_model)

    # The orchestrator holds NO data tools — it plans, delegates (task), verifies
    # findings, and synthesizes. Gathering lives in sub-agents, so raw data can't
    # enter the expensive orchestrator context at all.
    middleware = [
        # Shared trio first (loop guard / compaction / prompt cache) — compaction must
        # run BEFORE BudgetMiddleware so the budget check each step sees the shrunk
        # transcript plus the compacted_* counters it folds back in (never a half-state).
        *shared_middleware,
        # Hard backstop against runaway runs: cumulative tool-call + token ceilings,
        # soft wrap-up nudge then hard stop.
        BudgetMiddleware(
            max_tool_calls=cfg.max_tool_calls,
            max_total_tokens=cfg.max_total_tokens,
        ),
        ForceCompletionMiddleware(),
        # Bounce a finished report back to the model ONCE if it ships with uncited sources,
        # duplicate source lines, or raw field/tool names — things only the author can fix.
        ReportQualityGateMiddleware(tool_names=data_tool_names),
        ResearchOutputMiddleware(
            max_tool_calls=cfg.max_tool_calls,
            max_total_tokens=cfg.max_total_tokens,
            tool_names=data_tool_names,
            meter=meter,   # run clock → run time on the end status (done and error)
        ),
        SkillUsageMiddleware(),
        # Block request_clarification once research has started (TRIAGE-only), then the
        # fallback that surfaces narrated questions as the clarification card.
        ClarificationGuardMiddleware(),
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
        tools=[build_clarify_tool(), build_submit_report_tool(data_tool_names)],
        system_prompt=orchestrator_prompt(mcp_prompt, cfg.domain_prompt),
        subagents=subagents,
        middleware=middleware,
        skills=skills,
        backend=backend,
    )
    # deepagents bakes recursion_limit=9_999 into the graph; clamp it on the returned
    # runnable (LangGraph merges run configs, last wins) so a stuck loop still terminates.
    # Secondary guard — BudgetMiddleware is the primary cap.
    return agent.with_config({"recursion_limit": cfg.recursion_limit})
