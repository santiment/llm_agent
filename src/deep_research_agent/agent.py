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

from deepagents import (GeneralPurposeSubagentProfile, HarnessProfile, create_deep_agent,
                        register_harness_profile)
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
from .prompts import (coding_prompt, describe_mcp_sources, extract_prompt,
                      orchestrator_prompt, skills_block, subagent_prompt)
from .report_gate import ReportQualityGateMiddleware
from .skill_usage import SkillUsageMiddleware
from .tool_filter import (CODING_EXCLUDED_TOOLS, EXTRACT_EXCLUDED_TOOLS,
                          ORCHESTRATOR_EXCLUDED_TOOLS, ExcludeToolsMiddleware)
from .triage import TriageRouterMiddleware
from .events import instrument_tool, result_handling
from .tools.clarify import build_clarify_tool
from .tools.custom import load_custom_tools
from .tools.fetch import build_fetch_tool
from .tools.mcp import load_mcp_tools
from .tools.report import build_submit_report_tool
from .tools.search import build_search_tool

log = logging.getLogger("deep_research_agent.agent")

# deepagents auto-adds a `general-purpose` sub-agent (the planner's own toolbox — i.e. NO
# data tools) next to ours, and planners delegate to it: a run spent three `task` calls on
# it and got back "the report has been compiled" with nothing behind it. Provider-wide
# opt-out, merged onto deepagents' built-in openai profile (registrations are additive);
# every model here is a ChatOpenAI over OpenRouter, so "openai" covers the whole fleet.
register_harness_profile(
    "openai",
    HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
)


def _quiet_third_party_loggers() -> None:
    """Drop third-party request chatter to WARNING unless the operator asked for DEBUG.

    httpx (and httpx2, the OpenAI SDK's client) log every request at INFO, and the MCP
    client logs its session handshake. With a fresh streamable-HTTP session per tool
    call that is ~9 lines per MCP call, burying the agent's own RUN END / budget / loop
    lines. langgraph_api sets the root level from LOG_LEVEL before importing the graph,
    so honouring DEBUG here keeps the full trace one env var away."""
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        return
    for name in ("httpx", "httpx2", "mcp.client", "mcp.client.streamable_http"):
        logging.getLogger(name).setLevel(logging.WARNING)


_quiet_third_party_loggers()

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


def describe_skills(skills_dir: str) -> list[tuple[str, str]]:
    """``(name, description)`` per skill from each ``SKILL.md`` front matter — what the
    orchestrator gets instead of the files (it has no file tools). Unreadable front matter
    falls back to the directory name with no description, logged."""
    import yaml  # a langchain-core dependency; imported here to keep the module import cheap

    out: list[tuple[str, str]] = []
    if not skills_dir or not os.path.isdir(skills_dir):
        return out
    for skill in sorted(os.listdir(skills_dir)):
        path = os.path.join(skills_dir, skill, "SKILL.md")
        if not os.path.isfile(path):
            continue
        name, desc = skill, ""
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if text.startswith("---"):
                front = text.split("---", 2)[1]
                meta = yaml.safe_load(front) or {}
                name = str(meta.get("name") or skill)
                desc = " ".join(str(meta.get("description") or "").split())
        except Exception as exc:  # a broken skill must not take the graph down
            log.warning("skill %s: front matter unreadable (%s)", skill, exc)
        out.append((name, desc))
    return out


def skill_seed_files(skills_dir: str) -> list[tuple[str, bytes]]:
    """``skills/<skill>/*.py`` as ``(container_path, bytes)`` pairs seeded into every sandbox
    session at ``/workspace/<file>``. Underscore-prefixed files stay private; a basename
    clash between skills is skipped with a warning."""
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
    # The extract-subagent's model (map/extract over offloaded files).
    utility_model = build_chat_model(cfg.utility_model, cfg)
    # The compaction summarizer's model — rare, input-heavy, quality over depth.
    compaction_model = build_chat_model(cfg.compaction_model, cfg)
    # The coding-subagent's model: a dedicated coder on a small input (below).
    coding_model = build_chat_model(cfg.coding_model, cfg)
    log.info("models: research=%s subagent=%s utility=%s compaction=%s coding=%s",
             cfg.research_model, cfg.subagent_model, cfg.utility_model,
             cfg.compaction_model, cfg.coding_model)

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

    # Full-page reader; its own semaphore bounds concurrent downloads across the fleet.
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

    # Shared by the orchestrator and every sub-agent (all stateless). Compaction
    # self-disables at trigger_tokens <= 0; cache breakpoints are OpenRouter-only.
    shared_middleware: list = [
        LoopGuardMiddleware(),
        ContextCompactionMiddleware(compaction_model, trigger_tokens=cfg.compaction_tokens),
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
        # SubagentUsageMiddleware: the orchestrator's state never sees sub-agent tokens.
        "middleware": [SubagentFindingsMiddleware(),
                       SubagentUsageMiddleware(meter, "research-subagent",
                                               model=cfg.subagent_model),
                       SkillUsageMiddleware(),  # the "Skill applied" chip fires from here
                       *shared_middleware],
    }
    # Skills live with the sub-agents ONLY: they hold the file tools that load a SKILL.md.
    # The orchestrator gets names + descriptions in its prompt (describe_skills) and names
    # the skill in a brief — never the 4k-token file in its own context on every run.
    if skills:
        subagent_spec["skills"] = skills
    subagents = [subagent_spec]

    # Sandbox-only sub-agents — both work through `execute`, so neither exists without a
    # sandbox. `tools: []` — no data tools to re-fetch with; deepagents still mounts the
    # filesystem + `execute` built-ins, and ExcludeToolsMiddleware hides what each role must
    # not use (see tool_filter.py). Both are ALSO nested into the research-subagent via its
    # own `task` tool, since offloaded files and failing scripts appear in ITS context. That
    # direct path skips deepagents' default stack, so a nested spec carries its own
    # filesystem stack, then the same middleware as the top-level spec.
    nested_specs: list[dict] = []
    if sandbox is not None:
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
        from deepagents.middleware.subagents import SubAgentMiddleware


        def nested(spec: dict, files_prompt: str) -> dict:
            # The default filesystem prompt advertises read_file/grep, which the tool filter
            # may remove — so each nested spec states its own file rules.
            return {
                **spec,
                "middleware": [
                    FilesystemMiddleware(backend=backend, system_prompt=files_prompt),
                    PatchToolCallsMiddleware(),
                    *spec["middleware"],
                ],
            }

    # Utility-model consumer: reads offloaded /workspace result files on the cheapest
    # model. Registered only when offloading is live; the filter hides every built-in
    # except `execute`.
    if offload_sink is not None:
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
        nested_specs.append(nested(
            extract_spec,
            f"Files live under {SANDBOX_SEED_DIR}. You operate on them ONLY through the "
            "`execute` tool (Python), in bounded slices."))
        log.info("extract-subagent enabled on %s (reads offloaded files; "
                 "research-subagents delegate to it via task)", cfg.utility_model)

    # Coding worker: writes or fixes ONE script per task on a dedicated coder, so a failed
    # `execute` is repaired by a model good at it instead of by the planner narrating
    # shell-quoting retries. Small input (goal, file paths, code, error), so its price
    # barely registers. No findings gate — its handoff is a script path + output, not
    # findings. Keeps the file tools (it writes and edits scripts); loses grep/write_todos.
    if sandbox is not None:
        coding_spec = {
            "name": "coding-subagent",
            "description": (
                "Writes or fixes a Python script for you on a dedicated coding model. Give it "
                "the goal, the input file paths under /workspace, any existing code and the "
                "EXACT error text; it returns the script path and the script's real output. "
                "Use it for any script longer than a few lines and after ANY failed `execute` "
                "— never retry shell-quoting variants of `python -c` yourself."
            ),
            "system_prompt": coding_prompt(),
            "tools": [],
            "model": coding_model,
            "middleware": [SubagentUsageMiddleware(meter, "coding-subagent",
                                                   model=cfg.coding_model),
                           *shared_middleware,
                           ExcludeToolsMiddleware(CODING_EXCLUDED_TOOLS)],
        }
        subagents.append(coding_spec)
        nested_specs.append(nested(
            coding_spec,
            f"Scripts and data files live under {SANDBOX_SEED_DIR}: `write_file` a script "
            f"there, run it with `execute` (`python {SANDBOX_SEED_DIR}/<name>.py`), fix it "
            "with `edit_file`."))
        subagent_spec["middleware"] = [
            SubAgentMiddleware(backend=backend, subagents=nested_specs),
            *subagent_spec["middleware"],
        ]
        log.info("coding-subagent enabled on %s (writes/fixes scripts for the orchestrator "
                 "and the research-subagents)", cfg.coding_model)

    # The orchestrator holds NO data tools — it plans, delegates (task), verifies
    # findings, and synthesizes. Gathering lives in sub-agents, so raw data can't
    # enter the expensive orchestrator context at all.
    middleware = [
        # First: a ~250-token verdict on the research model. A pure-knowledge question is
        # answered right here and the turn ends, never paying the ~12k-token harness below.
        TriageRouterMiddleware(research_model),
        # Compaction must run before BudgetMiddleware so the budget check sees the shrunk
        # transcript plus the compacted_* counters.
        *shared_middleware,
        # Hard backstop against runaway runs: cumulative tool-call + token ceilings,
        # soft wrap-up nudge then hard stop.
        BudgetMiddleware(
            max_tool_calls=cfg.max_tool_calls,
            max_total_tokens=cfg.max_total_tokens,
            max_run_seconds=cfg.max_run_seconds,
            meter=meter,
        ),
        ForceCompletionMiddleware(),
        # Bounce a finished report back to the model ONCE if it ships with uncited sources,
        # duplicate source lines, or raw field/tool names — things only the author can fix.
        ReportQualityGateMiddleware(tool_names=data_tool_names),
        ResearchOutputMiddleware(
            max_tool_calls=cfg.max_tool_calls,
            max_total_tokens=cfg.max_total_tokens,
            tool_names=data_tool_names,
            meter=meter,
            max_run_seconds=cfg.max_run_seconds,
        ),
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
            model=cfg.research_model,
        ),
        # The orchestrator never touches files: no ls/read/write/edit/glob/grep in its
        # toolbox (nor their descriptions and schemas on every step). `execute` stays for
        # the odd one-line aggregate; `task` is how files get read or written.
        ExcludeToolsMiddleware(ORCHESTRATOR_EXCLUDED_TOOLS),
    ]
    if sandbox is not None:
        from .sandbox import SandboxCleanupMiddleware
        middleware.append(SandboxCleanupMiddleware(sandbox))

    agent = create_deep_agent(
        model=research_model,
        tools=[build_clarify_tool(), build_submit_report_tool(data_tool_names)],
        system_prompt=orchestrator_prompt(
            mcp_prompt, cfg.domain_prompt, skills_block(describe_skills(cfg.skills_dir))),
        subagents=subagents,
        middleware=middleware,
        skills=None,  # sub-agents only — see subagent_spec
        backend=backend,
    )
    # deepagents bakes recursion_limit=9_999 into the graph; clamp it on the returned
    # runnable (LangGraph merges run configs, last wins) so a stuck loop still terminates.
    # Secondary guard — BudgetMiddleware is the primary cap.
    return agent.with_config({"recursion_limit": cfg.recursion_limit})
