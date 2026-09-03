# How the Deep Research Agent Works

*A guide for the team — from a user's question to a finished, sourced research report.*

This document explains the agent end-to-end: what happens when a question comes in, how the work
is planned and delegated, how data is gathered and turned into analysis, and the guardrails that
keep a run honest. It is written to be read top-to-bottom by anyone on the team, but each section
also stands alone.

---

## 1. What this thing is, in one paragraph

The Deep Research Agent takes a natural-language question (e.g. *"Give me a deep research report on
XYZ"* or *"is the crowd long or short on $X right now?"*) and produces a thorough, **inline-cited
markdown report** — in the spirit of Gemini Deep Research or Claude's research mode. It plans an
investigation, splits it into independent units, fans those out to cheaper worker agents that each
gather and distill their slice, then synthesizes everything into one report. Along the way it
streams a live "thinking" view (searches, tool calls, sources, findings) so a frontend can show
progress. It is **model-agnostic** (any OpenAI-compatible model via OpenRouter), **portable**
(no host-app imports in the core), and wrapped in a layer of deterministic guardrails so weak/cheap
models still finish with a real, well-formed answer.

It is built on two libraries:

- **`deepagents`** — gives us the orchestrator + sub-agent pattern, the built-in `write_todos`,
  `task`, `read_file`/`execute` tools, the filesystem backends, and the middleware hook system.
- **`LangGraph`** — the runtime. The agent is a graph; it's served over LangGraph's HTTP/SSE API,
  and we stream typed events on its `custom` channel.

---

## 2. The big picture

```
                      ┌──────────────────────────────────────────────────────────────┐
   user question      │                      ONE RESEARCH RUN                         │
  ───────────────►    │                                                              │
  (LangGraph thread)  │   make_graph(config)  ──build──►  ORCHESTRATOR (smart model) │
                      │        │                                  │                   │
                      │        │ reads ResearchConfig             │ ReAct loop:       │
                      │        │ (per-run: keys, model tier,      │  triage → plan →  │
                      │        │  MCP servers, sandbox, skills)   │  gather → synth   │
                      │        ▼                                  │                   │
                      │   builds tools + middleware               │                  │
                      │                                           ▼                   │
                      │             delegates UNITS via `task`  ──────►  research-     │
                      │             (partition by entity/dim/      ◄──── subagents    │
                      │              period, run in PARALLEL)     findings  (cheap     │
                      │                                           JSON      model,     │
                      │   web_search │ MCP tools │ custom tools │ execute   isolated   │
                      │   (Tavily)   │ (Santiment…)│(social_msgs)│(sandbox) context)   │
                      │                                           │                   │
                      │                          synthesize ──► submit_report(md)     │
                      └───────────────────────────────┬──────────────────────────────┘
                                                       │
   live stream  ◄───── run_start / search / tool_call / source / skill / findings / report / usage
   (custom events)            (what the UI renders as the "research process" + final card)
```

The orchestrator's own context is the **scarce, expensive resource**. The central design idea is:
*keep raw data out of the orchestrator's context.* Heavy gathering is pushed down to sub-agents
(isolated contexts, cheaper model) and large tool results are offloaded to files — so the smart,
expensive model spends its tokens on planning and synthesis, not on re-reading thousands of rows.

---

## 3. Entry point and per-run configuration

### `make_graph` — built fresh for every run

`langgraph.json` points the deployment at `deep_research_agent.agent:make_graph`. This is an **async
config-factory**: LangGraph calls it *per run* with that run's `RunnableConfig`. That is what lets
one deployment serve many apps and many model choices — models, API keys, MCP servers, sandbox, and
skills all come from the request, not from import-time globals.

`make_graph` (`src/deep_research_agent/agent.py`) does, in order:

1. Resolve `ResearchConfig` from the run config.
2. Create a per-run `RunMeter` (usage ledger shared by tool wrappers and the usage middleware).
3. Build the **research model** (orchestrator) and **sub-agent model** from the chosen tier.
4. Build the tool list: web search, MCP tools, custom tools.
5. Mount skills read-only; wire the sandbox (if configured) as the default filesystem backend.
6. Define the `research-subagent` spec (its own prompt, model, tools, findings middleware).
7. Assemble the orchestrator's middleware stack (the guardrails — see §10).
8. Call `create_deep_agent(...)` and clamp the recursion limit.

### `ResearchConfig` — the portability seam

`config.py` resolves **every** setting with the precedence:

> per-run `configurable` override  →  environment variable  →  default

It also accepts compatibility aliases (`apiKeys`, `mcp_config`, `mcp_prompt`, `max_react_tool_calls`,
…) so an existing caller can adopt the agent with **zero backend changes**. Security is baked in
here: the `base_url` override is allow-listed (a hostile base URL would otherwise receive our API
key as a bearer token), and MCP URLs are vetted against SSRF (link-local / cloud-metadata hosts are
refused; `0.0.0.0` is rewritten to loopback).

### Model tiers (the only place models are chosen)

Models are **not** individually settable per run or per env var. You pick a **named package** via
`model_tier` / `DRA_MODEL_TIER`; the models behind each name live in code (`MODEL_TIERS`), in one
reviewed place. Every tier defines three roles:

| Tier        | research (orchestrator) | subagent (workers)    | utility (extract)  | Use for |
|-------------|-------------------------|-----------------------|--------------------|---------|
| `extra-low` | mimo-v2.5               | deepseek-v4-flash     | qwen3-30b          | demos, smoke tests, high-volume low-stakes |
| `low`       | deepseek-v4-pro         | deepseek-v4-flash     | deepseek-v4-flash  | cheapest sane agent |
| `mid`       | gemini-3.6-flash        | mimo-v2.5             | deepseek-v4-flash  | the value sweet spot |
| `high`      | claude-sonnet-5         | kimi-k2.6             | gemini-3.5-flash-lite | best quality per dollar |

The default is `extra-low` so a bare checkout can't silently burn money — production callers opt
**up** explicitly. The core idea of **model tiering**: a strong orchestrator *plans and
synthesizes*; cheaper workers *grind the data*. Handing the top model to the sub-agent fleet — which
makes most of the tool calls — would defeat the point, so that is an asserted invariant rather than
an intention: `tests/test_model_tiering.py` parses the `# $in / $out` comment beside every slug in
`MODEL_TIERS` and fails if a fleet is priced above its planner, if a tier is cheaper than the one
below it, or if a slug carries no price at all.
(The `utility` slot powers the `extract-subagent`: when a large tool result is offloaded to a
/workspace file, the file + a question goes to this cheapest model for reading/summarizing
instead of being loaded into a more expensive context. Research-subagents — where the offloaded
files actually appear, since they make the data calls — carry a nested `task` tool restricted to
extract-subagent for exactly this delegation; the orchestrator can also call it for files
surfaced in findings. It is registered only when a sandbox is attached — without one there are
no offloaded files to read.)

---

## 4. Orchestrator + sub-agents: the core architecture

`create_deep_agent` gives us a **ReAct loop** (the model thinks, calls a tool, reads the result,
repeats) with one orchestrator and a fleet of sub-agents.

- **The orchestrator** runs on the smart model. It triages, plans (`write_todos`), delegates,
  verifies the returned findings against its plan, and writes the final report. It holds **no data
  tools** — every gather, even a one-call lookup, goes through `task`, so raw data cannot enter the
  expensive context at all.
- **A `research-subagent`** owns **one unit** of research — a single entity, dimension, period, or
  segment. It makes *all* the calls that unit needs **in its own context** and returns only
  consolidated, dense findings. This is context isolation: a large scan's raw output stays trapped in
  the sub-agent's context instead of piling into the orchestrator's.

The orchestrator spawns sub-agents with the built-in **`task`** tool, one per unit, **in parallel**.
A sub-agent's reply comes back as a structured findings object (see §7) that the orchestrator *reads*
— pulling out the summary, the findings, and each finding's source for its `[n]` citations.

---

## 5. The lifecycle of a request (the four-step workflow)

This is the heart of the system — encoded in the orchestrator's system prompt (`prompts.py`,
`ORCHESTRATOR_PROMPT`). Every turn runs through it.

### Step 0 — TRIAGE (every turn, first)

The model classifies the incoming message:

- **SIMPLE** — a greeting, small talk, or a factual question it can answer reliably from its own
  knowledge (*"what's the capital of Bulgaria?"*, *"what does MVRV mean?"*). → answer in one or two
  sentences as plain text and **stop**. No tools, no report. This re-runs on **every** message, so a
  definitional follow-up after a big report is still answered as a quick plain reply.
- **AMBIGUOUS** — unclear scope, timeframe, entity, or goal. → call `request_clarification` with 1–3
  short questions, then **stop and wait**. This is allowed **only here, before any research**, at
  most twice. (See §9 on how the UI renders this and how the user's reply flows back.)
- **NEEDS RESEARCH** — needs current data, sources, or multi-step analysis. → continue to step 1.

### Step 1 — PLAN

The orchestrator uses **`write_todos`** to lay out the investigation as a short list of named angles
("Structuring the investigation", "Mapping the data landscape", "Next steps"). As it works it
narrates brief reasoning in prose — *this narration is streamed to the user as the live "thinking"
view*. Rule: no `#` markdown headings and no conclusions/Sources mid-research — those belong only in
the final report.

### Step 2 — GATHER (delegate; don't grind raw data yourself)

The orchestrator **partitions** the work into independent units and spawns one sub-agent per unit in
parallel. A "unit" is any slice researchable on its own:

- an analytical **dimension** — *"Analyze Bitcoin"* → one sub-agent each for price/market action,
  on-chain activity, social/sentiment, developer activity, tokenomics/supply;
- an **entity** — one asset per sub-agent when comparing several;
- a **period or segment** — one reporting period or category per sub-agent.

Each sub-agent gets its whole slice, makes all the calls, computes aggregates (in the sandbox when
available), and returns consolidated findings. The orchestrator only delegates-to-itself (calls a
tool directly) for a genuinely tiny ask — a single metric, a one-line lookup. Anything phrased as
"analyze / assess / deep dive / compare / research" gets delegated. **When unsure, delegate.**

### Step 3 — SYNTHESIZE

The orchestrator combines all sub-agent findings (plus anything it gathered directly) into **one**
comprehensive markdown report and delivers it with **`submit_report`**. That tool call — not
"whatever the last chat message was" — *is* the deliverable.

---

## 6. The tools

Tools are how the model touches the world. Both the orchestrator and sub-agents share the same data
tools; each discovers a tool by its name and description.

| Tool | Source file | What it does |
|------|-------------|--------------|
| `web_search` | `tools/search.py` | Tavily web search. Emits `search_query` + `search_results` events; returns **numbered** sources `[n]` for inline citation. |
| `task` | deepagents built-in | Delegate a unit of research to a `research-subagent`. The breadth/scale lever. |
| MCP tools | `tools/mcp.py` | Tools from configured MCP servers (e.g. the Santiment connector: `fetch_metric_data`, `show_chart`, …). Loaded per-server so each tool is attributed to a friendly **source label** for citations. |
| Custom tools | `tools/custom.py` | Deployment-specific tools dropped into the `custom_tools/` dir (e.g. `social_messages`), or into whatever `DRA_CUSTOM_TOOLS_DIR` points at. Auto-loaded, no edits to the core. |
| `execute` | sandbox (`sandbox.py`) | Runs **real** Python/shell in a sandbox container (the image ships pandas/numpy). Enabled only when `LLM_SANDBOX_URL` is set. Used to compute aggregates and to process offloaded result files. |
| `write_todos` | deepagents built-in | The plan/progress list (step 1). |
| `request_clarification` | `tools/clarify.py` | Ask the user 1–3 questions up front (triage only). Emits a `clarification` card and stops. |
| `submit_report` | `tools/report.py` | Deliver the final report. Called exactly once. **The only way to deliver an answer.** |

A few important behaviors:

- **A failed tool call never kills the run.** The error text is returned to the model *as the tool
  result*, classified `permanent` / `transient` / `unknown`, with guidance on how to proceed
  (`events.py`). A permanent failure for given arguments is remembered, so an identical retry is
  answered locally instead of hammering the server.
- **MCP concurrency is capped.** `langchain-mcp-adapters` opens a new connection per call, so a
  single shared semaphore (`mcp_max_concurrency`, default 10) bounds simultaneous MCP calls across
  the orchestrator *and* all parallel sub-agents. Rate-limit (429) responses trigger bounded backoff
  rather than failure.
- **No chart event (yet).** A chart-shaped MCP result (e.g. the connector's `show_chart`) streams
  as an ordinary `tool_result` like any other; there is no structural detection and no `chart`
  event in `EVENT_SCHEMAS`. A frontend that wants to draw one reads the tool result itself.

---

## 7. Sub-agents and the findings contract

A sub-agent's **final message is the only thing the orchestrator ever sees** of its work — the raw
tool output stays in the sub-agent's context. So that handoff must be checkable, especially when
sub-agents run on a cheaper model that tends to economize on attribution.

The contract (`prompts.FINDINGS_FORMAT`) requires the final message to be **exactly one JSON
object**:

```json
{"summary": "<dense prose digest — figures, dates, named entities>",
 "findings": [{"finding": "<one specific claim with its numbers>",
               "evidence": "<the data behind it>",
               "source": "<URL for web; the EXACT internal source label for data tools>"}],
 "gaps": ["<what couldn't be determined, and why>"]}
```

`SubagentFindingsMiddleware` (`findings_gate.py`) enforces this **deterministically — no model in the
loop**:

- The object must parse and have the right shape; **every finding must carry a source**.
- **Provenance check:** non-empty findings with *zero tool calls this run* means the model answered
  from memory — bounced.
- Empty findings are explicitly allowed (an honest "nothing found" beats a fabricated one).
- On a violation it bounces the work back to the sub-agent **once** with the specific problems, then
  accepts whatever comes back (prose degrades gracefully — the orchestrator can still read it; the
  gate must never fail a run over formatting).
- Valid findings are emitted as a `subagent_findings` event so the UI can render a folded table.

The orchestrator reuses each finding's `source` for its inline `[n]` citations and spawns follow-up
sub-agents for any non-empty `gaps`.

---

## 8. Handling scale: large results, offload, and the sandbox

Many medium-sized results, each under any single eviction threshold, used to silently pile up until
the context blew past the model's limit. Two mechanisms prevent that:

- **Per-call size cap.** Any tool result over `max_result_chars` (60k) or `max_result_rows` (1000) is
  considered too big for context.
- **Offload instead of truncate.** When a sandbox is configured, a too-big result is **written to a
  file** under `/workspace/data` and only a compact **stub** enters context — the path, row count,
  column list, and a small preview, plus an instruction to process the file with `execute`. The model
  then loads the full data with pandas in the sandbox and computes aggregates/joins/filters
  *there*. This is exactly how a large cross-entity sweep is handled: the raw rows never enter the
  model's context at all. Without a sandbox, the same thresholds become hard **truncation** caps.
- **MCP results are unwrapped first.** langchain-mcp-adapters hands back every MCP result as LangChain content blocks (`[{type: text, text: <the JSON>}]`), so the series/size checks would otherwise inspect the envelope, not the data — a metric series buried in a text block stayed inline and the model hand-transcribed it into a file. `events.unwrap_tool_result` flattens a text-only result to its JSON before the offload decision, so an MCP series offloads exactly like a string-returning custom tool. Results carrying an image/file block are left intact.
- **Time series are offloaded whatever their size.** `series.py` recognizes a metric series in any
  tool result (the metric server's `{data: {slug: [{datetime, value}]}}`, a bare list of points,
  `[ts, value]` pairs — 8+ points). A series result is written to a file even when it is small, and
  the stub carries a **computed summary** — points, span, first/last value with change, min/max with
  when, mean, median, direction — plus the rule that a series is never listed row by row. Without a
  sandbox the summary and the rule are prepended and the rows stay. This is the source-side half of
  the "never paste a metric table" rule; the delivery-side half is in §9: `report_hygiene.py`
  detects runs of timestamped rows (any date format, blank lines tolerated), the quality gate
  bounces them once, and `collapse_series` deletes any that remain on every emit path, leaving a
  one-line note with the same statistics. The skill (`R.describe`) gives the same numbers in the
  sandbox.

The sandbox itself (`sandbox.py`) is an HTTP client to an `llm-sandbox` sidecar service. Wiring it in
makes it the agent's **default filesystem backend**, which is what flips deepagents' `execute` tool
on. A session is created lazily, reused for the run, and destroyed at the end by
`SandboxCleanupMiddleware`. The skills directory is routed separately (read-only) so the agent's own
file ops never touch real disk.

Skill helper modules are **seeded** into every session: right after the session is created, the
backend uploads each `skills/<skill>/*.py` (public names only; first skill wins a basename clash) to
`/workspace/<file>`, before any `execute` can run. A skill's markdown then says `import recipes as R`
instead of carrying code the model would have to retype — the computation is tested Python, the
skill text is only about how to read its output. A failed seed is logged, not fatal; the skill's
fallback is to `read_file` the module from `/skills/` and `write_file` it into `/workspace`.

The prompt is strict here: **run code for real or not at all.** The model must never invent or
simulate program output — only show an "output" block when it is the verbatim result of a real
`execute` call.

---

## 9. Skills (progressive disclosure)

Skills are drop-in capabilities — a folder under `skills/` with a `SKILL.md` (YAML frontmatter +
markdown instructions) and optional supporting files. On each run the agent mounts the directory
read-only at the virtual path `/skills/` and injects each skill's **name + description** into the
system prompt. The model reads a skill's *full* instructions (via `read_file` on
`/skills/<name>/SKILL.md`) **only when a task matches its description** — that's progressive
disclosure: cheap to advertise, loaded on demand. A skill may also ship Python helpers next to its
`SKILL.md`; those are seeded into the sandbox as `/workspace/<file>` (section 8) so the instructions
call them rather than embed them.

The shipped example, **`crowd-positioning`**, turns raw `social_messages` data into a positioning
verdict (extremeness percentiles, organic-vs-manufactured, narrative-vs-chain divergence, crowd price
levels). Its `SKILL.md` is a worked example of the pattern: a tight "when to use" trigger, a precise
workflow that names exact tools and computations, and a strict output format. Its numeric
recipes ship as `recipes.py` beside the `SKILL.md`, seeded into the sandbox so the model calls them.

When the model reads a skill file, `SkillUsageMiddleware` emits a `skill` event so the UI can show a
"Skill applied: …" indicator.

---

## 10. The guardrails (middleware): the safety net

This is what makes cheap/weak models usable. `create_deep_agent` runs a stack of **middleware**, each
hooking into a point of the loop. They are deterministic — they don't ask a model to fix things they
can decide themselves. Hook points:

- `before_model` — runs before each model call.
- `after_model` — runs after the model responds (can send the loop back to the model).
- `awrap_tool_call` — wraps a specific tool call (can intercept/replace it).
- `after_agent` — runs once when the run ends.

The orchestrator's stack (assembled in `agent.py`, in this order):

| Middleware | Hook | Job |
|------------|------|-----|
| **BudgetMiddleware** | `before_model` | Hard backstop against runaway runs. Two ceilings (cumulative tool calls, cumulative tokens). At **75%** it injects one "wrap up and deliver now" nudge; at **100%** it jumps straight to `end`. |
| **ForceCompletionMiddleware** | `after_model` | Prevents premature termination. If the model stops with a bare *"Now I will compare…"* intent message and no tool call mid-research, it nudges the model to act (capped). If the model wrote the whole report as a plain message, one mechanical "resubmit via `submit_report` verbatim" nudge; a raw JSON blob gets a "rewrite as a real report" nudge instead. |
| **ReportQualityGateMiddleware** | `awrap_tool_call` | Intercepts `submit_report` **before** delivery. If the (scrubbed) report still violates the contract — uncited sources, an internal source split across many Sources lines, raw field names, file paths / file names / "offloaded files" / code calls — it bounces it back **once** with specific fixes. Then delivers as-is. |
| **ResearchOutputMiddleware** | `after_agent` | Harvests sources, persists the final report into state, and authoritatively classifies *why the run ended* (see §13). Also the salvage path: if the model researched but never called `submit_report`, it recovers genuine report-looking prose (never a JSON blob, never a bare intent stub). |
| **SkillUsageMiddleware** | `after_model` | Emits a `skill` event the first time each skill is read in a turn. |
| **ClarificationGuardMiddleware** | `awrap_tool_call` | Blocks `request_clarification` *after* research has begun — so a weak model can't pop a nonsensical question card after minutes of work. Tells it to finish instead. |
| **ClarificationFallbackMiddleware** | `after_model` | If the model *narrates* clarifying questions as plain text (pre-research) instead of calling the tool, this emits the `clarification` card anyway — so the UI behaves the same regardless of model. |
| **UsageMeterMiddleware** | `before_agent` / `after_agent` | Starts the run clock and emits `run_start` (`started_at`); at the end emits the per-run `usage` event and the `RESEARCH USAGE` log line (tool calls, errors, rows/bytes, tokens, model calls, run time). `ResearchOutputMiddleware` reads the same clock, so the end `status` carries the run time in the success and the no-report case alike. |
| **SandboxCleanupMiddleware** | `after_agent` | Destroys the run's sandbox session (only present when a sandbox is configured). |

**Turn-scoping** underpins all of this (`turn.py`). A LangGraph thread accumulates *every* message
across multi-turn chat, so every "did we deliver a report this turn?" check looks only at messages
from the most recent genuine user message onward — otherwise a follow-up would inherit the previous
turn's report. Synthetic nudge messages are tagged so they're never mistaken for the start of a new
turn, which also makes the per-turn nudge caps self-reset.

`SubagentFindingsMiddleware` (§7) is the same idea pointed at the *other* deliverable — it's attached
to the sub-agent, not the orchestrator.

---

## 11. The streaming event protocol (what a frontend renders)

The agent emits typed JSON events on LangGraph's `custom` stream channel (`events.py`). This is **the
contract any frontend renders** — the agent core is the only producer; your app is just a consumer.
That's what keeps the agent portable.

| Event | Renders as |
|-------|-----------|
| `run_start` | nothing visible — the protocol handshake (`protocol_version`, `engine_version`) a frontend pins against before rendering the rest; `started_at` (UTC) is the anchor for run time when a run dies before its end events |
| `search_query` | the globe row ("how to analyze key metrics") |
| `search_results` | the favicon + title grid ("7 results") |
| `tool_call` / `tool_result` (and `mcp_call` / `mcp_result`) | tool/MCP call rows |
| `source` | a registered citation for the live source list |
| `skill` | "Skill applied: …" |
| `subagent_findings` | a folded findings table from a worker |
| `clarification` | the question card (re-enables input) |
| `status` | lifecycle: `mcp_ready` / `mcp_error` (tool loading), `budget_soft` / `budget_halt` (ceilings), `revising` (a gate bounced a deliverable back), `compacting` / `compacted` (context compaction), `loop_detected` / `loop_halt` (repeated-identical-call guard), `subagent_start` / `subagent_done` (a sub-agent run, with `role` + `model`), then exactly one end-state — `done` or `error`, with a `reason` code and the run time (`elapsed_s` / `elapsed`, also appended to `detail`: "… Run time 4m 12s.") |
| `usage` | the per-run usage summary, incl. run time (`elapsed_s`, `elapsed`, `started_at`, `finished_at`) |
| `report` | the final markdown answer (also persisted in state) |

The protocol is **pinned in code, not just documented**: `events.EVENT_SCHEMAS` registers every type's
required keys, `emit()` warns (never raises) when a payload drifts from its schema, and the test suite
asserts that every emit site in `src/` matches a registered schema. `run_start` carries
`protocol_version` (bumped only on breaking shape changes — additive keys and new event types do not
bump it), so a consumer pins the version once at the handshake instead of failing mid-render.

The assistant's **reasoning prose** (the italic narration between steps) is *not* a custom event — it
streams as normal AI tokens on the `messages` channel, which the UI puts in the "show thinking
process" pane.

`examples/client.py` is a minimal consumer that shows exactly how an app subscribes to these events
— no package import needed, just HTTP/SSE against the LangGraph server.

### How clarification round-trips

When `request_clarification` (or the fallback) fires, the UI shows a question card and re-enables
input. The user's reply arrives as the **next message on the same thread**, so the agent now has the
Q&A in context and proceeds straight to research — no special plumbing.

---

## 12. Citations and report hygiene

Citations are interleaved like Claude's research mode: claims are cited inline with `[n]`, and the
report ends with a `## Sources` list, one source per line, the bracket numbers matching.

- **Web sources** get one line per URL (only URLs that actually appeared in tool results — never
  invented).
- **Internal data sources** (MCP/custom data tools — they have no URL) get **one line**, named
  exactly as the source label, grouping all their `[n]` numbers (e.g. `- [1][2][5] Santiment …`).

Two deterministic last-mile helpers (`report_hygiene.py`) guarantee what prompt rules alone can't:

- **`scrub_report`** strips leaked data-layer machinery — the **run's own tool names** (agent.py
  passes the loaded search/MCP/custom list, so any deployment's naming scheme is covered, with the
  legacy `get_*` family always matched as a fallback; only snake_case names are scrubbed, since a
  plain-word name like "screener" is real English and stripping it would damage prose), "server-side", a
  trailing `(get_x, get_y)` tool list, and **sandbox file paths** (`/workspace/…`, `/skills/…`)
  left in prose) — prose-safe and idempotent. It runs both inside `submit_report` and again when
  persisting, so the user never sees plumbing in the report.
- **`report_problems`** (the gate's detector) additionally flags what a regex cannot rewrite
  safely: file names (`…json`), "offloaded files", function/recipe calls (`R.card(d)`), sub-agent
  mentions — seen live as a "Source: R.price_levels(d) on /workspace/data/…" line and a whole
  "Offloaded Files" section. The gate bounces the report once so the model deletes the machinery
  sentence itself. `findings_gate.py` applies the same test to a sub-agent finding's `source`, so a
  path or recipe name is caught at the handoff, before the orchestrator can copy it.
- **`lint_citations`** reports inline-vs-Sources mismatches (orphans: listed but never cited;
  danglers: cited but not listed) for observability — detection only, it warns rather than silently
  deleting a real source.

The judgment-level check (which claim maps to which source) is the `ReportQualityGateMiddleware`
bounce in §10, because only the authoring model has that knowledge.

The report has a hard length ceiling (~50k chars) in `submit_report` as a backstop against a
pathological raw-row dump; the prompt rules ("aggregate, never transcribe"; "size the finding as a
share of the universe"; "no next-steps") are the primary guard.

---

## 13. How a run ends (and how we know why)

A research turn is *supposed* to end by delivering a report via `submit_report`. `ResearchOutputMiddleware`
classifies the actual end state authoritatively (`_classify`), most-specific cause first:

| End state | Reason | Meaning |
|-----------|--------|---------|
| `done` | `report_delivered` | `submit_report` delivered the report. The happy path. |
| `done` | `direct_answer` | A simple question answered conversationally; no research ran. |
| `done` | `awaiting_clarification` | Paused to ask the user a question. |
| `done` | `report_salvaged` | Model wrote the report as plain prose instead of calling `submit_report`; recovered, scrubbed, and delivered. A recovery, counted so a rising rate flags a misbehaving model. |
| `error` | `budget_exhausted` | Hit the tool-call/token ceiling before delivering. |
| `error` | `stalled_after_nudges` | Kept stopping mid-research with no tool call; force-completion gave up. |
| `error` | `ended_without_report` | Researched, then ended with nothing salvageable. |

The *absence* of this log line means the run died via an exception (e.g. `GraphRecursionError`)
before `after_agent` ran — which the host surfaces as a stream error. This classification, plus the
`usage` event, is the main observability surface for operators.

---

## 14. A worked example

*"Analyze Bitcoin's last 30 days — price, on-chain, and social."*

1. **Triage:** NEEDS RESEARCH (multi-dimensional "analyze").
2. **Plan:** `write_todos` → ["Market action", "On-chain activity", "Social/sentiment", "Synthesis"].
   The orchestrator narrates a sentence of intent for each (streamed live as thinking
   tokens).
3. **Gather:** spawns three `research-subagent`s in parallel via `task`, one per dimension. Each, on
   the cheaper model and in its own context:
   - calls the data tools (`fetch_metric_data`, `social_messages`, `web_search`, …);
   - when a result is large, it offloads to a file and computes aggregates with `execute`;
   - returns a findings JSON — every finding sourced — checked by the findings gate.
4. **Synthesize:** the orchestrator reads the three findings objects, weaves the numbers into one
   markdown report with interleaved `[n]` citations and a grouped `## Sources` section, and calls
   `submit_report`.
5. **Gate + deliver:** the report quality gate verifies citations/hygiene (one bounce if needed),
   `scrub_report` strips any leaked plumbing, the `report` event fires, and the run ends `done /
   report_delivered`. A `usage` event reports the cost.

Throughout, the frontend showed search rows, MCP/tool rows, sources, sub-agent findings, and finally
the report card — driven entirely by the streamed events.

---

## 15. Quick configuration reference

All overridable per-run (`configurable`) or via env var; defaults shown.

| Setting | Env | Default | Purpose |
|---------|-----|---------|---------|
| model tier | `DRA_MODEL_TIER` | `extra-low` | which model package (§3) |
| `OPENAI_API_KEY` | same | — | OpenRouter/OpenAI-compatible key |
| `TAVILY_API_KEY` | same | — | web search; unset → search disabled |
| MCP servers | `DRA_MCP_SERVERS` / `DRA_MCP_URL` | none | data sources |
| `max_tool_calls` | `DRA_MAX_TOOL_CALLS` | 200 | runaway-run ceiling |
| `max_total_tokens` | `DRA_MAX_TOTAL_TOKENS` | 4,000,000 | runaway-run ceiling |
| `compaction_tokens` | `DRA_COMPACTION_TOKENS` | 100,000 | in-flight context compaction trigger (est. tokens); older messages summarized on the utility model, budget counters carry over; 0 = off |
| `prompt_caching` | `DRA_PROMPT_CACHING` | true | `cache_control` breakpoints on system prompt + newest messages (OpenRouter only) |
| `web_fetch` | `DRA_WEB_FETCH` | true | full-page reader tool for sub-agents (big pages offload to the sandbox) |
| `mcp_max_concurrency` | `DRA_MCP_MAX_CONCURRENCY` | 10 | simultaneous MCP calls cap |
| `mcp_rate_limit_max_wait` | `DRA_MCP_RATE_LIMIT_MAX_WAIT` | 120 | seconds a rate-limited MCP call may back off before giving up |
| `max_result_chars` / `max_result_rows` | `DRA_MAX_RESULT_*` | 60k / 1000 | offload/truncate threshold |
| `offload_results` | `DRA_OFFLOAD_RESULTS` | true | offload large results to a file vs truncate |
| `offload_dir` | `DRA_OFFLOAD_DIR` | `/workspace/data` | where offloaded results land (inside the sandbox) |
| sandbox | `LLM_SANDBOX_URL` (+ `_TOKEN`) | unset | enables `execute`; unset → no code execution |
| sandbox network / timeout | `LLM_SANDBOX_NETWORK`, `LLM_SANDBOX_SESSION_TIMEOUT` | false / 900 | outbound network from inside the sandbox; session lifetime (s) |
| skills dir | `DRA_SKILLS_DIR` | `./skills` | read-only skills mount |
| custom tools dir | `DRA_CUSTOM_TOOLS_DIR` | `./custom_tools` | drop-in tools |
| `domain_prompt` | `DRA_DOMAIN_PROMPT` / `_FILE` | empty | deployment-specific text appended to both system prompts (§4); the `_FILE` variant reads it from a path |
| `streaming` | `DRA_STREAMING` | true | live token streaming (some models are force-disabled) |
| `streaming_denylist` | `DRA_STREAMING_DENYLIST` | `deepseek-v4-flash` | model-name substrings that force `streaming` off regardless of the flag |
| `request_timeout` / `max_retries` | `DRA_REQUEST_TIMEOUT`, `DRA_MAX_RETRIES` | 180 / 3 | per-model-call HTTP timeout and retry count — always set, so a hung provider call can't stall a run |
| `recursion_limit` | `DRA_RECURSION_LIMIT` | 4500 | LangGraph super-step ceiling (secondary guard; the budget is primary) |

---

## 16. Running it locally

```bash
./run.sh                       # sync deps + start the LangGraph dev server on :2024
./run.sh ask "<question>"      # stream one run against the running server
./run.sh ask @prompt.txt       # long prompt from a file
./run.sh smoke                 # canned question
./run.sh test                  # offline pytest suite (no API keys / network)
```

Config comes from `./.env` (`OPENAI_API_KEY`, `TAVILY_API_KEY`, `DRA_*`, optional MCP / sandbox).
The agent speaks the LangGraph HTTP/SSE API, so any LangGraph SDK client (or the example in
`examples/client.py`) can drive it.

---

## 17. Source map (where to look in the code)

| Concern | File |
|---------|------|
| Graph factory / wiring | `src/deep_research_agent/agent.py` |
| Per-run config, model tiers, SSRF/base-url guards | `config.py` |
| Model construction (OpenAI-compatible) | `models.py` |
| Prompts (orchestrator, sub-agent, findings format) | `prompts.py` |
| Turn-scoping, token/call counting | `turn.py` |
| Event protocol, tool instrumentation, offload, error classification | `events.py` |
| Tools | `tools/search.py`, `tools/mcp.py`, `tools/custom.py`, `tools/report.py`, `tools/clarify.py` |
| Sub-agent findings gate | `findings_gate.py` |
| Budget / force-completion / report gate / citations / clarify fallback | `budget.py`, `completion.py`, `report_gate.py`, `citations.py`, `clarify_fallback.py` |
| Report hygiene (scrub + lint) | `report_hygiene.py` |
| Skill-usage events | `skill_usage.py` |
| Usage metering | `metering.py` |
| Code sandbox | `sandbox.py` |
| Skills | `skills/` + `skills/README.md` |
| Custom tools guide | `docs/CUSTOM_TOOLS.md` |
| Example consumer | `examples/client.py` |

---

*Questions or corrections: this guide tracks the code as of the `improvements` branch.
The README covers operator/deployment details; this doc covers the runtime behavior.*
