# Features Plan — Deep Research Agent

What to build next to make the agent more useful, cheaper to run, and production-ready.
Ranked by impact. Each item lists concrete code changes, effort, and confidence.

**Current state (verified):** LangGraph/deepagents orchestrator + one `research-subagent`
type, both on the *same* `research_model` (`agent.py:64`, `agent.py:102-112`). Budget
middleware, report quality gate, citation lint, usage metering, MCP concurrency guard,
large-result offload to an optional gVisor sandbox. Sanbase consumes it via a LiveView
SSE client (`lib/sanbase/deep_research/`) with **zero persistence**. The sandbox sidecar
(`~/work/llm_sandbox`) is one Docker container per session, `sleep <timeout>` as PID 1.

---

## 1. Model tiering — cheap models for sub-agents and utility steps

**The single biggest cost lever.** Today every token — orchestrator planning,
sub-agent grinding through MCP pagination, the final report prose — is billed at the
research-model rate. `report_model` is parsed (`config.py:198-199`) but **never used**
(`agent.py:62-64` says "reserved for a future dedicated synthesis step"). Worse,
sanbase already *sends* `summarization_model` and `compression_model`
(`config.ex:62-64` in the worktree) and the agent **silently ignores both**.

The shape: **smart orchestrator, cheap sub-agents** — the expensive model plans,
delegates, and synthesizes; everything mechanical runs tiers down. Sub-agent work
splits into two job classes with different floors:

- **Tool-loop research units** (a `/hot` validation sub-agent making 5-15 sequential
  MCP calls with judgment between them): a model one tier down
  (e.g. `anthropic/claude-haiku-4-5`, `google/gemini-2.5-flash`) — flash-class models
  degrade on long tool loops (malformed calls, premature termination), and the
  orchestrator only sees findings, so silent low quality propagates into the report.
- **Pure map/extract/verify** (read an offloaded file slice → structured findings;
  compaction summaries (#2), fetch-page digests (#3), verifier rubric (#5)): the
  cheapest capable model, e.g. `deepseek/deepseek-v4-flash` — typically 5-15× below
  haiku-tier. No tool loop, nothing to watch stream. Its known OpenRouter streaming
  bug (dropped tool_calls, doubled metadata) is already mitigated: it ships in the
  streaming denylist (`config.py:141`) and `models.py` force-disables streaming.

Changes:
- `config.py` — two knobs: `subagent_model` (tool-loop units, default =
  `research_model`, env `DRA_SUBAGENT_MODEL`) and `utility_model` (map/extract/verify,
  default = `subagent_model`, env `DRA_UTILITY_MODEL`). Accept the already-sent
  `compression_model` as an alias for `utility_model` so sanbase works day one.
- `agent.py` — `subagent_spec["model"] = build_chat_model(cfg.subagent_model, cfg)`;
  a second `extract-subagent` spec on `utility_model` (same tools minus `task`, prompt
  tuned for "read file, return structured findings") that the orchestrator prompt
  steers map-style work to.
- **Escalation ladder** (cheap insurance against flash flakiness): map sub-agents
  return JSON against a schema; deterministic validation; on malformed/empty output
  retry once on `utility_model`, then re-run that slice on `subagent_model`. Failed
  slices cost a haiku-tier call; the rest ride flash rates.
- Actually wire `report_model`: see feature 6 (dedicated report writer).
- `metering.py` — split the usage ledger per model so the `usage` event shows
  orchestrator vs sub-agent vs utility spend; without this you can't prove the
  savings or spot a flash model burning retries.

Risk: a weak sub-agent model hallucinates findings the orchestrator trusts. Mitigate by
keeping the sub-agent prompt's "source after each claim" rule and adding the cheap
verifier (feature 5). Start tool-loop units one tier down, not at the floor; let the
metering data justify pushing lower.

**Effort:** S (knobs + wiring) / M (escalation ladder). **Impact:** very high (often
60-80% of tokens are sub-agent tokens; volume paths like cron briefs and per-coin
`/talk` map steps multiply it). **Confidence:** high — config plumbing exists,
deepagents accepts per-subagent models, the flash streaming bug is already handled.

## 2. Context compaction — keep the orchestrator's window from filling up

The budget middleware *stops* a run at the ceiling; nothing *shrinks* a run in flight.
Every tool result and sub-agent summary stays verbatim in the orchestrator's messages
until the end. Long multi-turn threads compound it (each turn replays the thread).

Changes:
- New `compaction.py` middleware: when the current turn crosses a threshold (e.g. 50%
  of `max_total_tokens`), replace *old, already-consumed* ToolMessages with one-line
  stubs ("result used; key numbers: …") produced by the cheap utility model — or, when
  a sandbox is attached, write the original body to `/workspace/context/` so the model
  can recover it via `execute` if it truly needs it. Tool results whose offload stub is
  already a file pointer compact to almost nothing.
- Per-sub-agent budgets: `budget.py:16-18` explicitly defers this. A sub-agent gone
  rogue today is bounded only by per-call caps and `recursion_limit`. Give the subagent
  graph its own `BudgetMiddleware` with `max_tool_calls = cfg.max_tool_calls // 4` (own
  config knob later).
- Multi-turn: `turn.py` already scopes budget counting to the current turn; add an
  option to summarize *prior turns* into a single context message (cheap model) instead
  of replaying them verbatim.

**Effort:** M. **Impact:** high — directly extends how deep a single run can go, and
cuts cost on every long run. **Confidence:** medium-high — middleware seam exists; the
trick is only compacting messages the model is done with (compact *after* a newer
AIMessage references them, never the most recent N).

## 3. `web_fetch` tool — read full pages, not 600-char snippets

`web_search` returns Tavily snippets truncated to 600 chars (`tools/search.py:73`).
That is fine for discovery, useless for actually *reading* a source — the thing
Gemini Deep Research and Claude research do constantly. The agent currently cites pages
it has effectively only seen the abstract of.

Changes:
- New `tools/fetch.py`: `web_fetch(url)` → httpx GET + readability extraction
  (`trafilatura` or Tavily Extract API — Tavily key already present, zero new infra).
  Emit `search_query`/`source` events so the UI shows it. Respect the same
  `max_result_chars` offload path: a long page goes to a sandbox file with a stub, and
  the model reads/greps it via `execute`.
- Optionally summarize-on-fetch with the cheap utility model when no sandbox is
  attached (snippet → relevant-to-query digest), so fetches don't blow up context.
- Prompt: add the tool to the TOOLS section in `prompts.py`; tell the model to fetch
  before citing any web source whose snippet doesn't contain the claimed fact.

**Effort:** S-M. **Impact:** high — research depth and citation accuracy jump
immediately. **Confidence:** high.

## 4. Tool-result cache — stop paying twice for the same call

Every run re-fetches everything. Multi-turn follow-ups on the same thread, repeated
analyses of the same asset, and any future scheduled/repeated runs re-pay the full MCP +
web cost for identical calls. MCP metric data and web pages are stable on the scale of
minutes-to-hours.

Changes:
- `tools/cache.py`: TTL cache keyed on `(server, tool, canonical-json-args)`, wrapped
  inside the existing MCP tool wrapper (`tools/mcp.py`) and `web_search`/`web_fetch`.
  Default backend: in-process dict per server process; optional sqlite/redis URL via
  `DRA_CACHE_URL` for cross-run reuse. TTL per source type (`DRA_CACHE_TTL`, default
  ~15 min; 0 disables).
- Meter cache hits in `RunMeter` → the `usage` event grows `cache_hits` /
  `saved_calls`.
- Also turn on **prompt caching**: with OpenRouter + Anthropic/Gemini models, marking
  the (large, stable) system prompt as cacheable cuts input-token cost on every
  ReAct iteration. The orchestrator prompt + MCP tool schemas are identical across all
  ~7 super-steps per loop. Investigate `extra_body={"usage": ...}`/cache_control
  pass-through in `models.py` — even partial support is a big win because the prompt is
  resent dozens of times per run.

**Effort:** M. **Impact:** high for repeated/iterative use; prompt caching alone can cut
input cost 50%+ on supported models. **Confidence:** high for the result cache;
medium for prompt caching via OpenRouter (provider-dependent pass-through).

## 5. Cheap verifier pass before the report ships

`ReportQualityGateMiddleware` is deterministic (format, duplicate sources, raw tool
names) — it cannot catch a number that doesn't match its source or an invented claim.
And when it bounces, the *expensive* model rewrites.

Changes:
- New `verifier.py` middleware (or a step in the report-writer node, feature 6): after
  `submit_report`, run the **cheap model** over (report, sub-agent findings, source
  stubs) with a narrow rubric: every `[n]` claim supported by findings? totals
  consistent? denominators present? Output: pass, or a short defect list that is bounced
  back exactly once (reuse the existing `revising` status event — UI already renders it).
- Adversarial option for decision-type asks (thesis/risk): spawn one cheap
  "bear-case" sub-agent whose only job is to attack the draft's weakest claim; attach
  its strongest objection to the report. This is the cheap version of the
  adversarial-bear idea from COMMANDS_PLAN.md, implemented in the agent where every
  client benefits — not in command templates.

**Effort:** M. **Impact:** medium-high — accuracy is the product; one caught
fabrication pays for the feature. **Confidence:** medium-high — bounce mechanics
already exist (`report_gate.py`), this adds a semantic check on top.

## 6. Dedicated report-writer step (finally use `report_model`)

Synthesis is a different job from research. Today the orchestrator writes the final
report inline, in a context polluted by tool noise, with the research model. A
dedicated writer step gets: a *clean* context (findings + source list only), a model
chosen for prose (can be cheaper OR better than the research model), and a natural home
for the verifier pass.

Changes:
- `tools/report.py` / new node: `submit_report` becomes "hand over findings"; a writer
  node builds the report from (user question, todos, consolidated findings, source
  registry) using `build_chat_model(cfg.report_model, cfg)`, then quality gate +
  verifier run against *it*.
- This also fixes the report-revision cost: bounces re-prompt the writer with a small
  context, not the orchestrator with the whole run.
- Keep a fallback flag (`DRA_DEDICATED_WRITER=false`) to preserve current behavior.

**Effort:** M-L. **Impact:** medium-high — report quality up, synthesis cost down,
revision cost way down. **Confidence:** medium — needs care so citations `[n]` survive
the handoff (pass the source registry explicitly; `citations.py` already builds it).

## 7. Persistence + background runs in sanbase

The agent already supports detach/reattach (LangGraph persists threads; README line
107). The sanbase UI throws all of it away: turns live in socket assigns, a page reload
loses the conversation and the report (`deep_research_live.ex`, no Ecto anywhere in
`lib/sanbase/deep_research/`).

Changes (sanbase worktree, not this repo):
- Ecto schema: `dra_threads` (user_id, langgraph_thread_id, title, inserted_at) and
  `dra_turns` (thread_id, question, report_md, sources_json, usage_json, status).
  Persist on `report` / terminal status events; backfill via
  `GET /threads/{id}/state` when the stream is lost.
- Thread list + reopen UI; "leave this page, research continues" — the run already
  survives disconnect server-side, the LiveView just needs to re-join by polling state
  (the `{:dra_poll, ...}` path already exists at `deep_research_live.ex:219-243`).
- This is also the **right home for `/watch`-style scheduled re-runs** (Oban cron job
  per saved watch, fresh thread per tick) — see the commands verdict below.

**Effort:** M. **Impact:** high for real users — without it the product forgets
everything. **Confidence:** high.

## 8. Sandbox hardening (fixes + capability)

The sidecar works but has real operational holes (verified in `~/work/llm_sandbox`):

| Problem | Fix |
|---|---|
| **Session dies mid-run.** Container PID 1 is `sleep <timeout>` (default 900 s) started at *first* sandbox use; a research run longer than 15 min loses every offloaded file and `execute` starts failing. | Keepalive: agent-side, extend on each call (service endpoint `POST /sessions/{sid}/touch` that restarts the sleep clock, or just create sessions with `timeout = max_expected_run` and rely on `SandboxCleanupMiddleware` + service reaper). Agent should also *recreate* the session and retry once on "container not found". |
| Stopped containers accumulate forever — `docker run` without `--rm` (`gvisor.py:65`). | Add `--rm`. Self-reaping then actually reaps. |
| Concurrent `/run` race — fixed filename `/workspace/_run_python.py` (`app.py:79`); parallel sub-agents sharing the session will stomp each other. | Unique suffix per request (`_run_<8hex>.py`). |
| No clean 404 for dead sessions — Docker errors propagate as confusing exec failures. | Map "No such container" to HTTP 404; agent retries with a fresh session. |
| No package install — image is fixed, network off by default. | Extend the image (requests, matplotlib, scipy, openpyxl, statsmodels cover most research compute); document that `network: true` + `pip install --user` is the escape hatch. |
| Default token `change-me` in both `.env` files. | Refuse to start with the default token unless `SANDBOX_ALLOW_INSECURE=1`. |

**Effort:** S-M total (each fix is small). **Impact:** medium alone, but features 1-4
all lean on offload/execute — an unreliable sandbox undermines them. **Confidence:**
high — all confirmed in code.

---

## Smaller bug fixes in this repo (do regardless)

1. **Config default mismatch.** `ResearchConfig.max_tool_calls` dataclass default is
   `200` (`config.py:151`) and README documents 200/4M — but `from_runnable_config`
   falls back to `or 80` (`config.py:289`) and `or 2_000_000` (`config.py:292`). Every
   real run goes through `from_runnable_config`, so the *actual* defaults are 80 / 2M.
   Pick one number per knob and use it in both places.
2. **Stale docstring in `budget.py:20-21`**: claims "`models.py` sets
   `stream_usage=True`" — `models.py:40-45` deliberately does the opposite. The
   chars/4 fallback carries token accounting; say so.
3. **Silently ignored configurable keys.** Sanbase sends `summarization_model`,
   `compression_model`, `search_api`, `allow_clarification` — none are read. At minimum
   log unknown configurable keys once per run; better, honor the model keys (feature 1)
   and `allow_clarification` (skip the clarify tool when false).
4. **Sanbase `direct_answer?` heuristic** (`timeline.ex:73-82`) misclassifies a run
   that emits no thinking and no report as a silent success. The `status: done` /
   `budget_halt` events already disambiguate — use them instead of inference.

---

## Verdict on COMMANDS_PLAN.md

The dispatch/registry/macro core is sound, but it's pointed at the wrong boundary:
**`examples/client.py` is a dev tool, not the product surface.** The production client
is the sanbase LiveView (Elixir) — a Python-client command layer is invisible to every
real user. Recommendation:

- **Keep** the idea of file-based macro templates (`commands/*.md` already written) —
  but expand them **server-side**, where every client benefits. Two options:
  (a) sanbase expands `/analyze BTC` → template before POSTing the run (Elixir port of
  the ~40-line dispatch — trivial); (b) the agent itself accepts
  `configurable.command` + args and prepends the expanded template at graph input.
  Option (a) is simpler and keeps the agent generic; start there.
- **Drop `/watch` from the client plan.** An in-session Python loop dies with the
  terminal. Scheduled re-runs belong in sanbase as Oban jobs over persisted threads
  (feature 7) — durable, per-user, survives restarts.
- **Drop `/model`, `/budget`, `/depth` as client commands** — they're one-line
  `configurable` overrides; expose them as UI controls in the LiveView instead.
- The decision-support discipline in `thesis.md`/`risk.md`/`dd.md` (invalidation
  triggers, confidence, adversarial bear) is the most valuable part of the catalog —
  and the adversarial bear is better built once in the agent (feature 5) than restated
  per template.

Net: salvage the templates and the discipline; skip Phases 3-4 of the Python-client
plan.

## Suggested order

1. Bug fixes (above) + feature 1 (model tiering) — one PR, immediate cost cut.
2. Feature 3 (`web_fetch`) + feature 8 sandbox keepalive — research depth + reliability.
3. Feature 4 (caching) + feature 2 (compaction) — the affordability pair.
4. Feature 6 (report writer) + feature 5 (verifier) — quality pair, share a seam.
5. Feature 7 (sanbase persistence) — parallel track, different repo.

| # | Feature | Effort | Impact | Confidence |
|---|---------|--------|--------|------------|
| 1 | Smart orchestrator / cheap sub-agents + utility model | S-M | very high | high |
| 2 | Context compaction + sub-agent budgets | M | high | med-high |
| 3 | `web_fetch` full-page tool | S-M | high | high |
| 4 | Result cache + prompt caching | M | high | high / med |
| 5 | Cheap verifier pass | M | med-high | med-high |
| 6 | Dedicated report writer | M-L | med-high | medium |
| 7 | Sanbase persistence + background runs | M | high | high |
| 8 | Sandbox hardening | S-M | medium | high |
