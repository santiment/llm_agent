# deep-research-agent

Portable, model-agnostic deep research agent. Built on [`deepagents`](https://github.com/langchain-ai/deepagents) + LangGraph. Plans, asks clarifying questions when the request is ambiguous, spawns parallel sub-researchers, calls web search + MCP tools, runs code in an optional sandbox, applies on-disk skills, and writes a cited report — exposing a **typed streaming-event protocol** so any frontend can render the Claude/Gemini-style live research UI (clarification cards, search queries, website grid, MCP calls, skills, thinking, interleaved `[n]` citations).

**Design goals**
- **No host-app dependency.** Copy this directory (or `uv pip install -e .`) into any app. The only seam is `ResearchConfig` (env + per-run `configurable`). Zero imports from your backend.
- **Not model-locked.** Every model goes through an OpenAI-compatible `base_url` (OpenRouter by default). Models are organized as named price tiers defined in code (`MODEL_TIERS`: any OpenRouter slug, local vLLM, …); runtime selects a tier by name (`DRA_MODEL_TIER=extra-low|low|mid|high`, see Model tiers).
- **Replaceable parts.** Search backend, MCP servers, skills, prompts, and the event emitter are all isolated modules.

## Run standalone

This project is managed with [`uv`](https://docs.astral.sh/uv/) (`uv.lock` is committed). Use `uv` — do **not** `pip install` into your base interpreter.

```bash
cp .env.example .env          # set OPENAI_API_KEY, TAVILY_API_KEY (+ optional DRA_MCP_*)
./run.sh                      # sync deps (first run) + start the dev server on :2024
```

`run.sh` is a one-command dev bring-up. It loads `./.env` (so the script and server share config), syncs `./.venv` on first run, and starts the LangGraph server:

| Command | Does |
|---|---|
| `./run.sh` (or `./run.sh up`) | Sync deps if `./.venv` is missing, then start the dev server (API + docs at `http://127.0.0.1:2024/docs`). Warns if `OPENAI_API_KEY` / `TAVILY_API_KEY` are unset. |
| `./run.sh --sync` | Force `uv sync --extra dev`, then start the server. |
| `./run.sh ask "<question>"` | Stream one research run against an **already-running** server. |
| `./run.sh smoke` | `ask` a canned question against a running server. |
| `./run.sh test` | Sync, then run the offline `pytest` suite (no API keys / network). |

Host/port follow `DRA_HOST` (default `127.0.0.1`) and `PORT` (default `2024`). `ask`/`smoke` need the server up in another shell first. The equivalent manual commands:

```bash
uv sync --extra dev           # create ./.venv with all deps + the langgraph CLI
uv run langgraph dev --host 127.0.0.1 --port 2024
uv run python examples/client.py "What are the recent trends across the tracked entities, and where can I find supporting data?"
```

Graph id: **`deep_research_agent`** — set this as your caller's `assistant_id`.

## Tests

Tests live in `tests/` (e.g. the deterministic report-hygiene guard — `scrub_report` + `lint_citations` / `report_problems`). Pure-Python, no API keys or network needed. `pytest` ships in the `dev` extra, so the suite runs inside `./.venv` alongside the runtime deps:

```bash
./run.sh test                # sync + run the suite (equivalent to the two commands below)
uv sync --extra dev          # installs pytest + deepagents + the langgraph CLI into ./.venv
uv run pytest tests/ -q
```

## Configuration

Resolution order for every field: per-run `configurable` override → env var → default. `configurable` accepts both this package's native keys and compatibility aliases (`research_model`, `final_report_model`, `apiKeys`, `mcp_config`, `mcp_prompt`) so an existing caller can adopt the agent with zero backend changes.

| Env var | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` (or `OPENROUTER_API_KEY`) | — | Key sent as Bearer to `OPENAI_BASE_URL` |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint |
| `DRA_ALLOWED_BASE_URLS` | — | Comma-separated allowlist of extra base URLs a run may override to (key-exfiltration guard) |
| `TAVILY_API_KEY` | — | Web search; if unset, the `web_search` tool is omitted |
| `DRA_MODEL_TIER` | `extra-low` | Named model package: `extra-low` \| `low` \| `mid` \| `high` (see Model tiers below). **The only model knob** — individual models are chosen in code (`MODEL_TIERS`), never per env/run |
| `DRA_MCP_URL` | — | Single MCP server (bare host → `/mcp` appended) |
| `DRA_MCP_LABEL` | — | Friendly name for that server in the report's Sources |
| `DRA_MCP_SERVERS` | — | JSON list of `{label, url}` for multiple servers |
| `DRA_MCP_BEARER` | — | Bearer token attached to every MCP server lacking explicit auth |
| `DRA_MCP_MAX_CONCURRENCY` | `10` | Hard ceiling on simultaneous MCP calls across the whole run |
| `DRA_MCP_RATE_LIMIT_MAX_WAIT` | `120` | Per-call 429 backoff budget (seconds) before the call fails |
| `DRA_SKILLS_DIR` | `./skills` | Directory of agent skills (see below) |
| `DRA_STREAMING` | `true` | Token-by-token streaming; set `false` for models with off-spec streaming chunks |
| `DRA_STREAMING_DENYLIST` | `deepseek-v4-flash` | Comma-separated model-name substrings that force `streaming` off |
| `DRA_RECURSION_LIMIT` | `4500` | LangGraph super-step ceiling for the orchestrator loop (caps loops, not tool calls) |
| `DRA_MAX_TOOL_CALLS` | `200` | Cumulative tool-call ceiling per run (BudgetMiddleware) before a hard stop |
| `DRA_MAX_TOTAL_TOKENS` | `4000000` | Cumulative token ceiling per run; soft wrap-up nudge at 75%, hard stop at 100% |
| `DRA_MAX_RESULT_CHARS` | `60000` | Per-call MCP result size over which the result offloads to a file (or truncates, no sandbox) |
| `DRA_MAX_RESULT_ROWS` | `1000` | Per-call MCP result row count that triggers the same offload/truncate |
| `DRA_OFFLOAD_RESULTS` | `true` | Offload large MCP results to the sandbox filesystem instead of truncating them |
| `DRA_OFFLOAD_DIR` | `/workspace/data` | Directory (inside the sandbox) for offloaded result files |
| `LLM_SANDBOX_URL` | — | Code-execution sandbox sidecar; when set, the `execute` tool runs real shell/Python/JS |
| `LLM_SANDBOX_TOKEN` | — | Auth token; must match the sandbox service's `LLM_SANDBOX_TOKEN` |
| `LLM_SANDBOX_NETWORK` | `false` | Allow outbound network from inside the sandbox |
| `LLM_SANDBOX_SESSION_TIMEOUT` | `900` | Sandbox session timeout (seconds) |

Per-run `configurable` keys mirror these: `model_tier`, `apiKeys.{OPENAI_API_KEY,TAVILY_API_KEY}`, `base_url` (allowlisted only), `temperature`, `search_max_results`, `max_concurrent_research_units`, `mcp_servers` / `mcp_config`, `mcp_prompt`, `mcp_max_concurrency`, `mcp_rate_limit_max_wait`, `skills_dir`, `streaming`, `streaming_denylist`, `recursion_limit`, `max_tool_calls`, `max_total_tokens`, `max_result_chars`, `max_result_rows`, `offload_results`, `offload_dir`, `sandbox_url`, `sandbox_token`, `sandbox_network`, `sandbox_session_timeout`.

### Model tiers (price packages)

Models are chosen by NAME only: `DRA_MODEL_TIER=mid` (or per-run `configurable.model_tier`). Which models a name means is decided in code — `MODEL_TIERS` in `config.py`, one reviewed place — and is **not** settable per env var or per run; legacy per-model keys (`research_model`, `final_report_model`, `compression_model`, …) are ignored with a warning. **The default, when nothing is configured, is `extra-low`** — a bare checkout can't silently burn money; opt up explicitly for real work. An unknown tier name warns and falls back to the default. OpenRouter slugs, prices $/M input/output as of 2026-06:

| Tier | Research (orchestrator) | Sub-agent | Utility |
|---|---|---|---|
| `extra-low` | `deepseek/deepseek-v4-flash` (0.10/0.20) | `openai/gpt-oss-120b` (0.04/0.18) | `openai/gpt-oss-20b` (0.03/0.14) |
| `low` | `deepseek/deepseek-v4-pro` (0.44/0.87) | `deepseek/deepseek-v4-flash` (0.10/0.20) | `deepseek/deepseek-v4-flash` |
| `mid` | `google/gemini-3.5-flash` (1.50/9) | `google/gemini-2.5-flash` (0.30/2.50) | `deepseek/deepseek-v4-flash` |
| `high` | `anthropic/claude-opus-4.8` (5/25) | `anthropic/claude-sonnet-4.6` (3/15) | `anthropic/claude-haiku-4.5` (1/5) |

`extra-low` is rock bottom — cheapest tool-capable models everywhere (~$0.02 of orchestrator spend per medium run). Expect noticeably weaker planning and earlier give-ups; the force-completion / findings-gate / budget backstops keep runs honest, not great. For demos, smoke tests, and high-volume low-stakes scheduled ticks — not for decisions. `high` deliberately keeps sub-agent/utility at sonnet/haiku tier — Opus plans and synthesizes only; an Opus sub-agent fleet would defeat the tiering. To add your own packaging: add an entry to `MODEL_TIERS` (code), pick a name, and document it in this table — callers then select it with `DRA_MODEL_TIER=<name>`. An unknown tier name is ignored with a warning (plain defaults apply).

## Streaming event protocol

Stream with `stream_mode=["messages","updates","custom"]` and `stream_subgraphs=True`. The `custom` channel carries protocol events (each a JSON object with `type`); the `messages` channel carries assistant **thinking** tokens for the collapsible pane.

| `type` | Key fields | Renders as |
|---|---|---|
| `clarification` | `questions[]` | Question card; input re-enabled (user replies on the same thread) |
| `search_query` | `id`, `query`, `source` | Globe row |
| `search_results` | `id`, `query`, `ok`, `count`, `results[].{title,url,domain,snippet}` | Favicon + title grid |
| `source` | `title`, `url`, `domain` | Live citation list entry |
| `mcp_call` | `id`, `tool`, `args` | MCP call row |
| `mcp_result` | `id`, `tool`, `ok`, `summary`; on failure `error_class` = `permanent` \| `transient` \| `unknown` (+ `repeated` when an identical failed call was answered locally) | MCP result row |
| `skill` | `name`, `path`, `state` | "Skill applied: `<name>`" indicator |
| `report` | `markdown` | Final answer (also in state `final_report`) |
| `usage` | `tool_calls`, `total_tokens`, `model_calls`, `limits{}`, … | Per-run ledger at run end (no UI; logging / cost tracking) |
| `status` | `state` = `mcp_ready` \| `mcp_error` \| `budget_soft` \| `budget_halt` \| `revising` \| `done` | Lifecycle / errors |

`status` detail: `mcp_ready` carries `tool_count` + `tools[]`; `mcp_error` carries `detail`, `server`, `label`; `budget_soft` is the 75% wrap-up nudge and `budget_halt` the hard ceiling stop (see budgets below); `revising` fires when a gate bounces a deliverable back for one revision — `reason: report_quality` (final report) or `reason: subagent_findings` (a sub-agent's findings handoff); `done` fires when the report is finalized.

The `usage` event (from `metering.py`) reports orchestrator-level token counts plus global tool-call / result-size totals and the configured ceilings — emitted once at run end for logging and cost tracking.

Final thread state also exposes `final_report` (string) and `sources` (`[{index,url,domain}]`) — structured citations independent of the inline `[n]` markers the writer model produces.

**Async / background runs (Gemini-style "leave this chat").** LangGraph persists the thread, so a run survives client disconnect. Reconnect by joining the run stream or polling `GET /threads/{id}/state` for `final_report`.

## Clarifying questions

When a request is ambiguous (unclear scope, timeframe, entity, or goal) the orchestrator calls `request_clarification` up front, emits a `clarification` event, and stops. The user's reply lands on the **same thread** as the next message, so the agent then has the Q&A in context and proceeds to research. A deterministic fallback (`ClarificationFallbackMiddleware`) emits the same event if a model narrates questions in prose without calling the tool, so the card always appears regardless of model.

## Skills

Skills are folders under `./skills/`, each with a `SKILL.md` (progressive-disclosure instructions the agent reads on demand). They're mounted **read-only** at the virtual path `/skills/`; the agent reads them via `read_file("/skills/<name>/SKILL.md")` while its own scratch files stay in an ephemeral state backend. The first time a skill is read in a turn, a `skill` event fires ("Skill applied: `<name>`"). Point elsewhere with `DRA_SKILLS_DIR` / `configurable.skills_dir`; if the directory is absent the agent runs normally with no skills.

## Wiring into an existing app

It speaks the LangGraph HTTP/SSE API, so any consumer (the included `examples/client.py`, the JS `@langchain/langgraph-sdk`, or raw SSE) works. To wire it into an existing deployment:

1. Run this graph (point your dev script / `langgraph.json` at it).
2. Set `assistant_id` to `deep_research_agent`.
3. Pass per-run config via `configurable` (see the Configuration table above).
4. To get the rich live UI, have the frontend additionally consume the `custom` event channel above.

### MCP connection notes

**Who connects, and where the config comes from.** The agent is always the MCP *client* — it opens the connection itself (at graph build, `agent.py` → `load_mcp_tools`) and the model calls the resulting tools during research. There is no separate connector process. What varies is where the server list (url + auth) is resolved from. Precedence (first non-empty wins, `config.py`):

1. `configurable.mcp_servers` — per-run request (native).
2. `configurable.mcp_config` — per-run request (compat alias). **The normal host-app path:** the backend injects url + `headers` (incl. auth) into every run, so the env vars below are never consulted.
3. `DRA_MCP_SERVERS` — env (JSON list).
4. `DRA_MCP_URL` (+ `DRA_MCP_LABEL`) — env (single server).

So when a request arrives **with** MCP config, the agent connects using *that* (and its auth). When a bare run arrives **without** it — e.g. a Studio / `langgraph dev` trigger, or any caller that omits `configurable.mcp_config` — it falls back to the `DRA_MCP_*` env entry. The env entry is a standalone-run fallback, not the primary path. If that fallback has no auth, you get the failure below.

**Auth / `401 Unauthorized`.** A `401` means the connection *reached* the server and was rejected for missing/wrong credentials — the path is correct, so do **not** strip `/mcp` (that would give `404`, a different error). Attach credentials instead:
- request-supplied servers: put them in `headers` (e.g. `{"Authorization": "Bearer …"}` or a server-specific header like `x-litellm-api-key`).
- env-supplied servers: set `DRA_MCP_BEARER=<token>` — it's attached as `Authorization: Bearer <token>` to every server that doesn't already carry explicit auth.

To keep bare local runs from attempting an auth-less connect at all, leave `DRA_MCP_URL` unset and rely on the backend to inject `mcp_config`.

**`/mcp` path rule differs by source.** Under `mcp_config`, `url` is treated as a **base** and `/mcp` is appended for you — pass the url *without* `/mcp`. Under `DRA_MCP_URL` / `mcp_servers`, the url is used as given except that a **bare host** gets `/mcp` appended; a url that already has a path is left untouched — so pass the full url *with* `/mcp`.

**Other guards.**
- Connect to **`127.0.0.1`**, never `0.0.0.0` (bind address — dialing it fails). Config normalizes `0.0.0.0` → loopback defensively.
- Each call is bounded by a shared semaphore (`mcp_max_concurrency`) so the agent's fan-out can't exhaust the server's file descriptors; 429s back off and retry within `mcp_rate_limit_max_wait` rather than failing immediately.
- SSRF guard: only `http(s)` schemes are allowed and link-local / cloud-metadata targets are refused. Loopback / private hosts are allowed (the internal gateway uses them).
- Connection failures emit `status: mcp_error` (with detail) instead of failing silently — one unreachable server does not take down the others or the run.
- A FAILED tool call never kills the run: the error is returned to the model as the tool result with retry guidance, classified `permanent` (validation / unknown names — fix the arguments, never retry) vs `transient` (one retry ok). Servers can tag explicitly by prefixing the error message with `[permanent]` / `[transient]`; an identical retry of a permanently-failed call is answered locally without hitting the server.

## Code execution, large results & budgets

- **Code execution.** Set `LLM_SANDBOX_URL` (+ `LLM_SANDBOX_TOKEN`) to attach an llm-sandbox sidecar; deepagents' `execute` tool then runs real shell / Python / JS in the container, so the model computes aggregates and joins instead of doing arithmetic in its head. With no sandbox configured the agent falls back to an in-memory backend and execution is disabled — it degrades gracefully and says so rather than faking output.
- **Large-result offload.** When a single MCP result exceeds `DRA_MAX_RESULT_CHARS` / `DRA_MAX_RESULT_ROWS`, the full payload is written to a file under `DRA_OFFLOAD_DIR` and only a compact stub (path, row count, columns, head) enters context; the model reads the file back with `execute`. Without a sandbox these bounds become hard truncation caps instead. This is how a large cross-entity scan stays within the context window.
- **Budgets.** `BudgetMiddleware` enforces cumulative per-run ceilings — `DRA_MAX_TOOL_CALLS` and `DRA_MAX_TOTAL_TOKENS` — emitting a `budget_soft` wrap-up nudge at 75% and a `budget_halt` hard stop at 100%. `DRA_RECURSION_LIMIT` separately caps orchestrator super-steps. The `usage` event reports the run's spend against these ceilings at the end.

## Layout

```
src/deep_research_agent/
  agent.py            make_graph(config) factory  ← langgraph.json entrypoint
  config.py           env + per-run config (the portability seam)
  models.py           OpenAI-compatible model builder
  events.py           event protocol + tool instrumentation (mcp_call/mcp_result)
  prompts.py          orchestrator + subagent prompts (citation + MCP-source rules)
  citations.py        output middleware → final_report + sources[]
  completion.py       force-completion middleware (no premature ReAct termination)
  findings_gate.py    sub-agent findings gate — JSON contract, validator, bounce (report_gate's twin)
  budget.py           BudgetMiddleware — hard tool-call + token ceilings (soft nudge → hard stop)
  clarify_fallback.py emits clarification event when a model narrates questions in prose
  skill_usage.py      emits a skill event the first time each skill is read in a turn
  turn.py             scopes thread messages to the current turn (multi-turn safety)
  report_hygiene.py   deterministic scrub + citation lint applied to the final report
  report_gate.py      report quality gate — bounces a report back once for fixable defects
  metering.py         per-run usage ledger → usage event + RESEARCH USAGE log
  sandbox.py          wires the execute / filesystem tools to the llm-sandbox sidecar
  tools/search.py     Tavily web_search, emits search events
  tools/mcp.py        MCP loader + per-call instrumentation, concurrency + 429 backoff
  tools/clarify.py    request_clarification tool → clarification event
  tools/report.py     submit_report tool — the single explicit deliverable → report event
skills/               agent skills (each a folder with SKILL.md), mounted read-only at /skills/
examples/client.py    reference SSE consumer
```
