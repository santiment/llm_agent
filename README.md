# deep-research-agent

Portable, model-agnostic deep research agent. Built on [`deepagents`](https://github.com/langchain-ai/deepagents) + LangGraph. Plans, asks clarifying questions when the request is ambiguous, spawns parallel sub-researchers, calls web search + MCP tools, runs code in an optional sandbox, applies on-disk skills, and writes a cited report — exposing a **typed streaming-event protocol** so any frontend can render the Claude/Gemini-style live research UI (clarification cards, search queries, website grid, MCP calls, skills, thinking, interleaved `[n]` citations).

**Design goals**
- **No host-app dependency.** Copy this directory (or `uv pip install -e .`) into any app. The only seam is `ResearchConfig` (env + per-run `configurable`). Zero imports from your backend.
- **Not model-locked.** Every model goes through an OpenAI-compatible `base_url` (OpenRouter by default). Models are organized as named price tiers defined in code (`MODEL_TIERS`: any OpenRouter slug, local vLLM, …); runtime selects a tier by name (`DRA_MODEL_TIER=extra-low|low|mid|high`, see Model tiers).
- **Replaceable parts.** Search backend, MCP servers, skills, prompts, and the event emitter are all isolated modules.

This README is the operator/deployment reference. For how the runtime actually behaves — the middleware chain, turn scoping, the gates, delegation, offloading — read [`docs/HOW_THE_AGENT_WORKS.md`](docs/HOW_THE_AGENT_WORKS.md).

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
| `./run.sh ask "<question>"` | Stream one research run against an **already-running** server (fails fast with a message if none is up). |
| `./run.sh smoke` | `ask` a canned question against a running server. |
| `./run.sh doctor` | Check deps, `.env` keys, whether the server is up, and whether the sandbox is reachable — starts nothing. |
| `./run.sh test` | Sync, then run the offline `pytest` suite (no API keys / network). |

Host/port follow `DRA_HOST` (default `127.0.0.1`) `DRA_PORT` (default `2024`; bare `PORT` still works) and `DRA_JOBS` (default `10`, passed as `--n-jobs-per-worker`; without it `langgraph dev` executes one run at a time, so a slow run leaves every later run stuck in `pending`). Auto-reload is off unless `DRA_RELOAD=1`: every source save would otherwise restart the server and kill in-flight runs. The dev server pickles threads, checkpoints and in-flight runs to `./.langgraph_api/` and reloads them on start, so a run that was executing when it stopped resumes by itself with no client attached; `LANGGRAPH_DISABLE_FILE_PERSISTENCE=true` starts empty instead, at the cost of forgetting every thread on restart. `ask`/`smoke` need the server up in another shell first — or use `run-stack.sh` below, which starts everything for you.

### `run-stack.sh` — agent + sandbox together

`run.sh` starts only this agent. When `LLM_SANDBOX_URL` is set, the agent's `execute` tool needs the [llm-sandbox](../llm_sandbox) service running too. `run-stack.sh` brings up both and wires them:

| Command | Does |
|---|---|
| `./run-stack.sh` (or `up`) | Start the sandbox in the background, then the LangGraph server in the foreground. Ctrl-C stops both. |
| `./run-stack.sh ask "<q>"` | Start both, stream one run, tear everything down after. |
| `./run-stack.sh smoke` | Same with a canned question. |
| `./run-stack.sh doctor` | Check both halves — keys, token match, and what isolation the host can actually provide — without starting anything. |

It reuses an already-running sandbox instead of starting a second one, and on exit it kills the sandbox and reaps any session containers a crash left behind. Sandbox repo location: `LLM_SANDBOX_REPO` (default `../llm_sandbox`).

Two things it checks up front, because both otherwise fail deep inside a paid run:

- **Token match.** `LLM_SANDBOX_TOKEN` must be identical in this repo's `.env` and the sandbox's. A mismatch is a 401 on the agent's first tool call.
- **Isolation.** The sandbox refuses to start under a runtime the Docker daemon doesn't have, and `run-stack.sh` relays its verdict. On macOS you will see **NO ISOLATION** — sessions run under `runc`, which is fine for developing and unfit for untrusted code. `./run-stack.sh doctor` prints how to get real gVisor.

A run allocates **one sandbox container**, lazily on the first `execute`/file operation, reuses it for the whole run (so `/workspace` persists), and deletes it when the run ends — the same lifecycle as production, where a session is a gVisor pod instead. Skill helper modules (`skills/<skill>/*.py`) are uploaded
into `/workspace` when the session is created, so a skill can `import` tested code instead of
embedding it in its markdown.

The equivalent manual commands:

```bash
uv sync --extra dev           # create ./.venv with all deps + the langgraph CLI
uv run langgraph dev --host 127.0.0.1 --port 2024 --n-jobs-per-worker 10 --no-reload
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

The same three steps run in CI (`.github/workflows/ci.yml`) on every pull request and on pushes to `main`: `uv lock --check`, then `uv sync --extra dev`, then the suite.

## Dependency policy

Two guards in `pyproject.toml`:

- **Freshness window** — `[tool.uv] exclude-newer` makes uv refuse any distribution *published* in the last ~2 weeks, so a freshly-hijacked package release can't reach this project before the ecosystem notices. uv only takes a static timestamp, so the date is hardcoded; move it forward with `./update_safe_deps_date.sh` (sets it to today − 14 days; `--lock` also re-locks and syncs). CI's `uv lock --check` fails if the lock and the window drift apart.
- **Framework caps** — `deepagents<0.7` and `langchain<2`, because the engine reaches into their internals (middleware hooks, backends, the `tool_call` request field) and a mid-range bump has silently no-op'd a gate before. Raising a cap is a deliberate act: bump, re-lock, run the suite.

**How `pyproject.toml` and `uv.lock` relate** (and why CI checks them as a pair): `pyproject.toml` states what the project *wants* — loose version ranges, plus the freshness date; `uv.lock` records what was *actually picked* — exact versions and checksums for every package (dependencies of dependencies included), so every machine installs identical bits. The freshness date is an **input** to that picking, and the lock records which date it was resolved under. Moving the date without re-locking therefore desyncs the two files, and CI's `uv lock --check` ("would re-resolving change the lock?") fails — that failure is the guard working, not noise.

The rule: never change the date (or any dependency bound) alone. Either run

```bash
./update_safe_deps_date.sh --lock    # moves the date + re-locks + syncs
uv run pytest tests/ -q              # confirm the refreshed versions still pass
```

or follow a manual `pyproject.toml` edit with `uv lock && uv sync --extra dev` and the tests — then commit `pyproject.toml` and `uv.lock` **together**.

## Configuration

Resolution order for every field: per-run `configurable` override → env var → default. `configurable` accepts both this package's native keys and compatibility aliases (`apiKeys`, `mcp_config`, `max_react_tool_calls`) so an existing caller can adopt the agent with zero backend changes. Legacy *model* keys are the exception — they are ignored with a warning, see [Model tiers](#model-tiers-price-packages).

| Env var | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` (or `OPENROUTER_API_KEY`) | — | Key sent as Bearer to `OPENAI_BASE_URL` |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint |
| `DRA_ALLOWED_BASE_URLS` | — | Comma-separated allowlist of extra base URLs a run may override to (key-exfiltration guard) |
| `TAVILY_API_KEY` | — | Web search; if unset, the `web_search` tool is omitted |
| `DRA_MODEL_TIER` | `extra-low` | Named model package: `extra-low` \| `low` \| `mid` \| `high` (see Model tiers below). **The only model knob** — individual models are chosen in code (`MODEL_TIERS`), never per env/run |
| `DRA_REQUEST_TIMEOUT` | `180` | Per-request ceiling (seconds) on every model call — stops one stalled upstream request from pinning a research unit for the run |
| `DRA_MAX_RETRIES` | `3` | Retries per model call on transient 429/5xx; `0` disables |
| `DRA_TEMPERATURE` | `0.3` | Sampling temperature on every model call; `0` (greedy) is what sends small models into endless repetition |
| `DRA_MAX_OUTPUT_TOKENS` | `16000` | Output cap per model call — the bound on one runaway response (a response that hits it is trimmed and the model nudged to continue); `0` sends no cap |
| `DRA_MCP_URL` | — | Single MCP server (bare host → `/mcp` appended) |
| `DRA_MCP_LABEL` | — | Friendly name for that server in the report's Sources |
| `DRA_MCP_SERVERS` | — | JSON list of `{label, url}` for multiple servers |
| `DRA_MCP_BEARER` | — | Bearer token attached to every MCP server lacking explicit auth |
| `DRA_MCP_MAX_CONCURRENCY` | `10` | Hard ceiling on simultaneous MCP calls across the whole run |
| `DRA_MCP_RATE_LIMIT_MAX_WAIT` | `120` | Per-call 429 backoff budget (seconds) before the call fails |
| `DRA_MCP_TOOLS_TTL` | `600` | Seconds a server's tool listing is reused across graph builds (0 = list every run) |
| `DRA_SKILLS_DIR` | `./skills` | Directory of agent skills (see below); in an installed package the default is off — set explicitly |
| `DRA_CUSTOM_TOOLS_DIR` | `./custom_tools` | Directory of drop-in deployment tools (see Custom tools); same installed-package rule |
| `DRA_DOMAIN_PROMPT` | — | Deployment/domain guidance injected into both system prompts (see Domain prompt) |
| `DRA_DOMAIN_PROMPT_FILE` | — | Path to a file with the same; inline `DRA_DOMAIN_PROMPT` wins |
| `DRA_STREAMING` | `true` | Token-by-token streaming; set `false` for models with off-spec streaming chunks |
| `DRA_STREAMING_DENYLIST` | `deepseek-v4-flash` | Comma-separated model-name substrings that force `streaming` off |
| `DRA_RECURSION_LIMIT` | `4500` | LangGraph super-step ceiling for the orchestrator loop (caps loops, not tool calls) |
| `DRA_MAX_TOOL_CALLS` | `200` | Cumulative tool-call ceiling per run (BudgetMiddleware) before a hard stop |
| `DRA_MAX_TOTAL_TOKENS` | `4000000` | Cumulative token ceiling per run; soft wrap-up nudge at 75%, hard stop at 100% |
| `DRA_MAX_RUN_SECONDS` | `2700` | Wall-clock ceiling per run (45 min); same two stages. 0 = no cap |
| `DRA_COMPACTION_TOKENS` | `100000` | Context size (est. tokens) at which older messages are summarized in-flight on the utility model and replaced (compacted work still counts against the budget); `0` disables |
| `DRA_PROMPT_CACHING` | `true` | Inject `cache_control` prompt-cache breakpoints into every model request (OpenRouter base URL only; ignored by non-caching providers) |
| `DRA_WEB_FETCH` | `true` | `web_fetch` tool: sub-agents read a page's FULL text instead of citing search snippets; oversized pages offload like any big tool result |
| `DRA_MAX_RESULT_CHARS` | `60000` | Per-call MCP result size over which the result offloads to a file (or truncates, no sandbox) |
| `DRA_MAX_RESULT_ROWS` | `1000` | Per-call MCP result row count that triggers the same offload/truncate |
| `DRA_OFFLOAD_RESULTS` | `true` | Offload large MCP results to the sandbox filesystem instead of truncating them |
| `DRA_OFFLOAD_DIR` | `/workspace/data` | Directory (inside the sandbox) for offloaded result files |
| `LLM_SANDBOX_URL` | — | Code-execution sandbox sidecar; when set, the `execute` tool runs real shell/Python (the sandbox image is python-only — no node runtime) |
| `LLM_SANDBOX_TOKEN` | — | Auth token; must match the sandbox service's `LLM_SANDBOX_TOKEN` |
| `LLM_SANDBOX_NETWORK` | `false` | Allow outbound network from inside the sandbox |
| `LLM_SANDBOX_SESSION_TIMEOUT` | `5400` | Sandbox session lifetime (seconds). The container — and every file the run offloaded into it — is gone after this, so keep it above `DRA_MAX_RUN_SECONDS` |

Per-run `configurable` keys mirror these: `model_tier`, `apiKeys.{OPENAI_API_KEY,TAVILY_API_KEY}`, `base_url` (allowlisted only), `temperature`, `max_output_tokens`, `request_timeout`, `max_retries`, `search_max_results`, `mcp_servers` / `mcp_config`, `mcp_prompt`, `mcp_max_concurrency`, `mcp_rate_limit_max_wait`, `mcp_tools_ttl`, `domain_prompt`, `skills_dir`, `custom_tools_dir`, `streaming`, `streaming_denylist`, `recursion_limit`, `max_tool_calls`, `max_total_tokens`, `max_run_seconds`, `compaction_tokens`, `prompt_caching`, `web_fetch`, `max_result_chars`, `max_result_rows`, `offload_results`, `offload_dir`, `sandbox_url`, `sandbox_token`, `sandbox_network`, `sandbox_session_timeout`.

### Model tiers (price packages)

Models are chosen by NAME only: `DRA_MODEL_TIER=mid` (or per-run `configurable.model_tier`). Which models a name means is decided in code — `MODEL_TIERS` in `config.py`, one reviewed place — and is **not** settable per env var or per run; legacy per-model keys (`research_model`, `final_report_model`, `compression_model`, …) are ignored with a warning. **The default, when nothing is configured, is `extra-low`** — a bare checkout can't silently burn money; opt up explicitly for real work. An unknown tier name warns and falls back to the default. OpenRouter slugs, prices $/M input/output verified live 2026-09-03 (they drift — re-check before relying on them):

| Tier | Research (orchestrator) | Sub-agent | Utility (extract) | Compaction | Coding |
|---|---|---|---|---|---|
| `extra-low` | `deepseek/deepseek-v4-flash-0731` (0.07/0.18) | `deepseek/deepseek-v4-flash-0731` (0.07/0.18) | `deepseek/deepseek-v4-flash-0731` | `deepseek/deepseek-v4-flash-0731` | `z-ai/glm-5.3-flash` (0.07/0.25) |
| `low` | `qwen/qwen3.8-27b` (0.42/2.55) | `deepseek/deepseek-v4-flash-0731` (0.07/0.18) | `deepseek/deepseek-v4-flash-0731` | `openai/gpt-5.6-luna` (0.20/1.20) | `openai/gpt-5.6-luna` |
| `mid` | `google/gemini-3.7-flash` (0.75/3.75) | `deepseek/deepseek-v4-flash-0731` | `google/gemini-3.5-flash-lite` (0.30/2.50) | `openai/gpt-5.6-luna` | `google/gemini-3.6-flash` (0.75/3.75) |
| `high` | `openai/gpt-5.6-sol` (2/10) | `google/gemini-3.7-flash` (0.75/3.75) | `google/gemini-3.5-flash-lite` | `google/gemini-3-flash-preview` (0.50/3.00) | `deepseek/deepseek-v4-pro` (1.04/2.08) |

**Two invariants every package keeps, both asserted in `tests/test_model_tiering.py`:** the sub-agent is never pricier than the orchestrator (the fleet makes most of the tool calls, so a fleet at planner prices makes the tier's cost the *fleet's* cost), and each tier costs more than the one below it on the research slot. The tests parse the `# $in / $out` comments beside each slug in `MODEL_TIERS`, so those comments are load-bearing — change a model without its price and the suite fails. The **Coding** and **Compaction** slots sit outside the fleet-price rule: the coder gets a *small* input (a goal, file paths, code + error) and writes or fixes one script, and the compactor summarizes a transcript a few times per run at most, so their per-token price barely shows — both are picked for quality. The coder is asserted to be a distinct model, never a silent alias of the fleet slot.

How each slot was picked (OpenRouter benchmarks and task-spend rankings, 2026-09-03): the **research** slot on τ²-bench airline (multi-step tool calling under a policy) plus the `agent:multi_step_planning` / `research_report` spend leaders; the **sub-agent** slot on τ²-bench plus `agent:web_search` / `tool_dispatch`, requiring ≥1M context and the lowest credible price; **utility** on `data:extraction` with context length over depth; **compaction** on the `summarization` category, where the input-heavy call makes the input price rule; **coding** on `code:shell_execution` / `code:debugging`. Dedicated "coder" models (the qwen3-coder family) were dropped: they score 43–47% on τ²-bench and lack the `reasoning` parameter — poor instruction followers for a fix-this-script task.

`extra-low` is rock bottom: one model family end to end (planner and fleet cost about the same), so delegation pays off only via context isolation. Expect weaker planning and earlier give-ups than higher tiers; the force-completion / findings-gate / budget backstops keep runs honest, not great. For demos, smoke tests, and high-volume low-stakes ticks — not for decisions. Every slot in this tier matches the `deepseek-v4-flash` streaming-denylist substring (the `-0731` build included), so nothing in it streams. Every DeepSeek Flash slot is pinned to `-0731`, the current V4 Flash build: the bare `deepseek/deepseek-v4-flash` slug still resolves to the older 0423 one (0.09/0.18, 1.0M context, no `parallel_tool_calls`), so it is strictly worse and unused. `low` buys a real planner (`qwen3.8-27b`, τ² 78.7%, but ~5 min per task median — slow). `mid` puts τ²'s #2 model overall behind the planner at a flash price. `high` swaps `claude-sonnet-5` for `gpt-5.6-sol` (same price and τ² score, +8.7pp GPQA, half the latency) over a fleet that is itself a stronger tool-caller than the planner — intended, since the fleet makes the calls — with the utility slot picked for context length rather than reasoning, since its job is input-heavy map/extract. New models (`gemini-3.8-flash`, released 2026-09-02) were skipped until they appear on the benchmarks.

To add your own packaging: add an entry to `MODEL_TIERS` (code), pick a name, and document it in this table — callers then select it with `DRA_MODEL_TIER=<name>`. An unknown tier name is ignored with a warning (plain defaults apply).

## Streaming event protocol

Stream with `stream_mode=["messages","updates","custom"]` and `stream_subgraphs=True`. The `custom` channel carries protocol events (each a JSON object with `type`); the `messages` channel carries assistant **thinking** tokens for the collapsible pane.

The contract is pinned in code: `events.EVENT_SCHEMAS` registers every type's required keys, `emit` warns on drift, and tests enforce both. Every run opens with a `run_start` handshake — check `protocol_version` there (bumped only on breaking shape changes; additive keys/types don't bump) instead of failing mid-render on an unfamiliar shape.

| `type` | Key fields | Renders as |
|---|---|---|
| `run_start` | `protocol_version`, `engine_version`, `started_at` | Version handshake, first event of every run (no UI); `started_at` (UTC) anchors the run time if the run dies before its end events |
| `clarification` | `questions[]` | Question card; input re-enabled. On submit, reply on the **same thread** with each answer paired to its question (`1. Q: … A: …`) — not bare answers |
| `search_query` | `id`, `query`, `source` | Globe row |
| `search_results` | `id`, `query`, `ok`, `count`, `results[].{title,url,domain,snippet}` | Favicon + title grid |
| `source` | `title`, `url`, `domain` | Live citation list entry |
| `mcp_call` | `id`, `tool`, `args` | MCP call row |
| `mcp_result` | `id`, `tool`, `ok`, `summary`; on failure `error_class` = `permanent` \| `transient` \| `unknown` (+ `repeated` when an identical failed call was answered locally) | MCP result row |
| `tool_call` / `tool_result` | same fields as the `mcp_*` pair above | Identical rows for NON-MCP instrumented tools (web search, drop-in `custom_tools/`); only the emitting layer differs, so a renderer can treat the two pairs as one |
| `skill` | `name`, `path`, `state` | "Skill applied: `<name>`" indicator |
| `subagent_findings` | `unit`, `summary`, `findings[].{finding,evidence,source}`, `gaps[]` | Folded findings table (one per sub-agent); emitted when a sub-agent's findings validate |
| `report` | `markdown` | Final answer (also in state `final_report`) |
| `usage` | `tool_calls`, `total_tokens`, `model_calls`, `elapsed_s`, `elapsed`, `limits{}`, … | Per-run ledger at run end (no UI; logging / cost tracking), incl. run time |
| `status` | `state` = `mcp_ready` \| `mcp_error` \| `budget_soft` \| `budget_halt` \| `revising` \| `compacting` \| `compacted` \| `loop_detected` \| `loop_halt` \| `subagent_start` \| `subagent_done` \| `done` \| `error` | Lifecycle / errors |

`status` detail: `mcp_ready` carries `tool_count` + `tools[]`; `mcp_error` carries `detail`, `server`, `label`; `budget_soft` is the 75% wrap-up nudge and `budget_halt` the hard ceiling stop (see budgets below); `revising` fires when a gate bounces a deliverable back for one revision — `reason: report_quality` (final report) or `reason: subagent_findings` (a sub-agent's findings handoff); `compacting` / `compacted` bracket a context compaction (with `tokens_estimate` before and the summarized-away counters after); `loop_detected` is the repeated-identical-call nudge and `loop_halt` its hard stop (both carry `repeats`); `sandbox_reset` means the sandbox session died mid-run and was replaced (`detail` carries the error; files written before it are gone); `triage` carries the pre-triage verdict (`detail: simple|research`); `model_call` fires right before every model step with `role` (`orchestrator` / a sub-agent role), `model`, `step`, `unit` (the brief it works on) and, after tool results, `after` + `after_chars` (which tools just returned, how much) — the heartbeat that says who is thinking about what while a non-streaming model produces nothing visible; `subagent_start` / `subagent_done` bracket every sub-agent run with its `role` (`research-subagent` / `extract-subagent` / `coding-subagent`) and `model` — the live signal that the cheap utility model is doing the reading (`subagent_done` adds `model_calls` and `total_tokens`). `done` / `error` is the run's authoritative end-state, emitted exactly once with a `reason` code: `done` for `report_delivered`, `awaiting_clarification`, `direct_answer` or `report_salvaged`; `error` for `budget_exhausted`, `stalled_after_nudges` or `ended_without_report`. Absence of either means the run died on an exception before the end-of-run hook and the host surfaces a stream error.

The `usage` event (from `metering.py`) reports orchestrator-level token counts plus global tool-call / result-size totals and the configured ceilings — emitted once at run end for logging and cost tracking.

**Run time is always reported at the end, success or error.** The final `status` (`done` or `error`) and the `usage` event both carry `elapsed_s` (seconds) and `elapsed` (`4m 12s`), and the end status's human `detail` sentence ends with `Run time 4m 12s.` — so a frontend that renders only `detail` for the "finished without producing a report" case still shows it. `run_start` carries `started_at` (UTC ISO): when a run dies before its end hooks (an exception the host streams as a stream `error`), no `status`/`usage` arrives, and the consumer computes the time from that anchor. `examples/client.py` does both — it prints the agent's `detail` and, in a `finally`, its own wall-clock run time after every run, whatever happened. The `RUN END` / `RUN ENDED WITHOUT REPORT` / `RESEARCH USAGE` log lines carry the same figure.

Final thread state also exposes `final_report` (string) and `sources` (`[{index,url,domain}]`) — structured citations independent of the inline `[n]` markers the writer model produces.

**Async / background runs (Gemini-style "leave this chat").** LangGraph persists the thread, so a run survives client disconnect. Reconnect by joining the run stream or polling `GET /threads/{id}/state` for `final_report`.

## Clarifying questions

When a request is ambiguous (unclear scope, timeframe, entity, or goal) the orchestrator calls `request_clarification` up front, emits a `clarification` event, and stops.

**The reply must pair each answer with its question.** The frontend collects the answers and sends them back on the **same thread** as one user message restating each question with its answer — e.g.:

```
Answers to your clarifying questions:
1. Q: Scope to US-listed BDCs or global? A: US-listed
2. Q: Timeframe? A: last 12 months
```

Sending bare answers (`"the first"`) loses meaning — without the question the agent can't tell what "the first" refers to. As insurance the `request_clarification` tool also echoes the questions into its own result, so they stay in context even if history is trimmed. The user's reply lands on the same thread, so the agent then has the full Q&A in context and proceeds to research. A deterministic fallback (`ClarificationFallbackMiddleware`) emits the same event if a model narrates questions in prose without calling the tool, so the card always appears regardless of model. See `examples/client.py` for the round-trip.

## Domain prompt

The base system prompts are deliberately **domain-neutral** — they carry the research workflow and the engine contracts (findings format, `submit_report` protocol, clarification protocol) that the middleware enforces. Everything specific to *your* deployment's domain — the analytical dimensions that matter (e.g. on-chain activity and tokenomics for crypto; yield and non-accruals for credit), terminology, example asks, report register — goes in the **domain prompt**, injected into both the orchestrator's and the sub-agents' system prompts as a labeled `DOMAIN CONTEXT` block.

Set it per run (`configurable.domain_prompt`), inline (`DRA_DOMAIN_PROMPT`), or from a file (`DRA_DOMAIN_PROMPT_FILE=/path/to/domain.md` — the usual place for anything longer than a sentence). Empty = the slot collapses and the base prompt runs as-is. The domain prompt **extends** the base prompt; the engine contracts are not replaceable — don't restate workflow, citation, or output rules in it, just the domain color.

## Skills

Skills are folders under `./skills/`, each with a `SKILL.md` (progressive-disclosure instructions read on demand). They're mounted **read-only** at the virtual path `/skills/` for the **research sub-agents**, which read them via `read_file("/skills/<name>/SKILL.md")` while their scratch files stay in the sandbox. The orchestrator holds no file tools; its prompt lists each skill's name and description (from the `SKILL.md` front matter) and it names the skill in a `task` brief — so the 4k-token skill file lives in the worker's context, not on every planner step. The first time a sub-agent reads a skill in a turn, a `skill` event fires ("Skill applied: `<name>`"). Point elsewhere with `DRA_SKILLS_DIR` / `configurable.skills_dir`; if the directory is absent the agent runs normally with no skills.

## Custom tools

Add a deployment-specific tool without touching the generic codebase: drop a `*.py` file in `./custom_tools/` and restart. Each file subclasses `CustomTool` — set `name` / `description`, implement `run` — and the loader auto-discovers it, infers the arg schema from `run`'s typed params, and gives it to the orchestrator **and** every sub-agent.

```python
# custom_tools/weather.py
from deep_research_agent.tools.custom import CustomTool

class WeatherNow(CustomTool):
    name = "weather_now"
    description = "Current weather for a city. Cite as 'OpenWeather'."

    async def run(self, city: str) -> str:   # sync def works too
        # self.cfg is the run config; return a string (hardcoded here as an example)
        return f"{city}: 21°C, clear skies, humidity 48%. Source: OpenWeather."
```

`run` may be sync or `async`; `self.cfg` is the live `ResearchConfig`. **Return value:** the model always sees a string — return a `str` (JSON-encode structured data yourself), or a `list`/`dict` and the framework JSON-encodes it for you; a large `list` of rows is offloaded to a file the `execute` tool reads back. Override `enabled(cls, cfg) -> bool` to load conditionally (e.g. only when an env var is set). Copy `custom_tools/_template.py` to start; for dynamic cases a `build_tools(cfg)` / `build_tool(cfg)` factory returning LangChain tools is also accepted. Point elsewhere with `DRA_CUSTOM_TOOLS_DIR`. Full guide: [`docs/CUSTOM_TOOLS.md`](docs/CUSTOM_TOOLS.md).

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
- The graph factory runs once per research run, and listing a server's tools is a network round-trip — the single biggest cost of a graph build (the dev server's "Slow graph load" warning). Listings are cached for `mcp_tools_ttl` seconds (default 10 min; failures are never cached), so only the first run after a start or a deploy pays it. Set `DRA_MCP_TOOLS_TTL=0` while iterating on the MCP server's tool schemas.
- SSRF guard: only `http(s)` schemes are allowed and link-local / cloud-metadata targets are refused. Loopback / private hosts are allowed (the internal gateway uses them).
- Connection failures emit `status: mcp_error` (with detail) instead of failing silently — one unreachable server does not take down the others or the run.
- A FAILED tool call never kills the run: the error is returned to the model as the tool result with retry guidance, classified `permanent` (validation / unknown names — fix the arguments, never retry) vs `transient` (one retry ok). Servers can tag explicitly by prefixing the error message with `[permanent]` / `[transient]`; an identical retry of a permanently-failed call is answered locally without hitting the server.

## Code execution, large results & budgets

- **Code execution.** Set `LLM_SANDBOX_URL` (+ `LLM_SANDBOX_TOKEN`) to attach an llm-sandbox sidecar; deepagents' `execute` tool then runs real shell / Python / JS in the container, so the model computes aggregates and joins instead of doing arithmetic in its head. One session per run, alive for `LLM_SANDBOX_SESSION_TIMEOUT`; if the container is gone anyway (`No such container`, 404), the backend opens a fresh one, retries the call once, emits `status: sandbox_reset`, and tells the model that earlier files are lost — instead of every later `execute` failing for the rest of the run. With no sandbox configured the agent falls back to an in-memory backend and execution is disabled — it degrades gracefully and says so rather than faking output.
- **Pre-triage router.** The first model step of every fresh user turn is a ~250-token call on the research model: SIMPLE (answerable from general knowledge — greetings, definitions, "capital of Bulgaria") or RESEARCH. SIMPLE is answered in one more small call and the turn ends without ever loading the ~12k-token orchestrator harness; RESEARCH, long messages (>1,500 chars) and any router failure run the full agent, whose own TRIAGE still handles clarifications and follow-ups answerable from the report in context. Emits `status: triage` with `detail: simple|research`; the verdict call is tagged `nostream` so it never shows as thinking.
- **Trimmed orchestrator toolbox.** The orchestrator has no `ls`/`read_file`/`write_file`/`edit_file`/`glob`/`grep` (nor their descriptions and schemas on every step) and no skills mount — files and skills are the sub-agents' business, reached through `task`; `execute` stays for the odd one-line aggregate.
- **Sub-agent roster is closed.** deepagents' auto-added `general-purpose` sub-agent (the planner's own toolbox, no data tools) is disabled through a harness profile, so `task` only ever offers `research-subagent`, `extract-subagent` and `coding-subagent` — planners were delegating whole units to it and getting back "the report has been compiled" with nothing behind it.
- **Coding sub-agent.** With a sandbox attached, every tool-loop role (orchestrator and research-subagents) can hand a programming job to `coding-subagent` via `task`: write a Python script for a goal over named `/workspace` files, or fix code given the code and the exact error. It runs on the tier's `coding_model` (a dedicated coder), keeps `write_file`/`edit_file`/`read_file`/`execute` but no data tools, and hands back `STATUS / SCRIPT / OUTPUT / NOTES` — the script path and the script's *real* output, no narration. The prompts route every failed `execute` there and forbid `python -c "…"` one-liners (shell quoting is what breaks them — scripts are files), which is what ends the "I will now try a safer quoting approach…" loops on weak models.
- **Large-result offload.** When a single MCP result exceeds `DRA_MAX_RESULT_CHARS` / `DRA_MAX_RESULT_ROWS`, the full payload is written to a file under `DRA_OFFLOAD_DIR` and only a compact stub (path, row count, columns, head) enters context; the model reads the file back with `execute`. Without a sandbox these bounds become hard truncation caps instead. This is how a large cross-entity scan stays within the context window. A **metric time series is offloaded regardless of size**: the stub carries a computed summary (span, first/last, min/max with when, mean, median, direction) and the rule that a series is never listed row by row; any timestamped rows that still reach a report are collapsed to that summary on delivery.
- **Budgets.** `BudgetMiddleware` enforces cumulative per-run ceilings — `DRA_MAX_TOOL_CALLS`, `DRA_MAX_TOTAL_TOKENS` and the wall clock `DRA_MAX_RUN_SECONDS` (calls and tokens do not bound time: a fleet retrying against a dead sandbox once ran 90 minutes under both) — emitting a `budget_soft` wrap-up nudge at 75% and a `budget_halt` hard stop at 100%; the status `reason` names the binding ceiling (`tool_calls` / `tokens` / `time`). `DRA_RECURSION_LIMIT` separately caps orchestrator super-steps. The `usage` event reports the run's spend against these ceilings at the end.
- **Context compaction.** When an agent's estimated context crosses `DRA_COMPACTION_TOKENS` (default 100k; 0 = off), everything before the current turn's newest messages is summarized on the tier's compaction model — findings with figures and sources, offloaded file paths, gaps, next steps — and the transcript is replaced with `[summary, user request, recent tail]`. The summarized-away tool calls and tokens keep counting against the budget (compaction never grants fresh budget); a failed summarizer call compacts nothing and the run continues untouched.
- **Loop guard.** Identical tool calls (same tool + args + result) repeated within the last 10 calls get a break-the-loop nudge at 3 repeats and a hard stop at 6 — the successful-but-useless twin of the permanent-failure memo. Applies to the orchestrator and every sub-agent.
- **Runaway output guard.** A response that degenerates into repetition (`BTC2026, BTC2027, …` for thousands of tokens) or hits `DRA_MAX_OUTPUT_TOKENS` is trimmed to its useful head, the model gets one nudge to continue (`status: runaway_output`), and a second runaway in the same turn ends the run (`status: runaway_halt`). Same scope as the loop guard.
- **Usage & cost.** The end-of-run `usage` event now includes per-role sub-agent model usage (`subagents`), a whole-run token grand total (`total_tokens_all_agents`), compaction counters, and a best-effort `cost_usd` — OpenRouter's actual charged cost, captured on non-streamed calls only (a lower bound, not an invoice).
- **Prompt caching.** With an OpenRouter base URL, every model request carries Anthropic-style `cache_control` breakpoints (system prompt + the two newest cacheable messages), cutting input cost on caching providers and ignored elsewhere. `DRA_PROMPT_CACHING=false` turns it off.

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
  triage.py           pre-triage router — SIMPLE questions answered in two tiny calls, no harness
  budget.py           BudgetMiddleware — tool-call + token + wall-clock ceilings (soft nudge → hard stop)
  compaction.py       in-flight context compaction — summarize-and-shrink near the window
  loop_guard.py       identical-tool-call loop detector (nudge → hard stop)
  caching.py          prompt-cache breakpoints (cache_control) per model request
  clarify_fallback.py emits clarification event when a model narrates questions in prose
  skill_usage.py      emits a skill event the first time each skill is read in a turn
  turn.py             scopes thread messages to the current turn (multi-turn safety)
  report_hygiene.py   deterministic scrub + citation lint applied to the final report
  report_gate.py      report quality gate — bounces a report back once for fixable defects
  metering.py         per-run usage ledger → usage event + RESEARCH USAGE log
  sandbox.py          wires the execute / filesystem tools to the llm-sandbox sidecar
  tool_filter.py      hides built-in tools per sub-agent role (extract: execute only; coding: no grep/todos)
  tools/search.py     Tavily web_search, emits search events
  tools/fetch.py      web_fetch full-page reader (stdlib HTML→text, SSRF-guarded)
  tools/mcp.py        MCP loader + per-call instrumentation, concurrency + 429 backoff
  tools/clarify.py    request_clarification tool → clarification event
  tools/report.py     submit_report tool — the single explicit deliverable → report event
  tools/custom.py     CustomTool base class + drop-in loader for custom_tools/
custom_tools/         drop-in deployment-specific tools (CustomTool subclasses), auto-loaded
skills/               agent skills (each a folder with SKILL.md), mounted read-only at /skills/
examples/client.py    reference SSE consumer
```
