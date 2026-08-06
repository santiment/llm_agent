# Model Tiering — Implementation Plan

FEATURES_PLAN #1: **smart orchestrator, cheap sub-agents**, plus the agreed
**structured sub-agent response with deterministic validation**. Status: sections
1-3, 6, 7 are IMPLEMENTED (module landed as `findings_gate.py`); sections 4
(escalation ladder) and 5 (metering split) remain deferred as planned.

## Verified before planning (so the plan stands on facts)

- deepagents' `SubAgent` spec (installed package,
  `deepagents/middleware/subagents.py:27-101`) supports **`model`** ("Override the
  main agent's model" — accepts a `BaseChatModel` instance) and **`middleware`**
  ("Additional middleware for custom behavior") per sub-agent. Both hooks we need
  exist; no fork required.
- The bounce mechanics are proven in-repo twice: `ForceCompletionMiddleware`
  (`completion.py:72-121`) uses `@hook_config(can_jump_to=["model"])` +
  `{"jump_to": "model", "messages": [HumanMessage(..., name=NUDGE)]}` with the nudge
  cap counted from message names in state; `ReportQualityGateMiddleware`
  (`report_gate.py`) shows the bounce-once-then-accept policy.
- Nothing constructs `ResearchConfig(...)` directly outside
  `from_runnable_config` — new required dataclass fields are safe to add.
- The sub-agent's final message IS what the orchestrator receives (the `task` tool
  relays it verbatim) — so the validation point is the sub-agent's own
  `after_model`, attached via `subagent_spec["middleware"]`.
- deepseek-v4-flash's broken OpenRouter streaming is already mitigated: it ships in
  `streaming_denylist` (`config.py:141`) and `models.py:26-29` force-disables
  streaming per model id — `build_chat_model(cfg.subagent_model, cfg)` gets this for
  free.
- Tests are sync, standalone-runnable (`python tests/test_x.py`), no pytest-asyncio
  (style: `tests/test_budget_caps.py`).

## 1. Config — the tier cascade

`config.py`, two new required fields right after `report_model`:

```python
# Model tiering — smart orchestrator, cheap sub-agents. The orchestrator plans,
# delegates and synthesizes on research_model; sub-agents run their tool loops on
# subagent_model (a tier down); utility_model is the floor (flash-class) for pure
# map/extract/verify work. Each defaults to the tier above it, so setting only
# research_model keeps today's single-model behavior.
subagent_model: str
utility_model: str
```

Resolution in `from_runnable_config`, after `report_model`:

```python
subagent_model = _strip_provider(
    c.get("subagent_model") or _env("DRA_SUBAGENT_MODEL", default=research_model))
utility_model = _strip_provider(
    c.get("utility_model") or c.get("compression_model")          # compat alias
    or _env("DRA_UTILITY_MODEL", default=subagent_model))
```

Cascade: `research_model → subagent_model → utility_model`. Unset = inherit the tier
above, so existing deployments change nothing. `compression_model` is accepted as a
`utility_model` alias because sanbase already sends it (`config.ex:62-64` in the
worktree) — it starts working the day this ships.

`utility_model` has **no consumer yet** — it's plumbed now so the verifier
(FEATURES_PLAN #5), compaction (#2), and `/talk` map workers configure against a
stable key. State this in the field comment.

## 2. Agent wiring

`agent.py`:

```python
research_model = build_chat_model(cfg.research_model, cfg)
subagent_model = (research_model if cfg.subagent_model == cfg.research_model
                  else build_chat_model(cfg.subagent_model, cfg))
```

- `subagent_spec["model"] = subagent_model`
- `subagent_spec["middleware"] = [SubagentFindingsMiddleware()]` (section 3)
- Log the tier split once per run (`research=%s subagent=%s utility=%s`) — the first
  question when debugging a bad report is "which model wrote these findings".
- Update the stale comment at `agent.py:61-64` ("both run on the research model").

Recommended defaults to try (not hardcoded — operator choice):
`DRA_SUBAGENT_MODEL=anthropic/claude-haiku-4-5` (tool loops need judgment),
`DRA_UTILITY_MODEL=deepseek/deepseek-v4-flash` (map/extract floor).

## 3. Structured findings + deterministic validation

New module `src/deep_research_agent/findings_gate.py`. Rationale: with sub-agents on a
cheaper tier, the handoff must be checkable — a weaker model economizes first on
attribution, and an unsourced finding poisons the report's citations downstream.

**Contract** (inserted into `SUBAGENT_PROMPT` in `prompts.py` as a
`FINDINGS_FORMAT` block):

```json
{"summary": "<dense prose digest of the unit — figures, dates, named entities>",
 "findings": [{"finding": "<one specific claim, with its numbers>",
               "evidence": "<the data behind it: values, quotes, dates>",
               "source": "<URL for web; the EXACT internal source label for data tools>"}],
 "gaps": ["<what could not be determined, and why>"]}
```

**`extract_findings(text)`** — tolerant parse: bare object, ```json fence, or first
`{` to last `}` substring (models add courtesy prose). None if nothing parses.

**`findings_problems(text) -> list[str]`** — deterministic, no model in the loop:
- parses to a dict;
- `summary`: non-empty string;
- `findings`: a list; each entry an object with non-empty `finding` AND non-empty
  `source`;
- **empty `findings` is explicitly allowed** — an honest "nothing found" beats
  validation pressure to fabricate;
- `gaps` optional, must be a list when present.

**`SubagentFindingsMiddleware`** — attached per sub-agent, NOT the orchestrator:
- `after_model`, `can_jump_to=["model"]`: last message is an `AIMessage` with no
  `tool_calls` and non-empty text → run `findings_problems`.
- Invalid → bounce ONCE: `jump_to: model` + `HumanMessage(name=FINDINGS_NUDGE_NAME)`
  listing the specific problems ("do not change any finding, number, or source —
  only the format").
- Cap (`MAX_FINDINGS_NUDGES = 1`) counted from message names in the sub-agent's own
  state, **never instance attributes** — one middleware instance serves every
  parallel `task` invocation.
- Cap exhausted → log + accept as-is. Prose degrades gracefully; the gate must never
  fail a run over formatting.

Orchestrator side needs no change: it reads the JSON as text and synthesizes. (A
later refinement can have the orchestrator prompt reference the structure, e.g.
"trust only findings that carry sources".)

## 4. Escalation ladder — deferred, by design

Re-running a failed slice on a better model can't happen inside one sub-agent run
(the model is fixed at graph build). The natural seam appears with the
`extract-subagent` for map work (`/talk` slices, FEATURES_PLAN #1): the orchestrator
sees the relayed findings, and its prompt can instruct "if a unit's findings came
back malformed or empty twice, re-delegate that unit to `research-subagent`" — i.e.
escalation = re-delegation to the stronger-typed sub-agent, orchestrated where the
judgment already lives. Ship validation-with-bounce first; add the second sub-agent
type + escalation guidance when the first `utility_model` consumer lands.

## 5. Metering split

`metering.py`: tag usage per model id so the `usage` event reports
`{by_model: {"<id>": {tokens, model_calls}}}` alongside the totals. Two purposes:
prove the savings, and spot a flash model burning its budget on retries (a quiet
escalation-ladder failure signature). Needs a look at how `RunMeter` attributes
sub-agent callbacks before sizing — if attribution isn't already per-model in the
callback metadata, this becomes its own small task; don't block tiering on it.

## 6. Tests — `tests/test_model_tiering.py`

Sync, standalone-runnable, no network. Construct configs via
`from_runnable_config({"configurable": {...}})` with explicit `research_model` so
ambient `DRA_*` env vars can't leak in (set/restore `os.environ` around the env-var
cases):

- Cascade: only `research_model` set → `subagent_model == research_model`,
  `utility_model == subagent_model`.
- Overrides: `subagent_model` set → utility follows it; both set → independent.
- Alias: `compression_model` → `utility_model`; native `utility_model` wins over it.
- Env: `DRA_SUBAGENT_MODEL` honored; configurable key beats env.
- Provider stripping: `openai:deepseek/deepseek-v4-flash` → bare slug.
- `findings_problems`: valid bare JSON / fenced / prose-wrapped → `[]`; empty
  findings + summary → `[]`; missing summary, finding without source, non-list
  findings, bare prose → named problems.
- Middleware: prose final message → `jump_to: model` with the named nudge; nudge
  already in state → accepted (None); valid JSON → None; `AIMessage` with
  tool_calls → None; empty content → None.

## 7. Docs

- README config table: `DRA_SUBAGENT_MODEL` (default = research model),
  `DRA_UTILITY_MODEL` (default = subagent model; alias `compression_model`); add both
  + `subagent_model`/`utility_model` to the configurable-keys line; layout entry for
  `findings_gate.py`.
- FEATURES_PLAN #1: mark config/wiring/validation as specced here.

## Build order (one PR, ~5 commits)

1. `config.py` fields + cascade resolution (+ config tests).
2. `findings_gate.py` (validator only) + validator tests.
3. `prompts.py` FINDINGS_FORMAT block + `SubagentFindingsMiddleware` + middleware
   tests.
4. `agent.py` wiring (model + middleware + log + stale-comment fix).
5. README + plan-doc updates.

Out of scope here (tracked in FEATURES_PLAN): `extract-subagent` type + escalation
re-delegation, metering per-model split (5 above, if non-trivial), dedicated report
writer (`report_model` stays unused for now).

## Risks / open decisions

- **JSON findings vs prose quality** (medium): structured output can slightly flatten
  nuance vs free prose. Mitigation: `summary` stays free prose; only attribution is
  structured. If synthesis quality drops, the format can become advisory
  (validate-and-log, no bounce) via a config flag — decide after a few real runs.
- **Which models** (operator decision, not code): haiku-tier for tool loops,
  flash-tier for map work is the starting recommendation; metering data should drive
  any push lower.
- **deepagents version drift** (low): `model`/`middleware` keys verified against the
  installed version; pin or re-verify on upgrade.
