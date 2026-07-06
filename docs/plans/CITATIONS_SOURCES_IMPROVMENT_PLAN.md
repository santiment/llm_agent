# Citations & Sources Rework — Plan

Make citations work on BOTH ends of the model spectrum: a deepseek-v4-flash
orchestrator must be physically unable to ship a garbage bibliography, and an
opus-class orchestrator must lose nothing. Plan only — no code changed.

## The principle (what SOTA systems converged on)

Anthropic's Citations API, Gemini's grounding metadata, and OpenAI's annotation
spans all split the job the same way: **the model decides WHERE a citation
attaches; the SYSTEM owns WHAT the sources are.** No major agent has the model
hand-write a bibliography as markdown — that's exactly the bookkeeping that
broke in our $0.36 mid-tier run (bare `- [1]` entries, lost URLs, a bolded
pseudo-heading). We can't use those provider features (we're multi-provider
through one OpenAI-compatible pipe), but we can implement the same division
provider-agnostically.

Today: the model invents per-run numbering in its head, re-numbers sub-agent
findings, and authors `## Sources` as prose; regex lint + a one-shot gate
police the result. Structure exists only *after* the fact. The rework moves the
structure *before* the model: sources get stable ids when tools return them,
the model only sprinkles ids inline, and code builds the bibliography.

## Design

### 1. `sources.py` — the run-global source registry (new module)

Per-run object, created in `make_graph` next to `RunMeter`, passed to tool
builders and the output middleware. Sub-agents share the orchestrator's tool
instances, so ids survive the `task` boundary for free.

```python
class SourceRegistry:
    def register_web(self, url, title) -> str      # "S12"; dedups by canonical URL
    def register_internal(self, label) -> str       # "S3"; dedups by label
    def lookup(self, sid) -> Source | None
    def resolve(self, marker) -> Source | None      # "S12" or bare "12" (sloppy models)
    def bibliography(self, cited) -> tuple[str, list[dict], dict[str, int]]
        # (markdown section, structured sources[], {Sx -> reader-facing n})
```

- Ids are `S<int>`, assigned in registration order, unique across the whole run
  (no more per-`web_search`-call `[1..6]` collisions — the renumbering the model
  does in its head today is the step flash-tier models fumble).
- `bibliography(cited_ids)`: filters to ids actually cited, renumbers them
  `1..k` in first-citation order, groups internal-source ids onto one line
  automatically. The grouping rules currently enforced by prompt + lint become
  code.

### 2. Tools stamp their results with ids

- `tools/search.py`: register each hit; return `[S12] Title — URL` lines and
  tell the model "cite as [S12]". The `source` event gains the id.
- `events.py` (MCP wrapper): on success, register the server's friendly label
  once and append one footer line to the result:
  `[cite this data as [S3] — Santiment Quantitative Data]`. Ordering matters:
  append AFTER `cap_result`/offload, so the truncation caps can never eat the
  footer (and the offload stub carries it too).
- The DATA SOURCES prompt block (`describe_mcp_sources`) also lists each
  server's id next to its label — belt and braces; the per-result footer is
  the one cheap models actually copy (locality beats instructions).
- Future `web_fetch` does the same as search.

### 3. Sub-agent findings carry ids, not labels

`FINDINGS_FORMAT`'s `source` field becomes "the [Sx] id shown in the tool
result" (validator accepts `S\d+` OR a non-empty label for back-compat — the
findings gate keeps degrading gracefully). The orchestrator can no longer lose
URLs during synthesis because it never holds URLs — only ids.

### 4. The model's citation job shrinks to: sprinkle ids inline

Prompt rewrite (`prompts.py` CITATIONS section): cite claims as `[S12]`; NEVER
write a `## Sources` section — it is appended automatically. The entire
internal-source grouping rulebook (one line per source, no URLs for internal
data, exact label matching…) is deleted from the prompt — that's a large chunk
of instructions cheap models currently violate, gone instead of enforced.
The rewritten section must lead with a worked EXAMPLE ("...volume rose
12%[S4] while sentiment cooled[S7].") — flash-tier models imitate examples far
more reliably than they follow rules; both the orchestrator and sub-agent
prompts carry it.

### 5. Code assembles the report's sources — ONE function, BOTH delivery paths

A single `assemble_report(markdown, registry)` used by (a) the `submit_report`
path and (b) the **salvage path** in `citations.py` — a prose report promoted by
the salvage must get the same bibliography treatment, or the recovery path
ships id-littered text. (Side note: once the prompt stops the model from
writing Sources sections, `completion._looks_delivered`'s "has a Sources
section" signal weakens — length + headings still carry it; leave as is.)

1. Strip any model-written `## Sources` section (deterministic; models will
   sometimes write one anyway — it is never trusted).
2. Scan the body for `[S\d+]` markers. **Bare-number fallback is modal, not
   per-marker**: only when the body contains ZERO `S`-markers do bare `[\d+]`
   markers resolve through the registry (the all-bare legacy/flash mode), and
   only for numbers within the registry's range. Never mix the two schemes in
   one report — a report with S-markers treats bare numbers as plain text.
3. Unknown ids (hallucinated `[S99]`, or stale ids from a previous turn — see
   Multi-turn below): strip the marker, log + count in the `usage` event.
4. Renumber cited ids to reader-facing `[1..k]` in first-citation order,
   rewrite inline markers, append the code-built `## Sources`. Readers see
   exactly today's format — numbered list, markdown links for web, grouped
   labels for internal data.
5. Emit `sources[]` (structured state + events) straight from the registry —
   replacing today's URL-regex harvesting of ToolMessages.

After-bounce end state (explicit): if the model still cites nothing after the
gate's one bounce, the report ships WITHOUT a Sources section and the run is
flagged (`usage.citations`) — an honest absence beats a fabricated list.

### 6. What the guards become

- `report_gate`: bounces on zero-citations-after-research and excessive
  unknown ids. It inspects the RAW `report_markdown` (it wraps the tool call,
  before assembly), so it needs the registry injected at construction —
  `make_graph` passes the same instance it gives the tools and `citations.py`.
  The empty-source-entry and duplicate-label checks become
  impossible-by-construction (kept as cheap backstops on the final output).
- `report_hygiene.scrub_report` (tool-name leaks etc.): unchanged.
- Claim↔source *semantic* mismatch stays out of scope here — that's the cheap
  verifier pass (FEATURES_PLAN #5), which this rework feeds nicely (it gets
  ids + registry instead of parsing prose).

### 7. Multi-turn threads — the stale-id problem (the plan's hardest edge)

The registry is per-run, but the THREAD's message history is not: in a
follow-up turn the model can see run-1's tool results (stamped `[S12]`) and
run-1's final report (renumbered to plain `[1..k]`). Run 2 has a fresh
registry where those markers mean nothing — or worse, mean a DIFFERENT source,
since ids are sequential per run. Naive resolution would silently attach
follow-up claims to the wrong sources.

Policy (deterministic, safe-by-default):

- A marker that is not in the CURRENT run's registry is stripped and counted —
  never resolved against history. Wrong-source attachment is the failure to
  ban; a dropped marker is visible and honest.
- The all-bare fallback mode (step 2 above) is the residual risk: prior-report
  `[1..k]` text pasted into a follow-up could collide with run-2 ids. Accepted
  and documented: the collision requires (zero S-markers) ∧ (copied old
  numbers) ∧ (numbers within range); the verifier pass later catches mis-cites
  semantically. Mitigation if it bites in practice: disable bare-mode when the
  thread has more than one turn.
- Long-term clean fix, NOT in this rework's scope: move the registry into
  checkpointed graph state so ids are stable across a thread's runs (follow-up
  reports could then legitimately re-cite run-1 sources). Listed as the
  follow-up if multi-turn citation continuity becomes a product requirement;
  it changes state schema and middleware access, and nothing simpler depends
  on it today.

## Why this works for both tiers

- **deepseek-v4-flash**: the failure surface shrinks from "maintain a global
  numbering scheme + author a formatted bibliography + follow 10 formatting
  rules" to "copy the id you can see in the tool result". Garbage Sources
  sections become structurally impossible; the worst remaining failure is
  citing a wrong-but-real id (verifier's job later). Bare-number tolerance
  (`resolve`) absorbs the most likely flash mistake.
- **Expensive models**: identical reader-facing output, a shorter prompt, and
  the same deterministic assembly — nothing to regress. Opus simply makes
  better WHERE-to-cite decisions, which stays its job.

## Phases

1. **Registry + stamping** (additive, no behavior change): `sources.py`,
   web_search id lines, MCP footer line, `source` events carry ids. Old
   citation flow untouched — both schemes coexist.
2. **Code-built Sources**: report assembly (strip/scan/resolve/renumber/append),
   prompt CITATIONS rewrite, gate conditions, registry-backed `sources[]`.
   Config flag `structured_sources` (default ON) to fall back to the legacy
   path for one release if something surprises us.
3. **Findings ids**: FINDINGS_FORMAT `source` = id, validator accepts both.
4. **Cleanup**: retire the legacy model-authored-Sources lint rules that are
   now dead weight; remove the flag once both tiers look clean in the
   `usage`/salvage metrics.

## Test list

- Registry: dedup (URL canonicalization, label), order, run-global uniqueness,
  bibliography renumbering + internal grouping.
- Resolution: `[S12]`; all-bare mode resolves `[12]` ONLY when zero S-markers
  exist; mixed-marker report treats bare numbers as plain text; hallucinated
  `[S99]` and stale prior-run ids dropped + counted; bare numbers out of
  registry range left untouched.
- Assembly: model-written Sources stripped; appended section matches readers'
  `[1..k]`; structured `sources[]` equals the cited subset; the SAME function
  produces identical output on the submit path and the salvage path; a
  zero-citation report (post-bounce) ships with no Sources section + a usage
  flag.
- Tools: web_search lines carry ids; MCP footer present incl. on offload stubs.
- Findings gate: id-form and label-form `source` both pass.
- Gate: zero-citations bounce; unknown-id-heavy bounce; bounce-once cap.
- End-to-end fixture replaying the garbage-sources report → impossible to
  reproduce (its Sources section is stripped and rebuilt).

## Risks / open points

- **Context cost**: one id footer line per MCP result, ~10 tokens per
  web_search hit — negligible against the offload savings.
- **Multi-turn threads**: see section 7 — stale markers are stripped, never
  resolved against history; bare-mode collision is the documented residual.
- **Model cites a source it only saw as a snippet**: unchanged from today;
  verifier territory.
- **Mid-transition mixed behavior**: phase 1+2 tolerate legacy `[n]`-only
  reports via the all-bare mode; the flag is the escape hatch.
- **Frontend**: no sanbase changes required — the report markdown keeps its
  shape, and the `source` event's new id field is additive (the EventParser
  ignores unknown keys).
- Effort: M-L (one new module + a shared assembly function on two delivery
  paths + registry injection into tools, gate and citations + prompt rewrite +
  tests). No new dependencies, no graph-shape change. Verified seams: the same
  `tools` list object feeds orchestrator and sub-agents (`agent.py:120,162`),
  so a closure-held registry is shared across the run; `report_gate` reads raw
  `report_markdown` pre-assembly (`report_gate.py:55`).
