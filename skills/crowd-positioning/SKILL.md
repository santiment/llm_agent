---
name: crowd-positioning
description: >-
  Read how the crowd is POSITIONED around a coin during a social spike and whether that
  positioning is EXTREME vs the coin's own history — not a news summary. Use when a token is
  trending, its social volume / dominance is spiking (now or at some past date), or someone asks
  "what happened with X / why is it being talked about / is this hype real / is the crowd long or
  short here / who is driving it". Turns raw `social_messages` into four signals: (1) extremeness
  vs a trailing baseline (percentiles, not adjectives), (2) organic vs manufactured (copies,
  channel/author concentration, acceleration shape), (3) narrative-vs-chain divergence (the crowd's
  concrete claims tested against on-chain flows), (4) crowd price-level clusters (the support /
  targets people actually name). Surfaces the dominant trending topics driving the spike (the "what
  happened") and delivers a one-page positioning VERDICT with denominators, then answers follow-up
  questions on the same pulled data.
---

# Crowd positioning

You are **positioning the crowd, not reporting on it.** The failure mode this skill exists to kill
is *journalism with the data* — a readable recap of what people said. Nobody pays for the recap.
They pay for the answer to one question:

> **How is the crowd positioned, and is that positioning extreme?**

Every line you emit is a **signal with a baseline**, never a story. Not "lots of bullish ETF talk"
— that's journalism. Instead: "social volume is 94th percentile of trailing 90d; 71% bullish vs a
53% trailing-30d median; 68% organic and still accelerating; chain confirms it (–$210M net off
exchanges this window)." If a sentence has no number and no comparison, cut it. If a list has more
than five lines, it is a data dump, not a finding.

## When to use

A token is trending or its social dominance/volume spiked (now or a named past window) and the user
wants to understand it — "what happened with $X around <date>", "why is everyone talking about Y",
"is this real or bots", "is the crowd long here". You have a token and (ideally) a time window. If
the window is missing, infer it with **one** call (`trending_stories`, or `fetch_metric_data` on
social volume to find the peak), **state the window you chose**, and move on — never iterate on
the window.

## The four signals (this is the whole product)

1. **Extreme vs history?** — A spike is meaningless without its own baseline. "12th percentile of
   trailing 90d" is a signal; "high social activity" is not. Rank this window's social volume,
   dominance, and sentiment skew against the coin's trailing distribution.
2. **Organic or manufactured?** — Same volume means opposite things if it's 5k people or one room
   pasting 200×. "Organic" is a DEFINED, computed number: run `R.card(d)` from `recipes.py`
   (pre-loaded in the sandbox; collapses exact, template and near-duplicate posts over the random
   stratum, sizes each cluster by accounts and rooms, and adds account concentration, posting
   cadence, link spam and hour fingerprint). Never eyeball uniqueness from the text and never read
   it off `copies` or `unique_after_dedup` alone — those see only byte-identical copies, so a price
   bot changing one number per post scores 100% unique. Output the number with its basis: "62%
   organic (near-duplicates collapsed), still accelerating" vs "bot campaign: one account = 31% of
   posts".
3. **Does the narrative match the chain?** — The crowd's *claims* are testable. "BlackRock
   accumulating" + chain shows $150M flowing **to** Coinbase = a divergence worth flagging. Pull
   each checkable claim's on-chain metric and report match or divergence, with the number.
4. **Where does the crowd put price?** — Price levels named in messages are positioning data.
   Extract them, cluster them, and report the one or two strongest levels below and above the
   current price, with how many voices back each. (E.g. "62772 was a trend word" → a level, not a
   topic.) Never the full list of every number people typed.

The computations live in `recipes.py` (next to this file), seeded into the sandbox as
`/workspace/recipes.py`. `signals.md` (also next to this file) says how to call each recipe, how to
read its output, and holds the claim→metric map. Read `signals.md`; never retype recipe code.

## Who does what

Three model slots are fixed by config (`MODEL_TIERS`), not by you. You pick the model only by picking
the subagent:

- **Orchestrator** (research model) — plans, spawns **one** `research-subagent` for the whole skill
  on one coin and window (a second one only for a second coin or a second window — the steps share
  one messages file, so splitting them duplicates the pull), verifies the findings, writes the
  report in the **Output format** below and calls `submit_report`. It never reads message text.
- **`research-subagent`** (subagent model) — runs steps 1–5 below: tool calls, `execute` math,
  the chain-divergence judgment. It returns FINDINGS (step 6), never a report — it has no
  `submit_report`. Its `task` tool offers only `extract-subagent` and `coding-subagent`; it never
  spawns another `research-subagent`.
- **`extract-subagent`** (utility model, the cheapest) — reads message TEXT from the offloaded file
  and returns distilled findings. The ONLY way to put work on the cheap model is
  `task(subagent_type="extract-subagent", description=...)`.
- **`coding-subagent`** — fixes a failed `execute` (pass the code and the exact error). One
  hand-off per failure; never retry shell-quoting variants yourself.

Rule of thumb: message **TEXT** (read, summarize, classify, quote) → `extract-subagent`. Message
**NUMBERS** (counts, shares, percentiles, price levels) → `execute` in your own context. Synthesis
and the chain-divergence judgment (step 5) stay with the research-subagent; the report stays with the
orchestrator.

The numeric recipes are one tested module, `recipes.py`, seeded into every sandbox session as
`/workspace/recipes.py`; `execute` imports it and never re-implements it. If the import fails, copy
`/skills/crowd-positioning/recipes.py` to `/workspace/recipes.py` with `read_file` + `write_file`
**once**; if it still fails, report the card figures as unavailable, use the stats block alone, and
say so in the findings. Never write your own version.

## Workflow (research-subagent, steps 1–6)

1. **Pull the window — one call.** `social_messages(asset, from_timestamp, to_timestamp)` returns
   `{stats, messages}`. **Read the stats block before anything else** — it carries every proportion
   you will cite (`total_matching`, `unique_after_dedup`, `sampled`, `by_source`, `volume_curve`,
   `sentiment_balance.by_bucket`, `trend_words`, `top_channels`). The sampled text only *explains*
   the stats; it never *counts*. Honor the strata: judge prevalence and mood ONLY from the `random`
   stratum and the stats block — `head`/`poles` are deliberately oversampled, use them for what
   spread and where the disagreement is. When large, the full result offloads to a file under
   `/workspace/data/` and you get a stub with the path. From then on the file's NUMBERS are yours
   (`execute`), its TEXT is read only by `extract-subagent`. Never re-call the tool to page the same
   window. **If `total_matching` is 0, stop here:** the finding is "no crowd data for this asset and
   window" — do not widen the window, try other sources, or substitute web search.

2. **Pull the baselines (signal 1).** With `fetch_metric_data`, get this coin's `social_volume_total`,
   `social_dominance_total`, `sentiment_balance_total` (or the sentiment metric
   `metrics_and_assets_discovery` resolves), and `price_usd` over a **trailing window ≥10× the
   spike window** (e.g. 90d for a 24h spike) ending at the spike's end, at the spike's interval. Each
   metric result is a saved file plus a computed summary; in `execute`, `json.load` the file and run
   `R.extreme(raw, spike_start, spike_end)` — the spike window's percentile rank and z-score against
   the rows before it (`spike_end` keeps a past spike from being diluted by what came after). If it
   returns `unbaselined`, pull a longer trailing window **once**; if still unbaselined, write
   "unbaselined (k points of history)" and move on. Keep the raw `price_usd` payload — it feeds
   `R.lead_lag` and `R.price_levels`.

3. **Run the card (signal 2 and the timing / mood reads)** in `execute` over the messages file —
   one call, no code of your own:
   ```python
   import sys; sys.path.insert(0, "/workspace"); import recipes as R
   d = R.load("/workspace/data/social_messages-<call_id>.json"); print(R.fmt(R.card(d)))
   ```
   It prints sanity checks, the dedup report (random stratum; exact → template → near-duplicate
   collapse; per-cluster accounts / rooms; top-channel concentration and trend from stats) with a
   deterministic organic / mixed / manufactured verdict and the rule that fired, account
   concentration and posting cadence, burst shape, mood (polarization, loud voices vs the crowd),
   link spam, question ratio and hour fingerprint. `copies` and `unique_after_dedup` are inputs to
   it, not the answer. Then `R.lead_lag(d["stats"]["volume_curve"], price_raw)` tells whether volume
   led or followed price; if it returns `unaligned`, re-fetch price at the bucket interval **once**,
   else report "no usable lead/lag". Quote the printed numbers; never retype a recipe and never ask
   `extract-subagent` whether posts are unique. How to read each block: `signals.md`.

4. **Discover the topics, claims, and price levels.** Two layers, two models:
   - *Numbers — your context, `execute`.* Start from the full-population `trend_words` — that IS
     the discovery: rank the dominant themes by their count/share of volume (the number carries
     it, the text only explains each). Group raw trend words into 3–5 candidate themes with their
     share. `R.price_levels(d["messages"], px)` clusters the price numbers people name into levels
     counted by distinct voices (a bot printing one price 300× is one voice) — keep the strongest
     1–2 below and 1–2 above the current price; the rest is noise you do not report.
   - *Text — utility model, `extract-subagent`.* Everything that requires READING messages — the
     one line of "what's actually being said" per theme, the concrete *checkable* claims (named
     actors, flows, events, targets) for signal 3, verbatim quotes, and where the crowd splits in
     `head`/`poles` — goes to `extract-subagent` via `task`. Never print message text into your own
     context. Send **exactly three** `task`s, in parallel: (a) themes + what's said, (b) checkable
     claims, (c) disagreement + quotes. Only for a monster file (over ~5,000 sampled messages)
     split a question by `source`, and never exceed six tasks in total. Do not re-ask a question
     because the answer was thin — a thin answer is the finding. Task template + the three
     questions are in `signals.md` § *Text extraction*. Exception: if the result arrived inline
     (small window, no file path), it is small enough to read directly.

5. **Test narrative against chain (signal 3).** For each checkable claim (at most five), pull the
   on-chain metric that would confirm or refute it (`fetch_metric_data` — exchange flows, supply on
   exchanges, whale txns, active addresses; map in `signals.md`) and label it **confirmed / diverges
   / unverifiable** with the number. For the 1–2 most consequential *factual* claims (a partnership,
   a listing, a hack), corroborate with **one** `web_search` each before repeating them as fact.

6. **Hand back findings** in the RETURN FORMAT from your instructions — one finding per readout
   line below (verdict inputs, extremeness, organic verdict, each claim's chain test, the price
   levels, the venues, the denominator), each with its numbers. Every finding's `source` is the
   data-source LABEL ("Santiment social messages", the metric source's label, a URL for web) — never
   a file path, file name, recipe or tool name. Findings never mention files, paths, "offloaded",
   or what remains unread; a question the text did not answer is one plain sentence in `gaps`.
   Then answer follow-ups from the data already pulled — re-slice the messages file (text questions
   → one more `extract-subagent` task, numeric ones → `execute`) or pull one more metric rather than
   re-running everything.

## Bounds (nothing here loops)

- One `social_messages` call per window; one optional prior-window call (`signals.md` § 7).
- One window inference at most; zero matches ends the analysis with "no crowd data".
- Recipe import fallback: once. Longer baseline for `unbaselined`: once. Price re-fetch for
  `unaligned`: once. After that, report the limitation in one clause.
- `extract-subagent`: three tasks; six absolute maximum. No re-asking.
- Chain checks: at most five claims; web corroboration: at most two searches.
- A failed `execute` goes to `coding-subagent` once; its output is final.

## Output format (orchestrator — a one-page positioning readout, not an article)

Lead with the verdict. Every bullet carries its denominator/baseline. Plain words throughout: say
"the unbiased sample", "trend words", "top channels" — never a field, stratum, function or file name.

- **POSITIONING VERDICT** — 1–2 lines: direction (long / short / split), conviction, *and the
  extremeness percentile*. "Crowd is aggressively long $X at a 94th-pct social spike; 68% organic,
  accelerating; chain confirms accumulation." This is the line they paid for — make it stand alone.
- **WHAT'S DRIVING IT** — the 3–5 dominant topics from the full-population trend words, each as
  *theme → share/count of volume → one line of what's being said*. This answers "what happened".
  Rank by share, not by how interesting the quote is. Flag any topic where the crowd splits (both
  sides, which has more voices), and add the mood line: N% bull / M% bear / K% neutral (split /
  consensus / apathetic), plus whether loud voices lean further than the crowd when that gap is
  ≥ 15 pts.
- **EXTREME?** — social volume Nth pct of trailing 90d (z=…); dominance Nth pct; sentiment skew
  N% bull vs M% trailing median. Spike vs baseline, always. Then the shape and timing: single-burst
  / multi-burst / ramp / plateau (peak bucket = N% of volume), and "volume led price by k buckets
  (r=…)" / "followed" / "no usable lead/lag".
- **ORGANIC?** — N% organic (near-duplicates collapsed; unbiased sample n=…, extrapolated to the
  total), biggest cluster = N posts by M accounts in K rooms (bot / room paste / viral copypasta),
  top account = N% of posts (scheduled cadence = bot), top-3 channels = N% of volume, promo domains
  if any, trend (rising / fading / flat). Verdict: organic / mixed / manufactured, naming the rule
  that fired.
- **NARRATIVE vs CHAIN** — one row per checkable claim (≤ 5): *claim → chain metric → confirmed /
  diverges (the number) / unverifiable*. Lead with any divergence — it's the highest-value finding.
- **CROWD PRICE LEVELS** — at most four lines: support @ L1 (N voices), current, target @ L2
  (M voices) — the strongest 1–2 levels each side, plus one clause for the rest ("nine smaller
  levels, none above 3 voices"). Never the full table of every level with its count.
- **WHERE IT'S HAPPENING** — the top 3 venues a reader can open: **twitter accounts**
  (`twitter.com/<screen_name>`) and **subreddits** (`reddit.com/r/<sub>`), plus any head-message
  link. Do **not** print telegram/discord channel IDs (raw numeric ids — unvisitable, meaningless to
  a reader); report those as an aggregate ("3 telegram channels ≈ 40% of volume") without the id.
  Internal social data has no URL — cite it as "Santiment social messages" per the citation rules.
- **The denominator, always:** "based on N sampled of M matching, sources: …" — plus any sanity
  FLAG the card printed (LOW-N, mismatched totals), in plain words.

Not in the report, ever: file paths, file names, "offloaded" / "saved" / "stored", recipe or
function names (`R.card`, `price_levels`), tool names, stratum or field names, a list of files or of
"what still needs extraction / validation", and any method narration ("computed over the sample
file"). A figure's provenance is its [n] citation, nothing else. The whole readout fits on one page:
if a section needs more than five lines, it is listing, not positioning.

## Discipline (the anti-journalism rules)

- No adjective without a number. "Spiking" → "94th pct". "Lots of" → "71%". "Some say" → "N voices".
- Key findings only. Three to five themes, one or two levels per side, three venues, at most five
  claims. Everything else is one aggregate clause or nothing. A ranked list of every value is
  transcription, and the reader did not ask for it.
- Never list a time series. No date/value tables and no per-hour/day/bucket lines, anywhere —
  report or message, any date format. A series is DESCRIBED (`R.describe`, `R.extreme`): span,
  first/last, min/max with when, mean, direction, percentile of the window. A metric result
  arrives as a saved file plus that summary; the file is for `execute`, not for quoting. Rows
  that reach the report are deleted before delivery, so listing them delivers nothing.
- Lead with divergence and extremeness; bury the recap. If everything is normal and organic and the
  chain agrees, say *that* in one line — a calm, confirmed read is also an answer.
- Honest denominator on every proportion; never estimate a share from the sampled text when the
  stats block has the full-population number.
- Every number comes off a recipe printout (`R.card`, `R.extreme`, `R.lead_lag`, `R.price_levels`)
  — never recomputed in prose, never from retyped code — and the printout's name stays with you: the
  report carries the number and its source, not the recipe.
- Analysis, not financial advice.
