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
  happened") and delivers a positioning VERDICT with denominators and source links, then answers
  follow-up questions on the same pulled data.
---

# Crowd positioning

You are **positioning the crowd, not reporting on it.** The failure mode this skill exists to kill
is *journalism with the data* — a readable recap of what people said. Nobody pays for the recap.
They pay for the answer to one question:

> **How is the crowd positioned, and is that positioning extreme?**

Every line you emit is a **signal with a baseline**, never a story. Not "lots of bullish ETF talk"
— that's journalism. Instead: "social volume is 94th percentile of trailing 90d; 71% bullish vs a
53% trailing-30d median; 68% organic and still accelerating; chain confirms it (–$210M net off
exchanges this window)." If a sentence has no number and no comparison, cut it.

## When to use

A token is trending or its social dominance/volume spiked (now or a named past window) and the user
wants to understand it — "what happened with $X around <date>", "why is everyone talking about Y",
"is this real or bots", "is the crowd long here". You have a token and (ideally) a time window. If
the window is missing, infer the spike window first (`trending_stories` / `fetch_metric_data` on
social volume to find the peak) and **state the window you chose**.

## The four signals (this is the whole product)

1. **Extreme vs history?** — A spike is meaningless without its own baseline. "12th percentile of
   trailing 90d" is a signal; "high social activity" is not. Rank this window's social volume,
   dominance, and sentiment skew against the coin's trailing distribution.
2. **Organic or manufactured?** — Same volume means opposite things if it's 5k people or one room
   pasting 200×. "Organic" is a DEFINED, computed number: run `R.card(d)` from `recipes.py`
   (pre-loaded in the sandbox; collapses exact, template and near-duplicate posts over the random
   stratum, sizes each cluster by accounts and rooms, and adds account concentration, posting
   cadence, link spam and hour fingerprint). Never eyeball uniqueness from the text and never read it off
   `copies` or `unique_after_dedup` alone — those see only byte-identical copies, so a price bot
   changing one number per post scores 100% unique. Output the number with its basis: "62% organic
   (near-duplicates collapsed), still accelerating" vs "bot campaign: one account = 31% of posts".
3. **Does the narrative match the chain?** — The crowd's *claims* are testable. "BlackRock
   accumulating" + chain shows $150M flowing **to** Coinbase = a divergence worth flagging. Pull
   each checkable claim's on-chain metric and report match or divergence, with the number.
4. **Where does the crowd put price?** — Price levels named in messages are positioning data.
   Extract them, cluster them, and show where the crowd places support / current / targets, with
   how many voices back each level. (E.g. "62772 was a trend word" → a level, not a topic.)

The computations live in `recipes.py` (next to this file), seeded into the sandbox as
`/workspace/recipes.py`. `signals.md` (also next to this file) says how to call each recipe, how to
read its output, and holds the claim→metric map. Read `signals.md`; never retype recipe code.

## Workflow

1. **Pull the window.** Call `social_messages(asset, from_timestamp, to_timestamp)`. It returns
   `{stats, messages}`. **Read the stats block before anything else** — it carries every proportion
   you will cite (`total_matching`, `unique_after_dedup`, `sampled`, `by_source`, `volume_curve`,
   `sentiment_balance.by_bucket`, `trend_words`, `top_channels`). The sampled text only *explains*
   the stats; it never *counts*. Honor the strata: judge prevalence and mood ONLY from the `random`
   stratum and the stats block — `head`/`poles` are deliberately oversampled, use them for what
   spread and where the disagreement is. When large, the full result offloads to a file under
   `/workspace/data/` and you get a stub with the path. From then on the file's NUMBERS are yours
   (`execute`), its TEXT is read only by `extract-subagent` (see **Model routing** below).

2. **Pull the baselines (signal 1).** With `fetch_metric_data`, get this coin's `social_volume_total`,
   `social_dominance_total`, a sentiment metric, and `price_usd` over a **trailing window ≥10× the
   spike window** (e.g. 90d for a 24h spike), at the spike's interval. If a metric name is uncertain,
   resolve it with `metrics_and_assets_discovery`. In `execute`, `R.extreme(raw_series, spike_start)`
   gives the spike window's percentile rank and z-score against that trailing distribution; each
   metric result is a saved file plus a computed summary, so `json.load` the file and pass it in.
   Keep the raw `price_usd` payload — it feeds `R.lead_lag` and `R.price_levels`. This is signal 1.

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
   led or followed price. Quote the printed numbers; never retype a recipe and never ask
   `extract-subagent` whether posts are unique. How to read each block: `signals.md`.

4. **Discover the topics, claims, and price levels.** Two layers, two models:
   - *Numbers — your context, `execute`.* Start from the full-population `trend_words` — that IS
     the discovery: rank the dominant themes by their count/share of volume (the number carries
     it, the text only explains each). Group raw trend words into 3–6 candidate themes with their
     share. `R.price_levels(d["messages"], px)` clusters the price numbers people name into levels
     counted by distinct voices (a bot printing one price 300× is one voice) — a handful of numbers,
     so this stays in `execute`.
   - *Text — utility model, `extract-subagent`.* Everything that requires READING messages — the
     one line of "what's actually being said" per theme, the concrete *checkable* claims (named
     actors, flows, events, targets) for signal 3, verbatim quotes, and where the crowd splits in
     `head`/`poles` — goes to `extract-subagent` via `task`. Never print message text into your own
     context. Send one `task` per question, in parallel: (a) themes + what's said, (b) checkable
     claims, (c) disagreement + quotes. For monster windows additionally split each question by
     `source` (one task per slice) and merge. Task template + the three questions are in
     `signals.md` § *Text extraction*. Exception: if the result arrived inline (small window, no
     file path), it is small enough to read directly.

5. **Test narrative against chain (signal 3).** For each checkable claim, pull the on-chain metric
   that would confirm or refute it (`fetch_metric_data` — exchange flows, supply on exchanges, whale
   txns, active addresses; map in `signals.md`) and label it **confirmed / diverges / unverifiable**
   with the number. For the 1–2 most consequential *factual* claims (a partnership, a listing, a
   hack), corroborate with `web_search` before repeating them as fact.

6. **Deliver** via `submit_report` in the format below. Then answer follow-ups from the data already
   pulled — re-slice the messages file (text questions → another `extract-subagent` task, numeric
   ones → `execute`) or pull one more metric rather than re-running everything.

## Model routing (who runs on what)

Three model slots are fixed by config (`MODEL_TIERS`), not by you. You pick the model only by picking
the subagent:

- orchestrator → research model: plans, verifies, synthesizes, `submit_report`;
- `research-subagent` → subagent model: this skill's steps 1–6 — tool calls, `execute` math, judgment;
- `extract-subagent` → **utility model, the cheapest**: reads big text from an offloaded file and
  returns findings. The ONLY way to put work on the cheap model is
  `task(subagent_type="extract-subagent", description=...)`.

Rule of thumb: message **TEXT** (read, summarize, classify, quote) → `extract-subagent`. Message
**NUMBERS** (counts, shares, percentiles, price levels) → `execute` in your own context. Synthesis and
the chain-divergence judgment (step 5) stay with you. Do not slice by spawning `research-subagent`s:
this skill runs inside one, and its `task` tool offers only `extract-subagent`.

The numeric recipes are one tested module, `recipes.py`, seeded into every sandbox session as
`/workspace/recipes.py`; `execute` imports it and never re-implements it. If the import fails, copy
`/skills/crowd-positioning/recipes.py` to `/workspace/recipes.py` with `read_file` + `write_file`.

## Output format (positioning readout, not an article)

Lead with the verdict. Every bullet carries its denominator/baseline.

- **POSITIONING VERDICT** — 1–2 lines: direction (long / short / split), conviction, *and the
  extremeness percentile*. "Crowd is aggressively long $X at a 94th-pct social spike; 68% organic,
  accelerating; chain confirms accumulation." This is the line they paid for — make it stand alone.
- **WHAT'S DRIVING IT** — the 3–6 dominant topics from the full-population `trend_words`, each as
  *theme → share/count of volume → one line of what's being said*. This answers "what happened"; it
  is the discovery layer, so it leads the body. Keep it signal-flavored — every theme carries its
  number; rank by share, not by how interesting the quote is. Flag any topic where the crowd splits
  (both sides, which has more voices) using the `head`/`poles` strata, and add the card's mood line:
  N% bull / M% bear / K% neutral (split / consensus / apathetic), plus whether loud voices lean
  further than the crowd when that gap is ≥ 15 pts.
- **EXTREME?** — social volume Nth pct of trailing 90d (z=…); dominance Nth pct; sentiment skew
  N% bull vs M% trailing median. Spike vs baseline, always. Then the shape and timing from the card:
  single-burst / multi-burst / ramp / plateau (peak bucket = N% of volume), and "volume led price by
  k buckets (r=…)" / "followed" / "no usable lead/lag".
- **ORGANIC?** — N% organic (near-duplicates collapsed; random stratum n=…, extrapolated to the
  total), biggest cluster = N posts by M accounts in K rooms (bot / room paste / viral copypasta),
  top account = N% of posts (scheduled cadence = bot), top-3 channels = N% of volume, promo domains
  if any, trend (rising / fading / flat). Verdict: organic / mixed / manufactured, naming the rule
  that fired.
- **NARRATIVE vs CHAIN** — one row per checkable claim: *claim → chain metric → confirmed / diverges
  (the number) / unverifiable*. Lead with any divergence — it's the highest-value finding here.
- **CROWD PRICE LEVELS** — clustered: support @ L1 (N mentions), current, resistance/target @ L2
  (M mentions). Where the crowd places its line.
- **WHERE IT'S HAPPENING** — the "where was it discussed most" deliverable. Name only sources a
  reader can open: **twitter accounts** (`twitter.com/<screen_name>`) and **subreddits**
  (`reddit.com/r/<sub>`), plus any `head`-message `url`s. Do **not** print telegram/discord channel
  IDs (raw numeric `chat_id`s — unvisitable, meaningless to a reader); report those as an aggregate
  ("3 telegram channels ≈ 40% of volume") without the id. Internal social data has no URL — cite it
  as "Santiment social messages" per the citation rules.
- **The denominator, always:** "based on N sampled of M matching, sources: …" — plus any sanity
  FLAG the card printed (LOW-N, mismatched totals).

## Discipline (the anti-journalism rules)

- No adjective without a number. "Spiking" → "94th pct". "Lots of" → "71%". "Some say" → "N voices".
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
  — never recomputed in prose, never from retyped code.
- Analysis, not financial advice.
