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
  targets people actually name). Delivers a positioning VERDICT with denominators and source links,
  then answers follow-up questions on the same pulled data.
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
   pasting 200×. Use the dedup `copies` count, channel/author concentration, and the acceleration
   shape. Output a number: "70% organic, still accelerating" vs "bot campaign, 3 channels = 80%".
3. **Does the narrative match the chain?** — The crowd's *claims* are testable. "BlackRock
   accumulating" + chain shows $150M flowing **to** Coinbase = a divergence worth flagging. Pull
   each checkable claim's on-chain metric and report match or divergence, with the number.
4. **Where does the crowd put price?** — Price levels named in messages are positioning data.
   Extract them, cluster them, and show where the crowd places support / current / targets, with
   how many voices back each level. (E.g. "62772 was a trend word" → a level, not a topic.)

`signals.md` (next to this file) has the exact computation recipe and the claim→metric map. Read it.

## Workflow

1. **Pull the window.** Call `social_messages(asset, from_timestamp, to_timestamp)`. It returns
   `{stats, messages}`. **Read the stats block before anything else** — it carries every proportion
   you will cite (`total_matching`, `unique_after_dedup`, `sampled`, `by_source`, `volume_curve`,
   `sentiment_balance.by_bucket`, `trend_words`, `top_channels`). The sampled text only *explains*
   the stats; it never *counts*. Honor the strata: judge prevalence and mood ONLY from the `random`
   stratum and the stats block — `head`/`poles` are deliberately oversampled, use them for what
   spread and where the disagreement is. The full message set offloads to a file when large.

2. **Pull the baselines (signal 1).** With `fetch_metric_data`, get this coin's `social_volume_total`,
   `social_dominance_total`, a sentiment metric, and `price_usd` over a **trailing window ≥10× the
   spike window** (e.g. 90d for a 24h spike), at the spike's interval. If a metric name is uncertain,
   resolve it with `metrics_and_assets_discovery`. In `execute`, compute the spike window's
   percentile rank and z-score against that trailing distribution. This is signal 1.

3. **Compute organic-vs-manufactured (signal 2)** in `execute` over the messages file:
   organic share from `copies`, top-channel / top-author concentration, and the acceleration shape
   from `volume_curve`. Recipe in `signals.md`. Output a percentage and a one-word verdict.

4. **Extract the crowd's claims and price levels (signals 3 & 4).** From `trend_words` + the
   `head`/`poles` text, list the 3–6 concrete, *checkable* claims (named actors, flows, events,
   targets) and ignore vague mood. For monster windows, partition the messages file by `source` and
   spawn one `research-subagent` per slice (`task`) to pull claims + verbatim quotes in parallel,
   then merge. In `execute`, extract price numbers from message text and cluster them into levels
   (recipe in `signals.md`).

5. **Test narrative against chain (signal 3).** For each checkable claim, pull the on-chain metric
   that would confirm or refute it (`fetch_metric_data` — exchange flows, supply on exchanges, whale
   txns, active addresses; map in `signals.md`) and label it **confirmed / diverges / unverifiable**
   with the number. For the 1–2 most consequential *factual* claims (a partnership, a listing, a
   hack), corroborate with `web_search` before repeating them as fact.

6. **Deliver** via `submit_report` in the format below. Then answer follow-ups from the data already
   pulled — re-slice the messages file or pull one more metric rather than re-running everything.

Cheap-tier note: steps 3–4 (mechanical extraction over slices) are the map step — fine for
research-subagents on the utility model; keep the synthesis and the chain-divergence judgment (step
5) on your own context.

## Output format (positioning readout, not an article)

Lead with the verdict. Every bullet carries its denominator/baseline.

- **POSITIONING VERDICT** — 1–2 lines: direction (long / short / split), conviction, *and the
  extremeness percentile*. "Crowd is aggressively long $X at a 94th-pct social spike; 68% organic,
  accelerating; chain confirms accumulation." This is the line they paid for — make it stand alone.
- **EXTREME?** — social volume Nth pct of trailing 90d (z=…); dominance Nth pct; sentiment skew
  N% bull vs M% trailing median. Spike vs baseline, always.
- **ORGANIC?** — % organic (unique/total), top-3 channels = N% of volume, max copies ×N,
  acceleration (rising / peaked / fading). Verdict: organic / mixed / manufactured.
- **NARRATIVE vs CHAIN** — one row per checkable claim: *claim → chain metric → confirmed / diverges
  (the number) / unverifiable*. Lead with any divergence — it's the highest-value finding here.
- **CROWD PRICE LEVELS** — clustered: support @ L1 (N mentions), current, resistance/target @ L2
  (M mentions). Where the crowd places its line.
- **WHERE IT'S HAPPENING** — top channels/threads by volume with links (the `head` messages' `url`s
  and `top_channels`); this is the "where was it discussed most" deliverable. Internal social data
  has no URL — cite it as "Santiment social messages" per the citation rules.
- **The denominator, always:** "based on N sampled of M matching, sources: …".

## Discipline (the anti-journalism rules)

- No adjective without a number. "Spiking" → "94th pct". "Lots of" → "71%". "Some say" → "N voices".
- Lead with divergence and extremeness; bury the recap. If everything is normal and organic and the
  chain agrees, say *that* in one line — a calm, confirmed read is also an answer.
- Honest denominator on every proportion; never estimate a share from the sampled text when the
  stats block has the full-population number.
- Analysis, not financial advice.
