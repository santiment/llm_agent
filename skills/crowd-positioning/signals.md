# Signal recipes — how to compute, how to read

All numeric work lives in **`recipes.py`**, pre-loaded into the sandbox as `/workspace/recipes.py`
(source `/skills/crowd-positioning/recipes.py`; stdlib only, covered by tests). Call it — never
retype it, never estimate in your head, never `print` message text. Anything that needs *reading*
text is a `task` for `extract-subagent` (template in § *Text extraction*). **The stats block carries
every full-population proportion; the sampled `messages` only explain it.**

```python
import sys; sys.path.insert(0, "/workspace"); import recipes as R
PATH = "/workspace/data/social_messages-<call_id>.json"   # path from the tool's offload stub
d = R.load(PATH)                                           # {"stats": ..., "messages": [...]}
print(R.fmt(R.card(d)))                                    # every local recipe, one run
```

- `print(R.index())` lists the recipes one line each; `help(R.lead_lag)` shows one signature.
- Recipes return numbers and labels only; the longest text they emit is an 80-char cluster
  fingerprint. Message text → utility model.
- If `import recipes` fails (seed missing): `read_file /skills/crowd-positioning/recipes.py`, then
  `write_file /workspace/recipes.py` with that content **unchanged**, import again, and say so in
  the findings. Do not write your own version.
- Card numbers go into the report verbatim: `*_share` / `*_pct` are already percentages, `n*` are
  counts. Every percentage travels with its n.

## 0 — The card (run first, read top to bottom)

| block | holds | feeds |
|---|---|---|
| `sanity` | `OK` / `FLAG` / `NOTE` lines on the payload | quote every FLAG in the denominator line; `LOW-N` → say "small sample" next to every share |
| `population` | total_matching, sampled, random_n, sources % | denominator line: "N matching, n sampled, k random" |
| `organic` | dedup shares, context, `verdict` + `rule` | ORGANIC? — the verdict word and the rule as printed |
| `top_clusters` | repeated clusters: posts, share, users, channels, sources, kind, fingerprint | name the 1–3 biggest by fingerprint + kind |
| `accounts` | top1 / top10 share, gini, cross-room, cadence, flags | ORGANIC? — bot / paid-room evidence |
| `burst` | shape, peak_share, half_life, bursts, trend | EXTREME? — *how* it spiked |
| `mood` | polarization + loud-vs-many | WHAT'S DRIVING IT — one line |
| `links`, `questions`, `hours` | promo domains, question %, hour fingerprint | ORGANIC? / WHAT'S DRIVING IT — only when notable |

Thresholds are applied inside the recipes; your job is to say which rule fired and what it means for
positioning.

## 1 — Extreme vs history (percentile, not adjective) — `R.extreme`

The spike value means nothing without a baseline. Two baselines, both worth reporting.

**(a) Temporal — vs the coin's own past.** Pull a trailing series **≥10× the spike window** at the
spike's interval and rank the window against it:

```python
import json                         # a metric result arrives as a saved file + a computed summary
vol_raw = json.load(open("/workspace/data/<tool>-<id>.json"))   # social_volume_total, trailing >=10x the window
px_raw  = json.load(open("/workspace/data/<tool>-<id2>.json"))  # price_usd, same range and interval; reused in §3 and §6
print(R.describe(vol_raw))                          # n, span, first/last, min/max (when), mean, median, direction
print(R.extreme(vol_raw, SPIKE_START))              # {pct, z, window, base_mean, base_median, n_window, n_base}
print(R.extreme(vol_raw, SPIKE_START, agg="max"))   # the peak instead of the window mean
```

- **Never list a series.** No date/value table, no per-bucket lines — not in the report, not in
  a message, in any date format; rows that reach the report are deleted on delivery. A series is
  *described*: the summary the tool hands you, `R.describe`, or `R.extreme`. The file is for
  `execute`, never for quoting.

- `pct` = share of baseline points below the window value; `z` = (window − base_mean) / base_sd.
  Report both: "top 2% of trailing 90d (z 3.1)". A 40th-pct spike is itself a finding — not extreme.
- Run it for **social_volume_total**, **social_dominance_total**, and the **sentiment skew** (bull% −
  bear% per bucket from `stats.sentiment_balance`, ranked against a trailing sentiment series).
- `{unbaselined: True, n_base: k}` → write "unbaselined (k points of history)", never a percentile.
- `R.to_series(raw)` gives `[(ts, value)]` from any `fetch_metric_data` shape; never rewrite parsers.

**(b) Cross-sectional — vs other coins right now.** `assets_by_metric` on `social_volume_total` /
`social_dominance_total`: "3rd of all assets by social volume today". Loud against own history yet
ordinary across the market (or the reverse) is the honest read — report both.

## 2 — Organic vs manufactured — card `organic`, `top_clusters`, `accounts`, `links`, `hours`

"Organic" is a DEFINED number; two models given the same file land on the same share. The payload's
own fields are NOT it:

- `stats.unique_after_dedup` — distinct EXACT text over the population. A bot changing one number,
  link or emoji per post is 100% "unique". Upper bound on organic, nothing more.
- `copies` on a message — exact duplicates collapsed INSIDE THE SAMPLE. `copies == 1` everywhere
  means no two rows were byte-identical, not that every row is a distinct voice.

The recipe: normalize (lower-case; strip URLs, @handles, punctuation, emoji; numbers → `0`) →
exact clusters → near clusters (bigram Jaccard ≥ 0.5, ≥ 6 tokens) → size each cluster by posts,
accounts, rooms, sources. Random stratum only (head/poles are oversampled by design), rows weighted
by `copies`, shares extrapolate to `total_matching`.

Reading `organic`:

| field | meaning |
|---|---|
| `template_dup_share` | % of random posts in a normalized-exact cluster of size ≥ 2 — bots, copypasta, price feeds |
| `near_dup_share` | % in near clusters — paraphrased pushes, retweet chains; gap vs template > 10 pts = paraphrased spam |
| `organic_share` | 100 − near_dup_share. **THE number.** "62% organic (near-duplicates collapsed; random n=1,830, extrapolated to 9,400 matching)" |
| `biggest_cluster_share` | one message's share of all random posts |
| `exact_unique_share` | the payload's exact-dedup figure, quoted only as the upper bound ("98% exact-unique, yet 41% template/near duplicates") |
| `chan_conc` | top-3 channels' % of `total_matching` (population, not sample) |
| `trend` | volume_curve last third vs first third: rising / fading / flat |
| `verdict`, `rule` | organic / mixed / manufactured and the rule that fired — quote both |

Verdict rules (`R.organic_verdict`): **manufactured** if organic ≤ 30%, or one cluster from ≤ 3
accounts holds ≥ 20% of posts, or chan_conc ≥ 70%; **organic** if organic ≥ 60%, chan_conc ≤ 40%
and biggest cluster < 5%; else **mixed**, with the reasons. Pair with `trend`: "62% organic, still
rising" vs "bot campaign, peaked early".

Cluster `kind`: `single-account bot` (1 user) · `room paste / coordinated push` (≤ 3 users or ≤ 2
rooms, ≥ 100 posts or ≥ 5%) · `viral copypasta (one message, many people)` (≥ 20 users, ≥ 5 rooms —
real enthusiasm, but ONE message: counts once for prevalence and themes) · `repeated`. An empty
`top_clusters` means nothing repeats.

Reading `accounts`: `top1_share` ≥ 20% or `top10_share` ≥ 50% = a few accounts *are* the crowd
(`flags` says so). `cadence: scheduled` (gap CV < 0.35 over ≥ 5 gaps; `median_gap_min` shown) = bot;
`bursty` = human. High `cross_room_share` = the same accounts re-posting across rooms. Large
`unknown_user_posts` → account reads are weak; say so.

Reading `links`: `promo_domains` = domains pushed by ≤ 3 accounts with ≥ 20 posts or ≥ 5% — name
them. `link_only_pct` = link drops with no words. `t.co` is Twitter's shortener on every tweet link,
not a promo domain by itself.

Reading `hours`: `flat` on ≥ 50 posts = round-the-clock posting (humans are `diurnal`). For one
suspect account: `R.hour_fingerprint(d["messages"], users={"<user>"})`.

## 3 — Timing — card `burst`, plus `R.lead_lag`

`burst.shape`: `single-burst` (one bucket ≥ 35% of volume, or a ≥ 4× peak fading within 2 buckets) =
news / listing / liquidation shock · `burst-then-fade` = event already digested · `multi-burst` =
repeated catalysts or a campaign re-firing · `ramp` = building interest, the pre-move pattern ·
`plateau` / `sustained` = ongoing conversation. `time_to_peak_pct` near 100 = the spike is still live
at the window's end.

```python
print(R.lead_lag(d["stats"]["volume_curve"], px_raw, max_lag=6))   # price at the buckets' interval
```

- `best_lag > 0` → volume **leads** price by k buckets (crowd positioned before the move);
  `< 0` → volume **follows** price (crowd reacting — late); `0` → same bucket.
- `|best_corr| < 0.3` or `n_pairs < 8` → "no usable lead/lag". Say it; do not pick a sign.
  `abs_return_corr_lag0` high while `corr_lag0` is low = volume tracks volatility, not direction.
- `{unaligned: True}` = buckets carry no timestamps and lengths differ — fetch price at the bucket
  interval and retry.

## 4 — Narrative vs chain (claim → metric map)

The crowd's *claims* are testable; its *mood* is not. List the concrete, checkable claims, then pull
the on-chain metric that confirms or refutes each. Lead the report with any divergence.

| Crowd claim (pattern) | Pull this metric (`fetch_metric_data`) | Divergence looks like |
|---|---|---|
| "whales / BlackRock / X accumulating" | `exchange_outflow` / `exchange_balance` (falling = accumulation) | claim "accumulating" but net **inflow** to exchanges |
| "everyone dumping / exit liquidity" | `exchange_inflow`, `supply_on_exchanges` | claim "dumping" but supply on exchanges flat/falling |
| "whales buying the dip" | `whale_transaction_count_100k_usd`, large-holder balances | spike in whale txns absent |
| "adoption / usage exploding" | `active_addresses_24h`, `network_growth` | flat addresses, no new wallets |
| "supply shock / coins locked" | `supply_on_exchanges`, staked/locked supply | supply on exchanges rising |
| "partnership / listing / hack" (factual event) | `web_search` + `fetch_insights` to corroborate, not on-chain | no credible source = rumor |

For each: label **confirmed** (chain agrees), **diverges** (chain contradicts — quote both numbers),
or **unverifiable** (no on-chain proxy; mark it, don't fake confirmation). Resolve uncertain metric
names with `metrics_and_assets_discovery` first.

## 5 — Mood — card `mood`

`polarization` (random stratum; lean = bull − bear, |lean| ≥ 0.2 counts, ≥ 0.5 is `strong_*`):
`split` (split_index ≥ 0.7) = a two-sided fight · `consensus bullish` / `consensus bearish` (one
side ≥ 2× the other) = a one-sided crowd · `apathetic` (neutral ≥ 70%) = volume without opinion —
listing or airdrop chatter, not positioning · `leaning …` otherwise.

`loud_vs_many`: mean lean per stratum plus engagement-weighted. `loud_minus_crowd_lean_pts` ≥ 15 →
"loud voices are MORE bullish than the crowd" (influencers pushing into a neutral base) or MORE
bearish (fear from the top the crowd hasn't followed). `engagement_top1pct_share` = the share of all
engagement taken by the top 1% of posts.

## 6 — Crowd price levels — `R.price_levels`

```python
px = R.to_series(px_raw)[-1][1]                          # live price = last point of the series
for lvl in R.price_levels(d["messages"], px): print(lvl)  # {level, voices, msgs, side}
```

- Counts **voices** (distinct users, random stratum): a bot printing "$61,832" 300 times is one
  voice. `msgs` sits next to it — when msgs ≫ voices, say "one feed".
- Everything derives from `px`: the `0.2×–5×` band drops years, percentages and tx counts; 1% bins;
  `level` = median of one value per voice. Tune band / bin per coin's volatility, never hardcode.
- `side: below` = where the crowd buys the dip, `above` = targets. Both dense with a `split` mood =
  the two-sided battle map. Report "support @ L (N voices), target @ M (K voices)".
- A bare-number `trend_words` entry (e.g. "62k") is a level only if it falls inside the band; check
  a candidate isn't a year or a % target before reporting it.

## 7 — Venue and novelty (optional second call)

A prior-window `social_messages` call with `max_words=2000` is cheap: its `stats` block is
full-population regardless of sample size.

```python
print(R.source_shift(d["stats"], prior["stats"]["by_source"]))      # shares, hhi, share_delta_pts, largest_excess
print(R.word_novelty(d["stats"]["trend_words"], prior["stats"]["trend_words"]))
```

- `largest_excess` = the source whose share grew most vs the baseline ("telegram +23 pts"): a move
  into rooms = coordinated retail; into Reddit / Twitter = broader interest. `hhi` > 0.5 = one venue
  dominates.
- `novelty_share` = share of current trend words absent before; `new` is the reading brief for the
  utility model; `persistent` with high `x` = an old story re-igniting; `faded` = what the crowd
  dropped.

## Text extraction (utility model via `extract-subagent`)

Reading the message text is the token-heavy part of this skill, so it runs on the cheapest model.
Do not open the text in your own context (no printing of text fields, no slicing text columns). Send
one `task(subagent_type="extract-subagent", description=...)` per question, in parallel. The reader
sees NOTHING but your description, so make it self-contained:

```
FILE: /workspace/data/social_messages-<call_id>.json — JSON object; the posts are the list under
  key "messages", each with the post text plus `source`, `stratum`, `copies`, `unit`, `url`
  (inspect the first element for the exact text field). Read in bounded slices, never the whole
  file at once.
SOURCE LABEL: Santiment social messages
CONTEXT: <coin>, <window>. Candidate themes from trend_words: <theme -> words -> share of volume>.
QUESTION: <one of the three below>
RULES: judge prevalence and mood ONLY from stratum == "random"; use head/poles for the spread and
  the disagreement. Every item carries the number of messages backing it and 1-2 verbatim quotes
  (with `url` when present). A text repeated many times (same words, different numbers/links) is
  ONE voice: count it once and say it repeats. Drop vague mood. No financial advice.
```

The three questions (one task each; for monster windows add `only source == "<source>"` and run one
task per source, then merge):

- **Themes** — for each candidate theme, one line of what is actually being said and how many
  `random`-stratum messages back it; list any theme present in the text but missing from the
  candidates. Hand over the card's biggest cluster fingerprints and `word_novelty.new` as extra
  candidates.
- **Checkable claims** — concrete claims with a named actor / flow / event / target ("X
  accumulating", "listing on Y", "hack", "target 1.20"), verbatim, with message counts. These feed
  the signal-4 table.
- **Disagreement** — where the crowd splits (bull case vs bear case), which side has more voices,
  and the best quote for each side.

Fold the returned findings into yours: a theme carries the `trend_words` share (yours) plus the
one-liner and quote (theirs); claims go to the narrative-vs-chain table; the split goes to WHAT'S
DRIVING IT next to the `polarization` numbers. Cite all of it as "Santiment social messages".

## Corroborate & visualize (optional tools)

- **`trending_stories` / `combined_trends`** — confirm the spike is a real, captured trend and
  cross-check your `trend_words`; the stories also give **linkable source URLs** for the report.
- **`assets_by_metric`** — the cross-sectional baseline for signal 1 (above).
- **`show_chart`** (if the Santiment MCP exposes it) — render the social-volume-vs-price overlay so
  the report carries visual evidence of the spike and how price reacted.

## Notes

- **Name only sources a reader can open.** When you report channels/authors (top channels, loudest
  voices, "where it's happening"), name only **twitter accounts** (`twitter.com/<screen_name>`) and
  **subreddits** (`reddit.com/r/<sub>`) — things with a real link. Do **not** print telegram/discord
  channel IDs (raw numeric `chat_id`s — unvisitable and meaningless to a reader); report their
  activity as an aggregate instead ("3 telegram channels drove ~40% of volume"). Concentration math
  still uses every `unit`; only the *named, linked* ones are filtered to linkable sources. The same
  goes for `top_accounts[].user` — name it only when it is a twitter handle or a subreddit.
- Strata discipline (from SKILL.md): prevalence and mood from the `random` stratum + stats block
  only; `head`/`poles` for spread and the extremes of disagreement.
- Always carry the denominator: "N sampled of M matching", plus any sanity FLAG.
