# Signal recipes

Concrete computations for the four signals. Run them with the `execute` tool over the offloaded
messages file and the `fetch_metric_data` series — never estimate in your head. **The stats block
carries every full-population proportion; the sampled `messages` only explain it.** When in doubt
about a tool's exact field names, inspect the first element before computing
(`print(series[:1])`, `m.columns`).

## 1 — Extreme vs history (percentile, not adjective)

The spike value means nothing without a baseline. Two baselines, both worth reporting:

**(a) Temporal — vs the coin's own past.** Pull a trailing series **≥10× the spike window** and
rank the window against it. Be schema-tolerant and normalize timestamps on BOTH sides, or the
compare throws.

```python
import pandas as pd

def to_series(raw):  # raw = fetch_metric_data list; tolerate datetime/dt/d, value/v
    df = pd.DataFrame(raw)
    dcol = next((c for c in ("datetime", "dt", "d", "time", "t") if c in df.columns), df.columns[0])
    vcol = next((c for c in ("value", "v", "val") if c in df.columns), df.columns[-1])
    out = pd.DataFrame({"datetime": pd.to_datetime(df[dcol], utc=True, errors="coerce"),
                        "value": pd.to_numeric(df[vcol], errors="coerce")}).dropna()
    return out.sort_values("datetime")

def extreme(raw, spike_start):
    s = to_series(raw)
    cut = pd.to_datetime(spike_start, utc=True)
    win, base = s[s.datetime >= cut].value, s[s.datetime < cut].value
    if len(win) == 0 or len(base) < 3:        # too little history to baseline
        return {"unbaselined": True, "n_base": len(base)}
    wv, sd = win.mean(), base.std()           # .max() instead of .mean() for the peak
    pct = float((base < wv).mean() * 100)
    z = float((wv - base.mean()) / sd) if sd and sd > 0 else None
    return {"pct": round(pct), "z": None if z is None else round(z, 2)}
```

Report `pct` (and `z`) for **social_volume_total**, **social_dominance_total**, and the **sentiment
skew** (bull% − bear%, ranked against its own trailing series). "94th pct, z=2.7" is the signal. A
spike that's only 40th-pct is itself a finding — it isn't extreme.

**(b) Cross-sectional — vs other coins right now.** A coin can be loud against its own history yet
ordinary across the market (or vice-versa). Use `assets_by_metric` on `social_volume_total` /
`social_dominance_total` to get the current ranking and report where this coin sits ("3rd of all
assets by social volume today"). Temporal + cross-sectional together = the honest extremeness read.

If a baseline pull fails (metric missing, short history), say the signal is **unbaselined** rather
than presenting the raw number as if it were extreme.

## 2 — Organic vs manufactured

**Compute the shares from the STATS block, not the sample.** The server dedups over the FULL
population; the sample only carries a subset of unique texts, so `len(sample)/sum(copies)` is NOT
the population ratio (it can read 85% organic when the truth is 17%). Use the full-population
counts:

```python
def organic(stats, m):
    tm, ud = stats.get("total_matching"), stats.get("unique_after_dedup")
    organic_share = round(ud / tm * 100) if tm and ud else None      # % of volume NOT copy-paste
    tc = stats.get("top_channels") or []
    top3 = sum(c.get("count", 0) for c in tc[:3])
    chan_conc = round(top3 / tm * 100) if tm and top3 else None       # % of volume from top 3 rooms
    # `copies` lives on the sample; guard — the column may be absent on some windows
    copies = m["copies"].fillna(1) if "copies" in m.columns else pd.Series(1, index=m.index)
    max_copies = int(copies.max()) if len(copies) else 1              # loudest single pasted text
    vc = pd.DataFrame(stats.get("volume_curve") or [])
    if len(vc) >= 2:                                                  # accel: last third vs first third
        n = max(1, len(vc) // 3)
        trend = "rising" if vc["count"].tail(n).mean() > vc["count"].head(n).mean() else "peaked/fading"
    else:
        trend = "unknown"
    return dict(organic_share=organic_share, chan_conc=chan_conc, max_copies=max_copies, trend=trend)
```

Verdict heuristics (tune on real data, state the numbers regardless):
- **organic**: organic_share ≳ 60%, top-3 channels ≲ 40%, `max_copies` small.
- **manufactured / coordinated push**: organic_share low, OR a single text with `max_copies` in the
  hundreds, OR top-3 channels ≳ 70% of volume. Say which one triggered the call.
- Pair with acceleration: "70% organic, still accelerating" vs "bot campaign, peaked early".

## 3 — Narrative vs chain (claim → metric map)

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

## 4 — Crowd price levels

Price numbers in messages are positioning data — extract, sanity-filter to a band around the live
price, cluster, rank by mentions. The earlier `\d{1,3}(,\d{3})*` pattern silently missed bare
prices like `62772` / `$68000` (no commas — the common form); this one catches them and still
rejects `100%` / `5x`.

```python
import re
PRICE = re.compile(r'(?<![\w.])\$?\d[\d,]*(?:\.\d+)?[kK]?(?![\w%])')

def to_num(x):
    x = x.replace(",", "").lstrip("$")
    return float(x[:-1]) * 1000 if x and x[-1] in "kK" else float(x)

def price_levels(texts, px):                 # px = current price from fetch_metric_data price_usd
    votes = []
    for t in texts:                          # count each level ONCE per message (a bot pasting a
        seen = set()                         # target 200× is one voice, not 200 — independent voices)
        for tok in PRICE.findall(str(t)):
            try:
                v = to_num(tok)
            except ValueError:
                continue
            if 0.2 * px <= v <= 5 * px:       # plausible band; drops phone numbers, market caps, years
                seen.add(round(v / (px * 0.01)) * (px * 0.01))   # bin to ~1% of price
        votes += list(seen)
    return pd.Series(votes).value_counts()    # level -> # of messages naming it, by mention count
```

Everything is derived from `px` (the live price you pulled from `fetch_metric_data price_usd`) — do
not hardcode a price or a band; the `0.2×–5×` window and `~1%` bin are relative heuristics, tune
them per coin's volatility, never bake in a number. Label relative to `px`: clusters **below** =
crowd support, **above** = resistance / targets. Report the top few by mention count in the form
"support @ <level> (<N> msgs), target @ <level> (<M> msgs)" with the actual extracted levels. A
`trend_words` entry that is a bare number is a level — fold it in. If a value inside the band looks
like a year, sanity-check it isn't a date before reporting it as a level.

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
  still uses every `unit`; only the *named, linked* ones are filtered to linkable sources.
- Strata discipline (from SKILL.md): prevalence and mood from the `random` stratum + stats block
  only; `head`/`poles` for spread and the extremes of disagreement.
- Always carry the denominator: "N sampled of M matching".
