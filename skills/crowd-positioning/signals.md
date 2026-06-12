# Signal recipes

Concrete computations for the four signals. Adapt to the real column names in the
`social_messages` output (`stats` keys and message fields are listed in SKILL.md). Run these with
the `execute` tool over the offloaded messages file and the `fetch_metric_data` series — never
estimate in your head.

## 1 — Extreme vs history (percentile, not adjective)

The spike window value means nothing without the coin's own trailing distribution. Pull a trailing
series **≥10× the spike window** and rank the window against it.

```python
import pandas as pd
# s = trailing social_volume_total series from fetch_metric_data (datetime, value)
s = pd.DataFrame(series)                       # columns: datetime, value
window_val = s[s.datetime >= spike_start].value.mean()   # or .max() for the peak
base = s[s.datetime < spike_start].value
pct = (base < window_val).mean() * 100         # percentile rank vs trailing baseline
z   = (window_val - base.mean()) / base.std()  # how many sigma
```

Report `pct` and `z` for **social_volume_total**, **social_dominance_total**, and the **sentiment
skew** (bull% − bear%, ranked the same way against its own trailing series). "94th pct, z=2.7" is
the signal. A spike that's only 40th-pct is itself a finding — the "spike" isn't extreme.

## 2 — Organic vs manufactured

Same raw volume is 5k independent voices or one room pasting 200×. The `copies` field (dedup count
from the tool) plus concentration plus acceleration separate them.

```python
m = pd.DataFrame(messages)                     # the offloaded message file
total_posts   = int(m.copies.fillna(1).sum())  # raw posts incl. duplicates
unique_posts  = len(m)                          # distinct texts after dedup
organic_share = unique_posts / total_posts * 100        # % of volume that is NOT copy-paste
top3_channels = m.groupby('unit').copies.sum().sort_values(ascending=False).head(3).sum()
chan_conc     = top3_channels / total_posts * 100       # % of volume from top 3 rooms
max_copies    = int(m.copies.max())
# acceleration: slope / shape of stats['volume_curve']
vc = pd.DataFrame(stats['volume_curve'])        # t, count
trend = 'rising' if vc['count'].iloc[-1] > vc['count'].iloc[:-1].mean() else 'peaked/fading'
```

Verdict heuristics (tune on real data, state the numbers regardless):
- **organic**: organic_share ≳ 60%, top-3 channels ≲ 40%, copies tail thin.
- **manufactured / coordinated push**: organic_share low, or a single text with `copies` in the
  hundreds, or top-3 channels ≳ 70% of volume. Say which one triggered the call.
- Pair with acceleration: "70% organic, still accelerating" vs "bot campaign, peaked 6h ago".

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
| "partnership / listing / hack" (factual event) | `web_search` to corroborate, not on-chain | no credible source = rumor |

For each: label **confirmed** (chain agrees), **diverges** (chain contradicts — quote both numbers),
or **unverifiable** (no on-chain proxy; mark it, don't fake confirmation). Resolve uncertain metric
names with `metrics_and_assets_discovery` first.

## 4 — Crowd price levels

Price numbers in messages are positioning data — extract, sanity-filter to a band around the live
price (drop phone numbers, years, market caps), cluster, rank by mentions.

```python
import re, pandas as pd
PRICE = re.compile(r'(?<![\w.])\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?[kK]?(?![\w%])')
def to_num(x):
    x = x.replace(',', '').lstrip('$')
    return float(x[:-1]) * 1000 if x[-1] in 'kK' else float(x)
levels = []
for _, r in m.iterrows():
    for tok in PRICE.findall(str(r.text)):
        v = to_num(tok)
        if 0.2 * px <= v <= 5 * px:            # px = current price; keep plausible levels only
            levels.append(v)
lv = pd.Series(levels)
# cluster: round to a sensible bin (~0.5–1% of price), count mentions per bin
binned = (lv / (px * 0.01)).round() * (px * 0.01)
clusters = binned.value_counts().sort_index()  # level -> mention count
```

Then label relative to the live price: clusters **below** = crowd support, **above** = resistance /
targets. Report the top few by mention count: "support @ 60.5k (41 mentions), target @ 72k (28)".
A `trend_words` entry that is a bare number (e.g. "62772") is a level — fold it in.

## Notes

- Strata discipline (from SKILL.md) applies to every count: prevalence and mood from the `random`
  stratum + stats block only; `head`/`poles` for spread and the extremes of disagreement.
- Always carry the denominator: "N sampled of M matching".
- If a baseline pull fails (metric missing, short history), say the signal is unbaselined rather
  than presenting the raw number as if it were extreme.
