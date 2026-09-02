"""Deterministic recipes for the crowd-positioning skill.

This file is SEEDED into every sandbox session as ``/workspace/recipes.py`` (see
``agent.skill_seed_files`` and ``sandbox.HttpSandboxBackend``), so the model calls tested
functions instead of retyping code out of ``signals.md``:

    import sys; sys.path.insert(0, "/workspace"); import recipes as R
    d = R.load(PATH); print(R.fmt(R.card(d)))          # every local recipe, one call
    print(R.extreme(raw_series, "2026-09-01T00:00:00Z"))  # recipes that need a pulled series

Design rules:
  - stdlib only — the sandbox image and the test venv need nothing extra;
  - inputs are the plain dicts/lists the tools return (`social_messages` -> {stats, messages};
    `fetch_metric_data` -> a list of {datetime, value}-ish rows), tolerant of field-name drift;
  - outputs are SMALL dicts of numbers. No recipe ever returns or prints message text beyond an
    80-char cluster fingerprint — text is the utility model's job, not this file's;
  - prevalence is judged on the ``random`` stratum only (unbiased draw), each row weighted by
    its server-side ``copies``; ``head``/``poles`` are oversampled on purpose and used only for
    the loud-vs-many read.

Tested on synthetic data in tests/test_recipes.py.
"""

from __future__ import annotations

import json
import math
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean, median, pstdev

__all__ = [
    "load", "fmt", "index", "parse_ts",
    "to_series", "describe", "extreme",
    "norm", "near_dup_clusters", "dedup_report", "context", "organic_verdict",
    "price_levels",
    "account_concentration", "burst_shape", "lead_lag",
    "loud_vs_many", "polarization",
    "source_shift", "word_novelty", "link_spam", "hour_fingerprint", "question_ratio",
    "sanity", "card",
]

# ----------------------------------------------------------------------------- helpers ----

def load(path_or_obj):
    """Read the offloaded `social_messages` file (or accept the parsed object) -> {stats, messages}."""
    d = json.load(open(path_or_obj)) if isinstance(path_or_obj, str) else path_or_obj
    if isinstance(d, list):
        return {"stats": {}, "messages": d}
    if not isinstance(d, dict) or "messages" not in d:
        raise ValueError("expected a {stats, messages} object or a list of messages")
    d.setdefault("stats", {})
    return d


def index():
    """One line per recipe: name — what it answers. Cheaper than help(R)."""
    lines = []
    for name in __all__:
        fn = globals()[name]
        doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        lines.append(f"{name} — {doc}")
    return "\n".join(lines)


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _w(m):
    """Row weight = server-side exact copies (>=1)."""
    v = _f(m.get("copies"))
    return max(1, int(v)) if v else 1


def _msgs(d):
    return d["messages"] if isinstance(d, dict) else d


def _random(msgs):
    """The unbiased stratum. Rows without a stratum field count as random."""
    return [m for m in msgs if (m.get("stratum") or "random") == "random"]


def _pct(a, b, digits=0):
    if not b:
        return None
    v = 100.0 * a / b
    return round(v) if digits == 0 else round(v, digits)


_ISO_FIX = re.compile(r"(\.\d{3})\d+")  # trim sub-millisecond digits fromisoformat may reject


def parse_ts(x):
    """Tolerant timestamp -> aware UTC datetime, or None. Accepts epoch s/ms, ISO 8601
    (with Z / offset / fractional seconds), or a bare date."""
    if x is None:
        return None
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    if isinstance(x, (int, float)):
        if not math.isfinite(x):
            return None
        secs = x / 1000.0 if abs(x) > 1e11 else float(x)
        try:
            return datetime.fromtimestamp(secs, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    s = str(x).strip()
    if not s:
        return None
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return parse_ts(float(s))
    s = s.replace("Z", "+00:00").replace("z", "+00:00")
    s = _ISO_FIX.sub(r"\1", s)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _curve(volume_curve):
    """volume_curve -> [(ts or None, count)] in given order (buckets are already ordered)."""
    out = []
    for b in volume_curve or []:
        if isinstance(b, dict):
            t = next((b[k] for k in ("t", "datetime", "dt", "time", "d") if k in b), None)
            c = next((b[k] for k in ("count", "value", "v", "n") if k in b), None)
        elif isinstance(b, (list, tuple)) and len(b) >= 2:
            t, c = b[0], b[1]
        else:
            continue
        c = _f(c)
        if c is not None:
            out.append((parse_ts(t), c))
    return out


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _gini(counts):
    xs = sorted(c for c in counts if c > 0)
    n = len(xs)
    if n < 2:
        return 0.0
    s = sum(xs)
    return round((2 * sum((i + 1) * x for i, x in enumerate(xs)) / (n * s)) - (n + 1) / n, 2)


def _one(v):
    if isinstance(v, dict):
        return ", ".join(f"{k}={_one(x)}" for k, x in v.items()) if v else "-"
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_one(x) for x in v) + "]"
    return json.dumps(v, default=str, ensure_ascii=False) if isinstance(v, str) else str(v)


def fmt(obj, _ind=0):
    """Compact, readable rendering of a recipe result (nested dicts -> indented lines)."""
    pad = "  " * _ind
    out = []
    if not isinstance(obj, dict):
        return pad + _one(obj)
    for k, v in obj.items():
        if isinstance(v, dict) and (len(v) > 5 or any(isinstance(x, (dict, list)) for x in v.values())):
            out.append(f"{pad}{k}:")
            out.append(fmt(v, _ind + 1))
        elif isinstance(v, list) and v and isinstance(v[0], (dict, list, tuple)):
            out.append(f"{pad}{k}:")
            out.extend(f"{pad}  - {_one(x)}" for x in v)
        elif isinstance(v, list) and v and isinstance(v[0], str) and len(v) > 3:
            out.append(f"{pad}{k}:")
            out.extend(f"{pad}  - {x}" for x in v)
        else:
            out.append(f"{pad}{k}: {_one(v)}")
    return "\n".join(out)


# ----------------------------------------------------------- 1. extreme vs history -------

def to_series(raw, slug=None):
    """Any fetch_metric_data shape -> sorted [(utc datetime, float)]. Tolerates datetime/dt/d/
    time/t + value/v/val keys, [ts, value] pairs, a {data: [...]} wrapper, the metric server's
    {data: {slug: [...]}} wrapper (one slug, or pick one with slug=...), or a {ts: value} map."""
    if isinstance(raw, dict):
        for k in ("data", "rows", "values", "result", "series", "timeseriesData"):
            inner = raw.get(k)
            if isinstance(inner, list):
                raw = inner
                break
            if isinstance(inner, dict) and inner and all(isinstance(v, list) for v in inner.values()):
                if slug is not None:
                    raw = inner[slug]
                elif len(inner) == 1:
                    raw = next(iter(inner.values()))
                else:
                    raise ValueError(f"several slugs {sorted(inner)}: pass slug=<one of them>")
                break
        else:
            raw = [(k, v) for k, v in raw.items()]
    out = []
    for r in raw or []:
        if isinstance(r, dict):
            t = next((r[c] for c in ("datetime", "dt", "d", "time", "t", "timestamp") if c in r), None)
            v = next((r[c] for c in ("value", "v", "val", "y") if c in r), None)
        elif isinstance(r, (list, tuple)) and len(r) >= 2:
            t, v = r[0], r[1]
        else:
            continue
        ts, fv = parse_ts(t), _f(v)
        if ts is not None and fv is not None:
            out.append((ts, fv))
    out.sort(key=lambda p: p[0])
    return out


def describe(raw, slug=None):
    """Series summary — the ONLY form a metric series takes in a report: n, span, first/last
    (+change %), min/max with when, mean, median, direction (last third vs first third, ±10%).
    Never print the rows themselves."""
    s = to_series(raw, slug)
    if len(s) < 2:
        return {"n": len(s), "note": "too few points"}
    vals = [v for _, v in s]
    n = len(vals)
    imin = min(range(n), key=vals.__getitem__)
    imax = max(range(n), key=vals.__getitem__)
    third = max(1, n // 3)
    head, tail = mean(vals[:third]), mean(vals[-third:])
    rel = (tail - head) / abs(head) if head else 0.0
    when = lambda i: s[i][0].strftime("%Y-%m-%dT%H:%M")
    d = {"n": n, "start": when(0), "end": when(n - 1), "first": vals[0], "last": vals[-1],
         "change_pct": round((vals[-1] - vals[0]) / abs(vals[0]) * 100, 1) if vals[0] else None,
         "min": vals[imin], "min_at": when(imin), "max": vals[imax], "max_at": when(imax),
         "mean": mean(vals), "median": median(vals),
         "direction": "rising" if rel > 0.1 else "falling" if rel < -0.1 else "flat"}
    if n >= 3 and vals[-1] == 0 and median(vals[:-1]) != 0:
        d["note"] = "last point is 0 — likely an incomplete current bucket; drop it or re-fetch"
    return d


def extreme(raw, spike_start, agg="mean"):
    """Rank the spike window (rows at/after spike_start) against the trailing baseline of the
    same series: percentile + z-score. agg="max" ranks the window's peak instead of its mean."""
    cut = parse_ts(spike_start)
    if cut is None:
        raise ValueError(f"unparseable spike_start: {spike_start!r}")
    s = to_series(raw)
    win = [v for t, v in s if t >= cut]
    base = [v for t, v in s if t < cut]
    if not win or len(base) < 3:
        return {"unbaselined": True, "n_window": len(win), "n_base": len(base)}
    wv = max(win) if agg == "max" else mean(win)
    bm, sd = mean(base), pstdev(base)
    pct = 100.0 * sum(1 for b in base if b < wv) / len(base)
    z = (wv - bm) / sd if sd > 0 else None
    return {"pct": round(pct), "z": None if z is None else round(z, 2), "window": round(wv, 4),
            "base_mean": round(bm, 4), "base_median": round(median(base), 4),
            "n_window": len(win), "n_base": len(base), "agg": agg}


# ------------------------------------------------------ 2. organic vs manufactured -------

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
# Standalone numbers only: not preceded by a letter OR digit (so "web3", "l2", "w123" stay words —
# a letter-only lookbehind would still match "23" inside "w123") and not followed by a word char.
_NUM = re.compile(r"(?<![a-z0-9])\d[\d,.:]*[%kmbx]?(?![a-z0-9])")
_JUNK = re.compile(r"[^a-z0-9 ]")
_WS = re.compile(r"\s+")


def norm(t):
    """Template key: case, URLs, @handles, numbers, punctuation and emoji do NOT make a post unique."""
    t = _MENTION.sub(" ", _URL.sub(" ", str(t or "").lower()))
    t = _NUM.sub(" 0 ", t)
    return _WS.sub(" ", _JUNK.sub(" ", t)).strip()


def _bigrams(tokens):
    return set(zip(tokens, tokens[1:])) if len(tokens) > 1 else {tuple(tokens)}


def near_dup_clusters(keys, thr=0.5, min_tokens=6):
    """Cluster id per normalized text. Texts sharing a bigram are compared to that bigram's first
    holder; Jaccard(bigrams) >= thr links them (union-find). Linear in total bigrams. Texts shorter
    than min_tokens link only when identical."""
    toks = [k.split() for k in keys]
    grams = [_bigrams(t) if len(t) >= min_tokens else set() for t in toks]
    parent = list(range(len(keys)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first = {}
    for i, g in enumerate(grams):
        for bg in g:
            j = first.setdefault(bg, i)
            if j != i and find(i) != find(j):
                if len(grams[i] & grams[j]) / len(grams[i] | grams[j]) >= thr:
                    parent[find(i)] = find(j)
    return [find(i) for i in range(len(keys))]


def dedup_report(msgs, top=5):
    """Organic share: exact -> template -> near-duplicate collapse over the random stratum
    (weighted by copies); each surviving cluster sized by posts, accounts, rooms, sources."""
    msgs = _msgs(msgs)
    rows = []
    for m in _random(msgs):
        k = norm(m.get("text"))
        if k:
            rows.append((k, _w(m), m.get("user"), m.get("unit"), m.get("source")))
    posts = sum(r[1] for r in rows)
    if not posts:
        return {"random_posts": 0, "exact_clusters": 0, "near_clusters": 0,
                "template_dup_share": None, "near_dup_share": None, "organic_share": None,
                "biggest_cluster_share": None, "top_clusters": []}
    keys = list(dict.fromkeys(r[0] for r in rows))
    kc = dict(zip(keys, near_dup_clusters(keys)))
    agg: dict = {}
    for k, w, u, unit, src in rows:
        c = agg.setdefault(kc[k], {"posts": 0, "users": set(), "channels": set(),
                                   "sources": set(), "fingerprint": k[:80]})
        c["posts"] += w
        if u is not None:
            c["users"].add(u)
        if unit is not None:
            c["channels"].add(unit)
        if src is not None:
            c["sources"].add(src)
    g = sorted(agg.values(), key=lambda c: -c["posts"])
    # Only REPEATED clusters are worth naming; an empty list means nothing repeats.
    tops = [{"posts": c["posts"], "share": _pct(c["posts"], posts), "users": len(c["users"]),
             "channels": len(c["channels"]), "sources": len(c["sources"]),
             "kind": _cluster_kind(c["posts"], len(c["users"]), len(c["channels"]), posts),
             "fingerprint": c["fingerprint"]} for c in g[:top] if c["posts"] >= 2]
    return {
        "random_posts": posts,
        "exact_clusters": len(keys),                       # template duplicates collapsed
        "near_clusters": len(g),                           # + paraphrase / variant duplicates
        "template_dup_share": round(100 * (1 - len(keys) / posts)),
        "near_dup_share": round(100 * (1 - len(g) / posts)),
        "organic_share": round(100 * len(g) / posts),      # THE number to report
        "biggest_cluster_share": _pct(g[0]["posts"], posts),
        "top_clusters": tops,
    }


def _cluster_kind(posts, users, channels, total):
    if posts < 3:
        return "single post"
    if users == 1:
        return "single-account bot"
    if (users <= 3 or channels <= 2) and (posts >= 100 or (total and posts / total >= 0.05)):
        return "room paste / coordinated push"
    if users >= 20 and channels >= 5:
        return "viral copypasta (one message, many people)"
    return "repeated"


def context(stats):
    """Population-side context from the stats block: exact-text upper bound, top-3 room
    concentration, first-third vs last-third acceleration."""
    stats = stats or {}
    tm, ud = _f(stats.get("total_matching")), _f(stats.get("unique_after_dedup"))
    tc = stats.get("top_channels") or []
    top3 = sum(_f(c.get("count")) or 0 for c in tc[:3] if isinstance(c, dict))
    counts = [c for _, c in _curve(stats.get("volume_curve"))]
    if len(counts) >= 2:
        n = max(1, len(counts) // 3)
        first, last = mean(counts[:n]), mean(counts[-n:])
        trend = "rising" if last > first * 1.2 else "fading" if last < first * 0.8 else "flat"
    else:
        trend = "unknown"
    return {"exact_unique_share": _pct(ud, tm) if ud is not None else None,  # upper bound, NOT organic
            "chan_conc": _pct(top3, tm) if top3 else None,                     # % volume from top-3 rooms
            "trend": trend}


def organic_verdict(rep, ctx):
    """Apply the skill's verdict thresholds to dedup_report + context -> (verdict, rule that fired)."""
    o, big = rep.get("organic_share"), rep.get("biggest_cluster_share") or 0
    cc = (ctx or {}).get("chan_conc")
    small_cluster = next((c for c in rep.get("top_clusters", [])
                          if c["users"] <= 3 and (c["share"] or 0) >= 20), None)
    if o is None:
        return "unknown", "no random-stratum posts to score"
    if o <= 30:
        return "manufactured", f"organic_share {o}% <= 30%"
    if small_cluster:
        return "manufactured", (f"one cluster from {small_cluster['users']} account(s) holds "
                                f"{small_cluster['share']}% of posts (>= 20%)")
    if cc is not None and cc >= 70:
        return "manufactured", f"top-3 channels carry {cc}% of volume (>= 70%)"
    if o >= 60 and (cc is None or cc <= 40) and big < 5:
        return "organic", f"organic_share {o}% >= 60%, top-3 channels {cc}% <= 40%, biggest cluster {big}% < 5%"
    why = []
    if o < 60:
        why.append(f"organic_share {o}% in 30-60%")
    if cc is not None and cc > 40:
        why.append(f"top-3 channels {cc}% > 40%")
    if big >= 5:
        why.append(f"biggest cluster {big}% >= 5%")
    return "mixed", "; ".join(why) or "between thresholds"


# ------------------------------------------------------------- 4. crowd price levels -----

_PRICE = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?[kK]?(?![\w%])")


def _to_num(x):
    x = x.replace(",", "").lstrip("$")
    return float(x[:-1]) * 1000 if x and x[-1] in "kK" else float(x)


def price_levels(msgs_or_texts, px, band=(0.2, 5.0), bin_pct=1.0, top=8):
    """Price levels the crowd names: numbers within band*px, binned to ~bin_pct% of px, ranked by
    VOICES = distinct accounts naming the level (a price bot printing 300 quotes is one voice).
    Given message dicts, uses the random stratum and `user`; given bare texts, one voice per
    text. -> [{level, voices, msgs, side}]; level = median of what those voices typed."""
    px = _f(px)
    if not px or px <= 0:
        raise ValueError("px must be the live price (> 0)")
    step = px * bin_pct / 100.0
    voters: dict = defaultdict(dict)          # bin -> {voter: first value typed}
    msgs_in: dict = defaultdict(int)          # bin -> messages naming it
    for i, item in enumerate(msgs_or_texts):
        if isinstance(item, dict):
            if (item.get("stratum") or "random") != "random":
                continue
            text, voter = item.get("text"), item.get("user")
            voter = f"#{i}" if voter in (None, "") else voter
        else:
            text, voter = item, f"#{i}"
        seen: dict = {}
        for tok in _PRICE.findall(str(text or "")):
            try:
                v = _to_num(tok)
            except ValueError:
                continue
            if band[0] * px <= v <= band[1] * px:
                seen.setdefault(round(v / step), v)
        for b, v in seen.items():
            voters[b].setdefault(voter, v)
            msgs_in[b] += 1
    digits = max(0, 3 - int(math.floor(math.log10(step)))) if step < 1000 else 0
    out = []
    for b, vs in sorted(voters.items(), key=lambda kv: (-len(kv[1]), -msgs_in[kv[0]]))[:top]:
        lvl = median(vs.values())             # what the voices typed, not the bin edge
        side = "at" if abs(lvl - px) < step else ("below" if lvl < px else "above")
        out.append({"level": round(lvl, digits) if digits else int(round(lvl)),
                    "voices": len(vs), "msgs": msgs_in[b], "side": side})
    return out


# ------------------------------------------------------- accounts, timing, mood ----------

def account_concentration(msgs, top=5):
    """Who is posting: distinct accounts, top-1/top-10 share, Gini, cross-room posters, and the
    posting cadence of the loudest accounts (a low gap CV = scheduled = bot)."""
    msgs = _msgs(msgs)
    by_user: Counter = Counter()
    rooms, srcs, times = defaultdict(set), defaultdict(set), defaultdict(list)
    unknown = 0
    for m in _random(msgs):
        u = m.get("user")
        w = _w(m)
        if u is None or u == "":
            unknown += w
            continue
        by_user[u] += w
        if m.get("unit") is not None:
            rooms[u].add(m.get("unit"))
        if m.get("source") is not None:
            srcs[u].add(m.get("source"))
        ts = parse_ts(m.get("ts") or m.get("datetime") or m.get("timestamp"))
        if ts is not None:
            times[u].append(ts)
    posts = sum(by_user.values())
    if not posts:
        return {"posts": 0, "accounts": 0, "note": "no user field on random rows"}
    ranked = by_user.most_common()
    counts = [c for _, c in ranked]
    cross = sum(1 for u in by_user if len(rooms[u]) >= 2)
    tops = []
    for u, c in ranked[:top]:
        if c < 3 and tops:                     # one-off posters are not "top accounts"
            break
        ts = sorted(times[u])
        gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:]) if b > a]
        if len(gaps) >= 5:
            mg, cv = median(gaps), (pstdev(gaps) / mean(gaps) if mean(gaps) else None)
            cadence = "scheduled" if cv is not None and cv < 0.35 else "bursty"
        else:
            mg, cv, cadence = None, None, "n/a"
        tops.append({"user": u, "posts": c, "share": _pct(c, posts), "rooms": len(rooms[u]),
                     "sources": sorted(srcs[u]), "median_gap_min": None if mg is None else round(mg / 60, 1),
                     "gap_cv": None if cv is None else round(cv, 2), "cadence": cadence})
    flags = []
    if tops and tops[0]["share"] >= 20:
        flags.append(f"one account = {tops[0]['share']}% of posts (>= 20%)")
    top10 = _pct(sum(counts[:10]), posts)
    if top10 >= 50:
        flags.append(f"top-10 accounts = {top10}% of posts (>= 50%)")
    sched = [t["user"] for t in tops if t["cadence"] == "scheduled"]
    if sched:
        flags.append(f"{len(sched)} of the top-{len(tops)} accounts post on a fixed schedule")
    return {"posts": posts, "accounts": len(by_user), "posts_per_account": round(posts / len(by_user), 1),
            "top1_share": tops[0]["share"] if tops else None, "top10_share": top10,
            "gini": _gini(counts), "cross_room_accounts": cross,
            "cross_room_share": _pct(cross, len(by_user)), "unknown_user_posts": unknown,
            "top_accounts": tops, "flags": flags}


def burst_shape(volume_curve):
    """Shape of the spike from the stats volume_curve: peak bucket, peak/mean, share of volume in
    the peak bucket, when the peak came, half-life after it, number of bursts, trend and a label
    (plateau / ramp / single-burst / burst-then-fade / multi-burst / sustained)."""
    pts = _curve(volume_curve)
    counts = [c for _, c in pts]
    n = len(counts)
    if n < 3:
        return {"insufficient": True, "n_buckets": n}
    total = sum(counts)
    if total <= 0:
        return {"insufficient": True, "n_buckets": n, "total": 0}
    peak_i = max(range(n), key=counts.__getitem__)
    peak, mean_c = counts[peak_i], total / n
    third = max(1, n // 3)
    first, last = mean(counts[:third]), mean(counts[-third:])
    trend = "rising" if last > first * 1.2 else "fading" if last < first * 0.8 else "flat"
    half = None
    for k in range(peak_i + 1, n):
        if counts[k] <= peak / 2:
            half = k - peak_i
            break
    bursts = sum(1 for i in range(n)
                 if counts[i] >= 2 * mean_c
                 and (i == 0 or counts[i] > counts[i - 1])
                 and (i == n - 1 or counts[i] >= counts[i + 1]))
    pom, pshare = peak / mean_c, 100.0 * peak / total
    # A steady ramp peaks late but never far above its mean (< 2x for a linear rise); a spike
    # at the very end does — so the pom guard keeps an end-burst out of "ramp".
    if peak_i >= n - third and last > first * 1.5 and pom < 3:
        shape = "ramp"
    elif pom < 1.8:
        shape = "plateau"
    elif bursts >= 2:
        shape = "multi-burst"
    elif pshare >= 35 or (pom >= 4 and half is not None and half <= 2):
        shape = "single-burst"
    elif half is not None:
        shape = "burst-then-fade"
    else:
        shape = "sustained"
    t_peak = pts[peak_i][0]
    return {"n_buckets": n, "total": int(total), "peak": int(peak),
            "peak_t": t_peak.isoformat() if t_peak else str(peak_i),
            "peak_over_mean": round(pom, 1), "peak_share": round(pshare),
            "time_to_peak_pct": round(100.0 * peak_i / (n - 1)), "half_life_buckets": half,
            "bursts": bursts, "trend": trend, "shape": shape}


def lead_lag(volume_curve, price_raw, max_lag=6, min_pairs=8):
    """Did chatter lead price or chase it? Pearson correlation of bucket volume with the price
    return `lag` buckets later, for lag in [-max_lag, max_lag]. lag > 0 = volume leads price;
    lag < 0 = volume follows price. Also the volume-vs-|return| coupling at lag 0."""
    vol = _curve(volume_curve)
    price = to_series(price_raw)
    if len(vol) < 3 or len(price) < 3:
        return {"insufficient": True, "n_volume": len(vol), "n_price": len(price)}
    if all(t is not None for t, _ in vol):
        pts_t = [t for t, _ in price]
        aligned = []
        for t, c in vol:
            i = bisect_right(pts_t, t) - 1        # last price at or before the bucket
            if i >= 0:
                aligned.append((c, price[i][1]))
        mode = "timestamp"
    elif len(vol) == len(price):
        aligned = [(c, p) for (_, c), (_, p) in zip(vol, price)]
        mode = "index"
    else:
        return {"unaligned": True, "why": "volume buckets lack timestamps and lengths differ"}
    v = [c for c, _ in aligned]
    p = [x for _, x in aligned]
    rets = [(p[i] - p[i - 1]) / p[i - 1] if p[i - 1] else 0.0 for i in range(1, len(p))]
    v = v[1:]                                  # v[i] and rets[i] now share bucket i
    n = len(v)
    if n < min_pairs:
        return {"insufficient": True, "n_pairs": n, "min_pairs": min_pairs}
    by_lag = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xs, ys = v[:n - lag] if lag else v, rets[lag:]
        else:
            xs, ys = v[-lag:], rets[:n + lag]
        if len(xs) >= min_pairs:
            r = _pearson(xs, ys)
            if r is not None:
                by_lag[lag] = round(r, 2)
    if not by_lag:
        return {"insufficient": True, "n_pairs": n}
    best = max(by_lag, key=lambda L: abs(by_lag[L]))
    r0 = by_lag.get(0)
    abs_r0 = _pearson(v, [abs(x) for x in rets])
    if abs(by_lag[best]) < 0.3:
        read = f"no usable lead/lag (max |r|={abs(by_lag[best]):.2f} < 0.3)"
    elif best > 0:
        read = f"volume LEADS price by {best} bucket(s) (r={by_lag[best]:+.2f})"
    elif best < 0:
        read = f"volume FOLLOWS price by {-best} bucket(s) (r={by_lag[best]:+.2f}) — chatter is reacting"
    else:
        read = f"volume and price move together in the same bucket (r={by_lag[best]:+.2f})"
    return {"n_pairs": n, "alignment": mode, "best_lag": best, "best_corr": by_lag[best],
            "corr_lag0": r0, "abs_return_corr_lag0": None if abs_r0 is None else round(abs_r0, 2),
            "by_lag": by_lag, "read": read}


def _lean(m):
    b, r = _f(m.get("bull")), _f(m.get("bear"))
    if b is None and r is None:
        return None
    return max(-1.0, min(1.0, (b or 0.0) - (r or 0.0)))


def _mood(rows, weight):
    """rows: (lean, w). -> shares with |lean| thresholds 0.2 (lean) and 0.5 (strong)."""
    n = sum(w for _, w in rows)
    if not n:
        return {"n": 0}
    bull = sum(w for l, w in rows if l >= 0.2)
    bear = sum(w for l, w in rows if l <= -0.2)
    return {"n": int(n) if weight == "copies" else round(n, 1),
            "bull_pct": _pct(bull, n), "bear_pct": _pct(bear, n), "neutral_pct": _pct(n - bull - bear, n),
            "strong_bull_pct": _pct(sum(w for l, w in rows if l >= 0.5), n),
            "strong_bear_pct": _pct(sum(w for l, w in rows if l <= -0.5), n),
            "mean_lean": round(sum(l * w for l, w in rows) / n, 2)}


def loud_vs_many(msgs):
    """Mood of the loud (head stratum, engagement-weighted) vs the many (random stratum), from
    per-post bull/bear scores. A gap >= 15 pts of lean means influencers and crowd disagree."""
    msgs = _msgs(msgs)
    strata = defaultdict(list)
    eng_rows, eng_all = [], []
    for m in msgs:
        l = _lean(m)
        s = m.get("stratum") or "random"
        e = _f(m.get("engagement"))
        if e is not None and e >= 0:
            eng_all.append(e)
        if l is None:
            continue
        strata[s].append((l, _w(m)))
        if e is not None and e > 0:
            eng_rows.append((l, e))
    out = {s: _mood(rows, "copies") for s, rows in strata.items()}
    if not out:
        return {"note": "no bull/bear scores on messages"}
    ew = _mood(eng_rows, "engagement") if eng_rows else {"n": 0}
    crowd = out.get("random", {})
    loud = out.get("head") or (ew if ew.get("n") else {})
    gap = None
    if crowd.get("n") and loud.get("n"):
        gap = round((loud["mean_lean"] - crowd["mean_lean"]) * 100)
    top1 = None
    if eng_all and sum(eng_all) > 0:
        k = max(1, math.ceil(len(eng_all) * 0.01))
        top1 = _pct(sum(sorted(eng_all, reverse=True)[:k]), sum(eng_all))
    if gap is None:
        read = "loud-vs-many not measurable (missing head stratum / engagement or scores)"
    elif abs(gap) < 15:
        read = f"loud and crowd agree (lean gap {gap:+d} pts)"
    else:
        read = (f"loud voices are {'MORE bullish' if gap > 0 else 'MORE bearish'} than the crowd "
                f"by {abs(gap)} pts of lean")
    return {"strata": out, "engagement_weighted": ew, "loud_minus_crowd_lean_pts": gap,
            "engagement_top1pct_share": top1, "read": read}


def polarization(msgs):
    """Consensus or split? Random-stratum bull/bear/neutral shares and a split index
    (1 = perfectly even split among leaning posts, 0 = one-sided), with a label."""
    rows = [(l, _w(m)) for m in _random(_msgs(msgs)) if (l := _lean(m)) is not None]
    mood = _mood(rows, "copies")
    if not mood.get("n"):
        return {"n": 0, "note": "no bull/bear scores on random rows"}
    b, r, neu = mood["bull_pct"], mood["bear_pct"], mood["neutral_pct"]
    split = round(1 - abs(b - r) / (b + r), 2) if (b + r) else None
    if neu >= 70:
        label = "apathetic (>= 70% neutral)"
    elif split is not None and split >= 0.7:
        label = "split"
    elif b >= 2 * max(r, 1):
        label = "consensus bullish"
    elif r >= 2 * max(b, 1):
        label = "consensus bearish"
    else:
        label = "leaning bullish" if b > r else "leaning bearish"
    return {**mood, "split_index": split, "label": label}


# ------------------------------------------------------ venue, novelty, links, hours -----

def _source_counts(x):
    if isinstance(x, dict):
        return {str(k): _f(v) or 0 for k, v in x.items()}
    out = {}
    for r in x or []:
        if isinstance(r, dict):
            s = r.get("source") or r.get("name") or r.get("key")
            c = _f(r.get("count") if "count" in r else r.get("value"))
            if s is not None and c is not None:
                out[str(s)] = out.get(str(s), 0) + c
    return out


def source_shift(stats, baseline=None):
    """Which venue carries the spike: per-source shares + concentration, and (given a baseline
    {source: count-or-share} for a trailing window) the source with the largest excess."""
    now = _source_counts((stats or {}).get("by_source"))
    total = sum(now.values())
    if not total:
        return {"note": "no by_source counts"}
    shares = {s: round(100.0 * c / total) for s, c in sorted(now.items(), key=lambda kv: -kv[1])}
    top_s = next(iter(shares))
    hhi = round(sum((c / total) ** 2 for c in now.values()), 2)
    out = {"shares": shares, "top_source": top_s, "top_share": shares[top_s], "hhi": hhi,
           "read": ("broad-based across venues" if shares[top_s] <= 50
                    else f"{shares[top_s]}% of the volume is on {top_s}")}
    base = _source_counts(baseline) if baseline else {}
    bt = sum(base.values())
    if bt:
        bshare = {s: 100.0 * c / bt for s, c in base.items()}
        delta = {s: round(shares[s] - bshare.get(s, 0.0)) for s in shares}
        for s, sh in bshare.items():
            delta.setdefault(s, round(-sh))
        exc = max(delta, key=delta.get)
        out.update({"baseline_shares": {s: round(v) for s, v in bshare.items()},
                    "share_delta_pts": delta, "largest_excess": exc,
                    "read": out["read"] + f"; {exc} is {delta[exc]:+d} pts vs its baseline share"})
    return out


def _words(x):
    out: dict = {}
    if isinstance(x, dict):
        for w, c in x.items():
            out[str(w).lower()] = _f(c) or 0
        return out
    for r in x or []:
        if isinstance(r, str):
            out[r.lower()] = out.get(r.lower(), 0) + 1
        elif isinstance(r, dict):
            w = r.get("word") or r.get("term") or r.get("key")
            c = _f(r.get("count") if "count" in r else r.get("score"))
            if w is not None:
                out[str(w).lower()] = (c if c is not None else 1)
    return out


def word_novelty(now_words, before_words, top=10):
    """New narrative or the same one? trend_words now vs a prior window: words that are new,
    persistent (with count ratio) and faded, plus the share of today's word volume that is new."""
    now, before = _words(now_words), _words(before_words)
    new = sorted(((w, c) for w, c in now.items() if w not in before), key=lambda x: -x[1])
    keep = sorted(((w, now[w], before[w]) for w in now if w in before), key=lambda x: -x[1])
    gone = sorted(((w, c) for w, c in before.items() if w not in now), key=lambda x: -x[1])
    tot = sum(now.values())
    return {"n_now": len(now), "n_before": len(before),
            "novelty_share": _pct(sum(c for _, c in new), tot) if tot else None,
            "new": [{"word": w, "count": c} for w, c in new[:top]],
            "persistent": [{"word": w, "now": c, "before": b,
                            "x": round(c / b, 1) if b else None} for w, c, b in keep[:top]],
            "faded": [{"word": w, "count": c} for w, c in gone[:top]]}


_DOMAIN = re.compile(r"https?://(?:www\.)?([^/\s:?#]+)", re.I)


def link_spam(msgs, top=8):
    """Promo pressure: share of random posts carrying links, share that are link-only, top
    domains by posts and by distinct accounts; domains pushed by <= 3 accounts are flagged.
    (t.co is Twitter's own shortener — a Twitter-heavy sample makes it big by itself.)"""
    msgs = _msgs(msgs)
    n = linked = link_only = 0
    dom_posts: Counter = Counter()
    dom_users = defaultdict(set)
    for m in _random(msgs):
        w = _w(m)
        t = str(m.get("text") or "")
        n += w
        doms = {d.lower() for d in _DOMAIN.findall(t)}
        if not doms and not _URL.search(t):
            continue
        linked += w
        if len(_URL.sub(" ", t).split()) < 3:
            link_only += w
        for d in doms:
            dom_posts[d] += w
            if m.get("user") is not None:
                dom_users[d].add(m.get("user"))
    if not n:
        return {"n": 0}
    tops = [{"domain": d, "posts": c, "share": _pct(c, n), "accounts": len(dom_users[d])}
            for d, c in dom_posts.most_common(top)]
    promo = [t for t in tops if t["accounts"] <= 3 and (t["posts"] >= 20 or t["share"] >= 5)]
    return {"n": n, "linked_pct": _pct(linked, n), "link_only_pct": _pct(link_only, n),
            "top_domains": tops, "promo_domains": [t["domain"] for t in promo]}


def hour_fingerprint(msgs, users=None):
    """Human or machine hours: UTC hour histogram of posts (optionally only for `users`),
    normalized entropy (1 = flat around the clock), night share (00-06 UTC) and a label."""
    hist = [0.0] * 24
    for m in _msgs(msgs):
        if users is not None and m.get("user") not in users:
            continue
        ts = parse_ts(m.get("ts") or m.get("datetime") or m.get("timestamp"))
        if ts is not None:
            hist[ts.astimezone(timezone.utc).hour] += _w(m)
    n = sum(hist)
    if n < 24:
        return {"insufficient": True, "n": int(n)}
    p = [h / n for h in hist if h > 0]
    ent = round(-sum(x * math.log(x) for x in p) / math.log(24), 2)
    peak_h = max(range(24), key=hist.__getitem__)
    ratio = round(max(hist) / max(min(hist), 0.5), 1)   # busiest hour / quietest hour
    # Humans keep ~0.90-0.95 normalized entropy (a 3-8x day/night swing); a scheduler is ~1.0.
    if n >= 50 and ent >= 0.97 and ratio <= 2.5:
        label = "flat around the clock (bot-like)"
    elif ent <= 0.95 or ratio >= 4:
        label = "diurnal (human-like)"
    else:
        label = "mixed"
    return {"n": int(n), "entropy": ent, "peak_trough_ratio": ratio,
            "night_share_00_06utc": _pct(sum(hist[:6]), n),
            "peak_hour_utc": peak_h, "peak_hour_share": _pct(hist[peak_h], n),
            "active_hours": sum(1 for h in hist if h >= n / 48), "label": label}


_QWORD = re.compile(r"^\s*(should|is|are|will|when|why|what|how|anyone|does|do|can|could|would|"
                    r"wen|thoughts|any)\b", re.I)
_TRADE = re.compile(r"\b(buy|sell|long|short|entry|exit|dip|ape|bag|hold|hodl|dca|leverage)\b", re.I)


def question_ratio(msgs):
    """Retail indecision proxy: share of random posts that are questions, and the share that
    are questions about a trade (buy / sell / entry / exit ...)."""
    n = q = tq = 0
    for m in _random(_msgs(msgs)):
        w = _w(m)
        t = str(m.get("text") or "")
        n += w
        if "?" in t or _QWORD.match(t):
            q += w
            if _TRADE.search(t):
                tq += w
    if not n:
        return {"n": 0}
    return {"n": n, "question_pct": _pct(q, n), "trade_question_pct": _pct(tq, n)}


# -------------------------------------------------------------- sanity + one card --------

def sanity(d):
    """Cross-checks between the stats block and the sample. Every line starts with OK / FLAG /
    NOTE; quote the FLAGs in the report and act on LOW-N."""
    d = load(d)
    stats, msgs = d.get("stats") or {}, d["messages"]
    out = []
    tm, ud, sm = _f(stats.get("total_matching")), _f(stats.get("unique_after_dedup")), _f(stats.get("sampled"))
    if tm is None:
        out.append("FLAG stats.total_matching missing — no population denominator; do not extrapolate")
    elif tm == 0:
        out.append("FLAG total_matching = 0 — there is NO crowd data in this window")
    else:
        out.append(f"OK total_matching={int(tm)}")
    if sm is not None and int(sm) != len(msgs):
        out.append(f"FLAG messages in file ({len(msgs)}) != stats.sampled ({int(sm)})")
    elif sm is not None:
        out.append(f"OK sampled={len(msgs)} rows")
    if tm and ud is not None and ud > tm:
        out.append(f"FLAG unique_after_dedup ({int(ud)}) > total_matching ({int(tm)})")
    bs = sum(_source_counts(stats.get("by_source")).values())
    if tm and bs:
        out.append((f"OK by_source sums to {int(bs)}" if abs(bs - tm) <= 0.01 * tm
                    else f"FLAG by_source sums to {int(bs)} but total_matching is {int(tm)}"))
    vc = sum(c for _, c in _curve(stats.get("volume_curve")))
    if tm and vc:
        out.append((f"OK volume_curve sums to {int(vc)}" if abs(vc - tm) <= 0.05 * tm
                    else f"FLAG volume_curve sums to {int(vc)} vs total_matching {int(tm)} "
                         f"({_pct(vc - tm, tm):+d}%) — bucket bounds differ from the window"))
    nr = len(_random(msgs))
    if nr < 200:
        out.append(f"FLAG LOW-N random stratum n={nr} < 200 — shares carry wide error; say so, "
                   f"do not extrapolate to the population with confidence")
    else:
        out.append(f"OK random stratum n={nr}")
    sb = ((stats.get("sentiment_balance") or {}).get("by_bucket") or [])[:5]
    bad = [b for b in sb if isinstance(b, dict)
           and abs(sum(_f(b.get(k)) or 0 for k in ("bullish", "bearish", "neutral")) - 1) > 0.02]
    if bad:
        out.append(f"FLAG {len(bad)} sentiment buckets do not sum to 1 — check the field names before quoting")
    if msgs:
        present = set().union(*(m.keys() for m in msgs[:50] if isinstance(m, dict)))
        missing = [k for k in ("user", "unit", "ts", "bull", "bear", "engagement") if k not in present]
        if missing:
            out.append(f"NOTE message rows lack {missing} — recipes needing them return n/a")
    return out


def card(d, top=5):
    """Every local recipe in one call over the offloaded file: sanity, population, organic
    (+ verdict and rule), clusters, accounts, burst shape, mood, links, questions, hours."""
    d = load(d)
    stats, msgs = d.get("stats") or {}, d["messages"]
    rep, ctx = dedup_report(msgs, top), context(stats)
    verdict, rule = organic_verdict(rep, ctx)
    src = _source_counts(stats.get("by_source"))
    st = sum(src.values())
    return {
        "sanity": sanity(d),
        "population": {"total_matching": stats.get("total_matching"), "sampled": len(msgs),
                       "random_n": len(_random(msgs)),
                       "sources_pct": {s: round(100 * c / st) for s, c in
                                       sorted(src.items(), key=lambda kv: -kv[1])} if st else {}},
        "organic": {**{k: v for k, v in rep.items() if k != "top_clusters"}, **ctx,
                    "verdict": verdict, "rule": rule},
        "top_clusters": rep["top_clusters"],
        "accounts": account_concentration(msgs, top),
        "burst": burst_shape(stats.get("volume_curve") or []),
        "mood": {"polarization": polarization(msgs), "loud_vs_many": loud_vs_many(msgs)},
        "links": link_spam(msgs),
        "questions": question_ratio(msgs),
        "hours": hour_fingerprint(msgs),
    }
