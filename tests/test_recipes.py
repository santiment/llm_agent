"""skills/crowd-positioning/recipes.py — the deterministic recipes the sandbox seeds as
/workspace/recipes.py. Run on a synthetic window with a known composition: a scheduled price
bot (300 posts, 1 account), a viral copypasta (200 posts, 200 accounts, 7 rooms) and ~500
organic posts, plus oversampled head/poles strata. Every recipe must land on the planted
answer, and none may return message text."""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_MOD_PATH = Path(__file__).resolve().parents[1] / "skills" / "crowd-positioning" / "recipes.py"


def _load_module():
    sys.dont_write_bytecode = True  # keep __pycache__ out of the skills dir
    spec = importlib.util.spec_from_file_location("crowd_recipes_under_test", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_module()
T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
PX = 60_000.0


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _diurnal_hour(rng: random.Random) -> int:
    # humans: busy 12-23 UTC, quiet 00-06
    return rng.choices(range(24), weights=[1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 7, 6, 5, 4, 3])[0]


def synthetic(seed: int = 7) -> dict:
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(400)] + ["bitcoin", "etf", "whales", "pump", "dump", "moon"]
    msgs: list[dict] = []

    # 1) scheduled price bot: one account, one room, every 288 s, numbers + link change
    for i in range(300):
        ts = T0 + timedelta(seconds=288 * i)
        msgs.append({"text": f"🚨 $BTC price: ${60_000 + 7 * i:,} ({(i % 5) - 2:+.1f}%) https://t.co/{i}",
                     "stratum": "random", "copies": 1, "user": "pricebot", "unit": "prices",
                     "source": "telegram", "ts": _iso(ts), "bull": 0.3, "bear": 0.3, "engagement": 0})

    # 2) viral copypasta: 200 people in 7 rooms across 3 sources, 30% with a one-word variant
    rooms = [f"room_{c}" for c in "abcdefg"]
    for i in range(200):
        base = "bitcoin etf inflows hit a record this week institutions are here to stay"
        text = base.replace("this week", "today") if rng.random() < 0.3 else base
        ts = T0 + timedelta(hours=_diurnal_hour(rng), minutes=rng.randrange(60))
        msgs.append({"text": text, "stratum": "random", "copies": 1, "user": f"cp_{i}",
                     "unit": rng.choice(rooms), "source": rng.choice(["reddit", "twitter_crypto", "telegram"]),
                     "ts": _iso(ts), "bull": 0.8, "bear": 0.1, "engagement": rng.randrange(0, 20)})

    # 3) organic: 500 distinct posts, 400 accounts, 40 rooms; some questions, prices, links
    domains = ["coindesk.com", "theblock.co", "x.com", "medium.com", "youtube.com"]
    for i in range(500):
        words = rng.sample(vocab, rng.randrange(8, 13))
        text = " ".join(words)
        if i < 30:
            text += " support at 58000"
        elif i < 50:
            text += " target $68,000"
        elif i < 60:
            text += " 62k is the level"
        if i % 10 == 0:
            text = "should i buy the dip here? " + text
        if i % 7 == 0:
            text += f" https://{rng.choice(domains)}/post/{i}"
        r = rng.random()
        bull, bear = (0.8, 0.1) if r < 0.45 else (0.1, 0.8) if r < 0.70 else (0.3, 0.3)
        ts = T0 + timedelta(hours=_diurnal_hour(rng), minutes=rng.randrange(60))
        msgs.append({"text": text, "stratum": "random", "copies": 2 if i % 50 == 0 else 1,
                     "user": f"org_{i % 400}", "unit": f"r{i % 40}",
                     "source": rng.choice(["reddit", "twitter_crypto", "telegram", "4chan"]),
                     "ts": _iso(ts), "bull": bull, "bear": bear, "engagement": rng.randrange(0, 50)})

    # 4) head stratum: loud twitter accounts, high engagement, decidedly more bullish
    for i in range(60):
        ts = T0 + timedelta(hours=_diurnal_hour(rng), minutes=rng.randrange(60))
        msgs.append({"text": " ".join(rng.sample(vocab, 10)), "stratum": "head", "copies": 1,
                     "user": f"influencer_{i}", "unit": f"influencer_{i}", "source": "twitter_crypto",
                     "ts": _iso(ts), "bull": 0.9, "bear": 0.05, "engagement": rng.randrange(500, 5000)})

    # 5) poles: extremes both ways
    for i in range(40):
        msgs.append({"text": " ".join(rng.sample(vocab, 10)), "stratum": "poles", "copies": 1,
                     "user": f"pole_{i}", "unit": "r1", "source": "reddit", "ts": _iso(T0),
                     "bull": 1.0 if i % 2 else 0.0, "bear": 0.0 if i % 2 else 1.0, "engagement": 3})

    # stats: single-burst curve (23 flat buckets + one 3000 spike) summing to total_matching
    curve = [{"t": _iso(T0 + timedelta(hours=h)), "count": 3000 if h == 14 else 200} for h in range(24)]
    total = sum(b["count"] for b in curve)  # 7600
    stats = {
        "total_matching": total, "unique_after_dedup": total - 200, "sampled": len(msgs),
        "by_source": {"telegram": 4000, "reddit": 2000, "twitter_crypto": 1200, "4chan": 400},
        "volume_curve": curve,
        "sentiment_balance": {"by_bucket": [{"t": b["t"], "bullish": 0.4, "bearish": 0.2, "neutral": 0.4}
                                            for b in curve]},
        "trend_words": [{"word": "etf", "count": 900}, {"word": "inflows", "count": 600},
                        {"word": "58000", "count": 120}, {"word": "bitcoin", "count": 2000}],
        "top_channels": [{"unit": "prices", "source": "telegram", "count": 2400},
                         {"unit": "room_a", "source": "reddit", "count": 500},
                         {"unit": "r1", "source": "reddit", "count": 400}],
    }
    return {"stats": stats, "messages": msgs}


@pytest.fixture(scope="module")
def data():
    return synthetic()


# ----------------------------------------------------------------- loading / helpers -----

def test_load_accepts_path_object_and_bare_list(tmp_path, data):
    p = tmp_path / "social_messages-x.json"
    p.write_text(json.dumps(data))
    assert R.load(str(p))["stats"]["total_matching"] == 7600
    assert R.load(data) is data
    bare = R.load([{"text": "a"}])
    assert bare["stats"] == {} and len(bare["messages"]) == 1
    with pytest.raises(ValueError):
        R.load({"nope": 1})


def test_parse_ts_tolerates_every_shape():
    want = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    assert R.parse_ts("2026-09-01T09:00:00.000Z") == want
    assert R.parse_ts("2026-09-01T09:00:00+00:00") == want
    assert R.parse_ts("2026-09-01T09:00:00.123456789Z") == want.replace(microsecond=123000)
    assert R.parse_ts("2026-09-01T09:00:00") == want
    assert R.parse_ts("2026-09-01") == want.replace(hour=0)
    assert R.parse_ts(int(want.timestamp())) == want
    assert R.parse_ts(int(want.timestamp()) * 1000) == want          # epoch ms
    assert R.parse_ts(str(int(want.timestamp()))) == want
    assert R.parse_ts("h0") is None and R.parse_ts(None) is None and R.parse_ts("") is None


def test_index_lists_every_public_recipe_with_a_docstring():
    text = R.index()
    for name in R.__all__:
        assert name in text
        assert getattr(R, name).__doc__, name
    assert "organic share" in text.lower()


def test_fmt_is_compact_and_readable():
    out = R.fmt({"a": 1, "b": {"x": 1.23456, "y": "s"}, "c": [{"k": 1}, {"k": 2}], "d": [1, 2]})
    assert "a: 1" in out and "b: x=1.235, y=\"s\"" in out
    assert "c:\n  - k=1\n  - k=2" in out and "d: [1, 2]" in out


# ------------------------------------------------------------- 1. extreme vs history -----

def test_to_series_tolerates_shapes():
    rows = [{"datetime": "2026-08-0%dT00:00:00Z" % i, "value": i} for i in range(1, 6)]
    assert [v for _, v in R.to_series(rows)] == [1, 2, 3, 4, 5]
    assert [v for _, v in R.to_series({"data": rows})] == [1, 2, 3, 4, 5]
    pairs = [["2026-08-02T00:00:00Z", 2], ["2026-08-01T00:00:00Z", 1]]
    assert [v for _, v in R.to_series(pairs)] == [1, 2]                  # sorted
    assert R.to_series([{"dt": "2026-08-01", "v": "3.5"}, {"dt": "bad", "v": 1}]) == [
        (datetime(2026, 8, 1, tzinfo=timezone.utc), 3.5)]


def test_extreme_percentile_and_z():
    rng = random.Random(1)
    base = [{"datetime": _iso(T0 - timedelta(days=90 - i)), "value": 10 + rng.random()} for i in range(90)]
    spike = [{"datetime": _iso(T0 + timedelta(hours=h)), "value": 50 + h} for h in range(5)]
    out = R.extreme(base + spike, "2026-09-01T00:00:00Z")
    assert out["pct"] == 100 and out["z"] > 3 and out["n_base"] == 90 and out["n_window"] == 5
    assert R.extreme(base + spike, T0, agg="max")["window"] == 54
    assert R.extreme(spike, "2026-09-01T00:00:00Z")["unbaselined"] is True
    with pytest.raises(ValueError):
        R.extreme(base, "not a date")


# ------------------------------------------------------- 2. organic vs manufactured -----

def test_norm_collapses_templates_but_keeps_word_internal_digits():
    assert R.norm("🚨 $BTC price: $60,000 (-2.0%) https://t.co/0") == R.norm("$BTC price: $61,500 (+1.0%) https://t.co/99")
    assert R.norm("web3 and l2 rollups @alice") == "web3 and l2 rollups"
    assert R.norm("w123 w7") != R.norm("w124 w7")


def test_dedup_report_finds_the_planted_clusters(data):
    rep = R.dedup_report(data["messages"])
    assert rep["random_posts"] == 1010                    # 1000 rows, ten of them copies=2
    assert 45 <= rep["organic_share"] <= 55
    assert rep["near_clusters"] < rep["exact_clusters"] <= 520
    bot, paste = rep["top_clusters"][0], rep["top_clusters"][1]
    assert bot["posts"] == 300 and bot["users"] == 1 and bot["kind"] == "single-account bot"
    assert paste["posts"] == 200 and paste["users"] == 200 and paste["channels"] == 7
    assert paste["kind"].startswith("viral copypasta") and paste["sources"] == 3
    assert rep["biggest_cluster_share"] == 30
    assert all(len(c["fingerprint"]) <= 80 for c in rep["top_clusters"])


def test_dedup_report_handles_empty_and_no_stratum():
    assert R.dedup_report([])["organic_share"] is None
    rep = R.dedup_report([{"text": "same thing here ok yes"}, {"text": "same thing here ok yes"},
                          {"text": "totally different message about eggs"}])
    assert rep["random_posts"] == 3 and rep["near_clusters"] == 2


def test_context_and_verdict(data):
    ctx = R.context(data["stats"])
    assert ctx["exact_unique_share"] == 97 and ctx["chan_conc"] == 43 and ctx["trend"] == "flat"
    rep = R.dedup_report(data["messages"])
    verdict, rule = R.organic_verdict(rep, ctx)
    assert verdict == "manufactured" and "1 account(s)" in rule and "30%" in rule
    # thresholds directly
    assert R.organic_verdict({"organic_share": 25, "biggest_cluster_share": 1, "top_clusters": []}, {})[0] == "manufactured"
    assert R.organic_verdict({"organic_share": 70, "biggest_cluster_share": 2, "top_clusters": []},
                             {"chan_conc": 30})[0] == "organic"
    v, why = R.organic_verdict({"organic_share": 45, "biggest_cluster_share": 2, "top_clusters": []},
                               {"chan_conc": 30})
    assert v == "mixed" and "30-60%" in why
    assert R.organic_verdict({"organic_share": 80, "biggest_cluster_share": 1, "top_clusters": []},
                             {"chan_conc": 75})[0] == "manufactured"


# ------------------------------------------------------------- 4. crowd price levels -----

def test_price_levels_counts_voices_not_bot_prints(data):
    out = R.price_levels(data["messages"], PX)
    levels = {l["level"]: l for l in out}
    assert [l["level"] for l in out[:3]] == [58000, 68000, 62000]        # ranked by voices
    assert levels[58000]["voices"] == 30 and levels[58000]["side"] == "below"
    assert levels[68000]["voices"] == 20 and levels[68000]["side"] == "above"
    # 10 people typed "62k"; the price bot's 85 prints in the same bin are ONE voice and do not
    # drag the reported level away from what the people typed
    assert levels[62000]["voices"] == 11 and levels[62000]["msgs"] == 95
    bot_bins = [l for l in out if l["voices"] == 1]
    assert bot_bins and all(l["side"] in {"at", "above"} for l in bot_bins)
    # bare texts: one voice per text
    plain = R.price_levels(["support 58000", "58,000 holds", "target $68k"], PX)
    assert [(l["level"], l["voices"], l["msgs"]) for l in plain] == [(58000, 2, 2), (68000, 1, 1)]
    assert R.price_levels(["100% sure 5x from here, call 555-0100, 2024 was wild"], PX) == []
    with pytest.raises(ValueError):
        R.price_levels(["x"], 0)


# ------------------------------------------------------- accounts, timing, mood ----------

def test_account_concentration_spots_the_scheduled_bot(data):
    acc = R.account_concentration(data["messages"])
    assert acc["posts"] == 1010 and acc["accounts"] == 601
    top = acc["top_accounts"][0]
    assert top["user"] == "pricebot" and top["share"] == 30 and top["rooms"] == 1
    assert top["cadence"] == "scheduled" and top["gap_cv"] == 0 and top["median_gap_min"] == 4.8
    assert any("one account = 30%" in f for f in acc["flags"])
    assert any("fixed schedule" in f for f in acc["flags"])
    assert 0 < acc["gini"] < 1 and acc["cross_room_accounts"] == 0
    assert R.account_concentration([{"text": "x"}])["accounts"] == 0


def test_burst_shape_labels(data):
    b = R.burst_shape(data["stats"]["volume_curve"])
    assert b["shape"] == "single-burst" and b["peak"] == 3000 and b["peak_share"] == 39
    assert b["half_life_buckets"] == 1 and b["bursts"] == 1 and b["trend"] == "flat"
    assert b["peak_t"].startswith("2026-09-01T14:00")
    ramp = [{"t": f"h{i}", "count": 100 + 40 * i} for i in range(12)]
    assert R.burst_shape(ramp)["shape"] == "ramp" and R.burst_shape(ramp)["trend"] == "rising"
    assert R.burst_shape([{"count": 100}] * 10)["shape"] == "plateau"
    fade = [{"count": c} for c in [100, 900, 700, 500, 300, 200, 150, 120, 100, 100]]
    assert R.burst_shape(fade)["shape"] == "burst-then-fade"
    multi = [{"count": c} for c in [50, 600, 50, 50, 50, 700, 50, 50, 50, 650, 50, 50]]
    assert R.burst_shape(multi)["shape"] == "multi-burst" and R.burst_shape(multi)["bursts"] == 3
    assert R.burst_shape([{"count": 1}])["insufficient"] is True


def test_lead_lag_recovers_a_planted_two_bucket_lead():
    rng = random.Random(3)
    n = 48
    vol = [rng.uniform(100, 1000) for _ in range(n)]
    curve = [{"t": _iso(T0 + timedelta(hours=i)), "count": vol[i]} for i in range(n)]
    price = [100.0]
    for i in range(1, n):
        drive = vol[i - 2] if i >= 2 else 500.0     # return into bucket i is driven by volume at i-2
        price.append(price[-1] * (1 + (drive - 500) / 20000))
    series = [{"datetime": _iso(T0 + timedelta(hours=i)), "value": price[i]} for i in range(n)]
    out = R.lead_lag(curve, series)
    assert out["alignment"] == "timestamp" and out["best_lag"] == 2 and out["best_corr"] > 0.95
    assert "LEADS price by 2" in out["read"]
    # volume reacting to price instead: return at i drives volume at i+1
    vol2 = [500.0] + [500 + 20000 * ((price[i] - price[i - 1]) / price[i - 1]) for i in range(1, n)]
    curve2 = [{"t": _iso(T0 + timedelta(hours=i)), "count": vol2[i]} for i in range(n)]
    out2 = R.lead_lag(curve2, series)
    assert out2["best_lag"] == 0 and "same bucket" in out2["read"]
    assert R.lead_lag(curve[:3], series)["insufficient"] is True
    assert R.lead_lag([{"t": "h0", "count": 1}] * 10, series)["unaligned"] is True


def test_loud_vs_many_sees_influencers_more_bullish(data):
    out = R.loud_vs_many(data["messages"])
    assert set(out["strata"]) == {"random", "head", "poles"}
    assert out["strata"]["head"]["bull_pct"] == 100 and out["strata"]["random"]["bull_pct"] < 70
    assert out["loud_minus_crowd_lean_pts"] >= 15 and "MORE bullish" in out["read"]
    assert out["engagement_weighted"]["n"] > 0 and 0 < out["engagement_top1pct_share"] <= 100
    assert R.loud_vs_many([{"text": "x"}])["note"]


def test_polarization_label(data):
    p = R.polarization(data["messages"])
    # organic: 45% bull / 25% bear / 30% neutral; bot neutral; copypasta bullish -> leaning bullish
    assert p["n"] == 1010 and p["label"] in {"leaning bullish", "consensus bullish"}
    assert 0 < p["split_index"] < 1 and p["bull_pct"] > p["bear_pct"]
    split = [{"bull": 1, "bear": 0}] * 50 + [{"bull": 0, "bear": 1}] * 50
    assert R.polarization(split)["label"] == "split"
    assert R.polarization([{"bull": 0.3, "bear": 0.3}] * 10)["label"].startswith("apathetic")
    assert R.polarization([{"text": "x"}])["n"] == 0


# ------------------------------------------------------ venue, novelty, links, hours -----

def test_source_shift_with_and_without_baseline(data):
    out = R.source_shift(data["stats"])
    assert out["top_source"] == "telegram" and out["top_share"] == 53 and out["hhi"] > 0.3
    base = {"telegram": 30, "reddit": 40, "twitter_crypto": 25, "4chan": 5}
    out = R.source_shift(data["stats"], base)
    assert out["largest_excess"] == "telegram" and out["share_delta_pts"]["telegram"] == 23
    assert "+23 pts" in out["read"]
    assert R.source_shift({"by_source": [{"source": "a", "count": 60}, {"source": "b", "count": 40}]})["top_share"] == 60
    assert R.source_shift({})["note"]


def test_word_novelty():
    now = [{"word": "etf", "count": 900}, {"word": "hack", "count": 300}, {"word": "bitcoin", "count": 2000}]
    before = {"etf": 300, "bitcoin": 1800, "halving": 500}
    out = R.word_novelty(now, before)
    assert out["new"] == [{"word": "hack", "count": 300}]
    assert out["faded"] == [{"word": "halving", "count": 500}]
    assert out["persistent"][0]["word"] == "bitcoin" and out["persistent"][1]["x"] == 3.0
    assert out["novelty_share"] == 9


def test_link_spam_flags_the_single_account_shortener(data):
    out = R.link_spam(data["messages"])
    assert out["n"] == 1010 and 35 <= out["linked_pct"] <= 40
    tco = out["top_domains"][0]
    assert tco["domain"] == "t.co" and tco["posts"] == 300 and tco["accounts"] == 1
    assert out["promo_domains"] == ["t.co"]
    assert out["link_only_pct"] <= out["linked_pct"]


def test_hour_fingerprint_bot_vs_humans(data):
    bot = R.hour_fingerprint(data["messages"], users={"pricebot"})
    assert bot["n"] == 300 and bot["entropy"] >= 0.98 and bot["label"].startswith("flat")
    humans = R.hour_fingerprint([m for m in data["messages"] if m["user"].startswith("org_")])
    assert humans["label"].startswith("diurnal") and humans["night_share_00_06utc"] < 15
    assert humans["peak_trough_ratio"] >= 4 > bot["peak_trough_ratio"]
    assert R.hour_fingerprint([])["insufficient"] is True


def test_question_ratio(data):
    q = R.question_ratio(data["messages"])
    assert q["n"] == 1010 and 4 <= q["question_pct"] <= 6 and q["trade_question_pct"] == q["question_pct"]


# -------------------------------------------------------------- sanity + one card --------

def test_sanity_flags(data):
    lines = R.sanity(data)
    assert any(l.startswith("OK total_matching=7600") for l in lines)
    assert any("OK by_source" in l for l in lines) and any("OK volume_curve" in l for l in lines)
    assert any("OK random stratum n=1000" in l for l in lines)
    assert not any("FLAG" in l for l in lines)
    broken = {"stats": {"total_matching": 100, "sampled": 5, "unique_after_dedup": 200,
                        "by_source": {"a": 10}, "volume_curve": [{"count": 50}]},
              "messages": [{"text": "x"}] * 3}
    flags = [l for l in R.sanity(broken) if l.startswith("FLAG")]
    assert len(flags) >= 5 and any("LOW-N" in f for f in flags) and any("!= stats.sampled" in f for f in flags)
    assert any(l.startswith("NOTE") and "user" in l for l in R.sanity(broken))
    assert any("NO crowd data" in l for l in R.sanity({"stats": {"total_matching": 0}, "messages": []}))


def test_card_runs_every_local_recipe_and_leaks_no_text(data):
    c = R.card(data)
    assert set(c) == {"sanity", "population", "organic", "top_clusters", "accounts", "burst",
                      "mood", "links", "questions", "hours"}
    assert c["organic"]["verdict"] == "manufactured" and c["organic"]["organic_share"] == 50
    assert c["population"]["random_n"] == 1000 and c["population"]["sources_pct"]["telegram"] == 53
    assert c["burst"]["shape"] == "single-burst" and c["mood"]["polarization"]["n"] == 1010
    text = R.fmt(c)
    assert "verdict: \"manufactured\"" in text and "sanity:" in text
    assert "  - OK total_matching=7600" in text and "shape: \"single-burst\"" in text
    # no organic post text, only <=80-char fingerprints of the top clusters
    organic_texts = [m["text"] for m in data["messages"] if m["user"].startswith("org_")]
    assert not any(t[:60] in text for t in organic_texts)
    assert len(text) < 6000
