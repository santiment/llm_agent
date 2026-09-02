"""``series.py``: detect a time series in a tool result (the real fetch_metric_data shape,
bare lists, pairs, JSON text), summarize it, and leave non-series results alone."""

from __future__ import annotations

import json

from deep_research_agent.series import (MIN_SERIES_POINTS, describe, describe_text, find_series,
                                        parse_ts, points_of, summary_block)


def santiment(n=8, slug="bitcoin", start=79000.0, step=-300.0, values=None):
    rows = [{"value": (values[i] if values else start + i * step),
             "datetime": f"2026-08-{1 + i:02d}T00:00:00Z"} for i in range(n)]
    return json.dumps({"data": {slug: rows}, "period": "Since 2026-08-01T00:00:00Z",
                       "interval": "1d", "metric": "price_usd", "slugs": [slug]})


def test_find_series_reads_the_metric_servers_shape():
    found = find_series(santiment())
    assert list(found) == ["bitcoin"]
    pts = found["bitcoin"]
    assert len(pts) == 8 and pts[0][1] == 79000.0 and pts[-1][1] == 79000.0 - 7 * 300
    assert pts[0][0].isoformat().startswith("2026-08-01") and pts == sorted(pts)


def test_find_series_multi_slug_and_bare_shapes():
    two = json.loads(santiment()); two["data"]["ethereum"] = two["data"]["bitcoin"][:]
    assert set(find_series(json.dumps(two))) == {"bitcoin", "ethereum"}
    bare = [{"dt": f"2026-08-{d:02d}", "v": d} for d in range(1, 11)]
    assert len(find_series(bare)[""]) == 10                       # list, not JSON text
    pairs = [[1_756_000_000_000 + i * 86_400_000, i] for i in range(9)]  # epoch ms pairs
    assert len(find_series(json.dumps(pairs))[""]) == 9
    assert parse_ts(1_756_000_000) == parse_ts(1_756_000_000_000)  # s == ms


def test_nulls_are_gaps_not_disqualifiers():
    rows = [{"datetime": f"2026-08-{d:02d}", "value": None if d % 4 == 0 else d} for d in range(1, 13)]
    assert len(points_of(rows)) == 9                                # 12 rows, 3 nulls
    sparse = [{"datetime": f"2026-08-{d:02d}", "value": None if d % 2 else d} for d in range(1, 13)]
    assert points_of(sparse) is None                                # 6 numeric < MIN


def test_non_series_results_are_left_alone():
    social = {"stats": {"total_matching": 10, "volume_curve": [{"t": f"2026-08-{d:02d}", "count": d} for d in range(1, 25)]},
              "messages": [{"ts": f"2026-08-{d:02d}T10:00:00Z", "text": "hi", "user": "u"} for d in range(1, 25)]}
    assert find_series(social) == {}                               # curve is under stats; rows have no value
    ranking = [{"slug": f"coin{i}", "value": i} for i in range(20)]
    assert find_series(json.dumps(ranking)) == {}                  # no timestamps
    assert find_series(santiment(n=MIN_SERIES_POINTS - 1)) == {}   # too short
    assert find_series("Prose about 2026-08-01 and 2026-08-02.") == {}
    assert find_series("[truncated: 5 of 9 rows omitted]") == {}   # not JSON
    assert find_series(None) == {} and find_series(42) == {}


def test_describe_gives_the_reportable_numbers_and_no_rows():
    vals = [19.2, 15.9, 14.9, 12.0, 11.5, 8.2, 47.7, 0.0]
    pts = find_series(santiment(values=vals))["bitcoin"]
    d = describe(pts)
    assert d["n"] == 8 and d["start"] == "2026-08-01" and d["end"] == "2026-08-08"
    assert d["first"] == 19.2 and d["last"] == 0.0 and d["change_pct"] == -100.0
    assert d["max"] == 47.7 and d["max_at"] == "2026-08-07" and d["min"] == 0.0
    assert d["direction"] == "rising"                              # last third mean > first third
    assert "incomplete current bucket" in d["note"]
    txt = describe_text("bitcoin", pts)
    assert txt.startswith("bitcoin: 8 points, 2026-08-01 to 2026-08-08;")
    assert "max 47.7 (2026-08-07)" in txt and "NOTE:" in txt
    assert "15.9" not in txt.replace("mean", "")                    # no per-row values leak
    block = summary_block({"bitcoin": pts, "ethereum": pts})
    assert block.count("\n") == 1 and block.startswith("  bitcoin:")


def test_describe_direction_and_formatting():
    flat = find_series(santiment(step=0.0, start=79037.9))["bitcoin"]
    d = describe(flat)
    assert d["direction"] == "flat" and d["change_pct"] == 0.0 and "note" not in d
    assert "first 79,038" in describe_text("", flat)              # thousands, no label prefix
    assert describe(find_series(santiment())["bitcoin"])["direction"] == "flat"      # -2.7% is noise
    assert describe(find_series(santiment(step=-3000.0))["bitcoin"])["direction"] == "falling"
