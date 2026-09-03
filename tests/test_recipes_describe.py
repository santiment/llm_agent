"""``recipes.describe`` and the metric server's {data: {slug: [...]}} wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location(
    "crowd_recipes_describe", Path(__file__).resolve().parents[1] / "skills/crowd-positioning/recipes.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def _payload(slugs=("bitcoin",), values=None):
    rows = [{"value": (values[i] if values else 79000.0 - 300 * i),
             "datetime": f"2026-08-{1 + i:02d}T00:00:00Z"} for i in range(8)]
    return {"data": {s: rows for s in slugs}, "interval": "1d", "metric": "price_usd", "slugs": list(slugs)}


def test_to_series_reads_the_slug_wrapper_and_demands_a_slug_when_ambiguous():
    s = R.to_series(_payload())
    assert len(s) == 8 and s[0][1] == 79000.0
    with pytest.raises(ValueError, match="several slugs"):
        R.to_series(_payload(("bitcoin", "ethereum")))
    assert len(R.to_series(_payload(("bitcoin", "ethereum")), slug="ethereum")) == 8


def test_describe_reports_span_extremes_and_direction_only():
    d = R.describe(_payload(values=[19.2, 15.9, 14.9, 12.0, 11.5, 8.2, 47.7, 0.0]))
    assert d["n"] == 8 and d["start"] == "2026-08-01T00:00" and d["end"] == "2026-08-08T00:00"
    assert d["first"] == 19.2 and d["last"] == 0.0 and d["change_pct"] == -100.0
    assert d["max"] == 47.7 and d["max_at"] == "2026-08-07T00:00"
    assert d["direction"] == "rising" and "incomplete current bucket" in d["note"]
    assert set(d) <= {"n", "start", "end", "first", "last", "change_pct", "min", "min_at", "max",
                      "max_at", "mean", "median", "direction", "note"}     # no rows in the dict
    assert R.describe(_payload())["direction"] == "flat"                    # -2.7% is noise
    assert R.describe(_payload(values=[9.0, 8.5, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]))["direction"] == "falling"
    assert R.describe([])["n"] == 0
    line = R.fmt({"vol": R.describe(_payload())})
    assert "direction" in line and "2026-08-03" not in line
