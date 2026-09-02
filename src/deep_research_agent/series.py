"""Time series in tool results: detect them, summarize them, keep the rows out of context.

The one result shape the model must NEVER hold verbatim is a metric series —
``{"data": {"bitcoin": [{"datetime": ..., "value": ...}, ...]}}``, a bare list of dated rows,
or the JSON text of either. Given the rows, it transcribes them into a date/value table in
the report (a 90-row table was the live failure). So ``instrument_tool`` offloads any detected
series to a file REGARDLESS of size and shows the model a computed summary
(``summary_block``) instead; the report side (``report_hygiene``) deletes any rows that still
get written. Stdlib only; tolerant of the field names metric servers use.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from typing import Any

MIN_SERIES_POINTS = 8          # fewer dated rows is a fact list, not a series
MAX_SCAN_BYTES = 4_000_000     # bigger strings are offloaded by size anyway; don't parse twice

_TS_KEYS = ("datetime", "dt", "timestamp", "ts", "date", "time", "t", "d")
_VAL_KEYS = ("value", "v", "val", "y", "count", "close", "price")
_WRAPPERS = ("data", "rows", "values", "result", "results", "series", "timeseriesData", "points")


def parse_ts(x: Any) -> datetime | None:
    """Epoch seconds/ms, ISO 8601 (Z / offset / fractional seconds) or a bare date → aware UTC
    datetime; None for anything else."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        if v > 1e11:                    # epoch milliseconds
            v /= 1000.0
        if not 0 < v < 4e9:             # 1970..2096
            return None
        return datetime.fromtimestamp(v, tz=timezone.utc)
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    if s[-1] in "Zz":
        s = s[:-1] + "+00:00"
    if "." in s:                        # fractional seconds → at most 6 digits
        head, _, tail = s.partition(".")
        i = 0
        while i < len(tail) and tail[i].isdigit():
            i += 1
        s = head + (("." + tail[:i][:6].ljust(6, "0")) if i else "") + tail[i:]
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _num(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if x == x else None          # NaN out
    if isinstance(x, str):
        try:
            return float(x.replace(",", ""))
        except ValueError:
            return None
    return None


def points_of(rows: Any) -> list[tuple[datetime, float]] | None:
    """The sorted (ts, value) points of a list of dated rows — dicts with a timestamp key and a
    value key, or [ts, value] pairs. None unless the list has >= MIN_SERIES_POINTS rows, at least
    90% of them are shaped like that (a list of records with an incidental date is NOT a
    series), and >= MIN_SERIES_POINTS carry a numeric value (nulls are allowed gaps)."""
    if not isinstance(rows, list) or len(rows) < MIN_SERIES_POINTS:
        return None
    shaped = 0
    pts: list[tuple[datetime, float]] = []
    for r in rows:
        ts = val = None
        if isinstance(r, dict):
            tk = next((k for k in _TS_KEYS if k in r), None)
            vk = next((k for k in _VAL_KEYS if k in r), None)
            if tk is None or vk is None:
                continue
            ts, val = parse_ts(r[tk]), _num(r[vk])
        elif isinstance(r, (list, tuple)) and len(r) == 2:
            ts, val = parse_ts(r[0]), _num(r[1])
        if ts is None:
            continue
        shaped += 1
        if val is not None:
            pts.append((ts, val))
    if shaped < 0.9 * len(rows) or len(pts) < MIN_SERIES_POINTS:
        return None
    pts.sort(key=lambda p: p[0])
    return pts


def find_series(result: Any) -> dict[str, list[tuple[datetime, float]]]:
    """Every series in a tool result, by label: '' for a bare list, the map key for one nested
    under a wrapper (``data.bitcoin`` → 'bitcoin'). Handles a list, a dict holding a list or a
    {label: list} map under a wrapper key (or at top level), and the JSON text of any of these.
    {} when the result holds no series — e.g. a social_messages payload (its `messages` rows
    have no value field) or an assets_by_metric ranking (no timestamps)."""
    obj = result
    if isinstance(obj, str):
        s = obj.strip()
        if not s or s[0] not in "[{" or len(s) > MAX_SCAN_BYTES:
            return {}
        try:
            obj = json.loads(s)
        except ValueError:
            return {}
    if isinstance(obj, list):
        p = points_of(obj)
        return {"": p} if p else {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, list[tuple[datetime, float]]] = {}

    def scan(container: Any) -> None:
        if isinstance(container, list):
            p = points_of(container)
            if p:
                out[""] = p
        elif isinstance(container, dict):
            for label, rows in container.items():
                p = points_of(rows) if isinstance(rows, list) else None
                if p:
                    out[str(label)] = p

    for key in _WRAPPERS:
        if key in obj:
            scan(obj[key])
            if out:
                return out
    scan(obj)
    return out


def _iso(dt: datetime) -> str:
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d") if (dt.hour, dt.minute, dt.second) == (0, 0, 0) \
        else dt.strftime("%Y-%m-%dT%H:%M")


def fmt_num(v: float | None) -> str:
    if v is None:
        return "n/a"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.4g}"


def describe(points: list[tuple[datetime, float]]) -> dict[str, Any]:
    """The numbers a report may say about a series: span, first/last (+change), min/max with
    when, mean, median, direction (last third vs first third, ±10%), and a note when the last
    point is a zero that the rest of the series is not (an incomplete current bucket)."""
    vals = [v for _, v in points]
    n = len(vals)
    (t0, v0), (t1, v1) = points[0], points[-1]
    imin = min(range(n), key=vals.__getitem__)
    imax = max(range(n), key=vals.__getitem__)
    third = max(1, n // 3)
    head, tail = statistics.fmean(vals[:third]), statistics.fmean(vals[-third:])
    rel = (tail - head) / abs(head) if head else (0.0 if tail == head else (1.0 if tail > head else -1.0))
    d: dict[str, Any] = {
        "n": n, "start": _iso(t0), "end": _iso(t1), "first": v0, "last": v1,
        "change_pct": (v1 - v0) / abs(v0) * 100 if v0 else None,
        "min": vals[imin], "min_at": _iso(points[imin][0]),
        "max": vals[imax], "max_at": _iso(points[imax][0]),
        "mean": statistics.fmean(vals), "median": statistics.median(vals),
        "direction": "rising" if rel > 0.1 else "falling" if rel < -0.1 else "flat",
    }
    if n >= 3 and v1 == 0 and statistics.median(vals[:-1]) != 0:
        d["note"] = "last point is 0 — likely an incomplete current bucket; ignore it or re-fetch"
    return d


def describe_text(label: str, points: list[tuple[datetime, float]]) -> str:
    d = describe(points)
    chg = f" ({d['change_pct']:+.1f}%)" if d["change_pct"] is not None else ""
    name = f"{label}: " if label else ""
    txt = (f"{name}{d['n']} points, {d['start']} to {d['end']}; first {fmt_num(d['first'])}, "
           f"last {fmt_num(d['last'])}{chg}; min {fmt_num(d['min'])} ({d['min_at']}), "
           f"max {fmt_num(d['max'])} ({d['max_at']}); mean {fmt_num(d['mean'])}, "
           f"median {fmt_num(d['median'])}; {d['direction']} (last third vs first third).")
    if d.get("note"):
        txt += f" NOTE: {d['note']}."
    return txt


def summary_block(series: dict[str, list[tuple[datetime, float]]]) -> str:
    """One indented summary line per series — what the model gets INSTEAD of the rows."""
    return "\n".join("  " + describe_text(label, pts) for label, pts in series.items())


SERIES_RULE = (
    "A time series is NEVER listed row by row — not in the report, not in any message, "
    "whatever the date format. Quote the summary or compute what you need and report the "
    "computed numbers."
)
