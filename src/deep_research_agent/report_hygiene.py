"""Deterministic last-mile report hygiene — the guarantee the prompt rules can't give.

Two pure helpers applied to the final report markdown:
  - ``scrub_report``: strip data-layer machinery that leaks despite the prompt rules. The
    persistent failure is a ``(get_x, get_y, …)`` tool list appended to the
    internal-data Sources line; bare inline tool calls/names are the rarer fallback. Only
    high-confidence, prose-safe rewrites — it never changes the meaning of a sentence.
  - ``lint_citations``: report inline-``[n]`` vs ``## Sources`` ``[n]`` mismatches (orphans /
    danglers) for observability. DETECTION only — auto-pruning a source the model merely
    forgot to cite would lose a real source, so this warns rather than edits.
  - ``series_runs`` / ``collapse_series``: a RAW TIME SERIES transcribed into the report
    (consecutive timestamped data rows — hourly sentiment buckets, a volume curve as a
    table). ``report_problems`` flags it so the quality gate bounces the report for a
    summary rewrite; ``collapse_series`` is the last-mile fallback that drops the rows if
    the rewrite still ships them. A report describes a series; it never lists it.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from functools import lru_cache
from typing import NamedTuple

# Tool names leak into reports in several shapes; handle each, since the model varies
# the delimiter (parentheses one run, an em-dash list the next). Which names to scrub
# comes from the RUN's actually-loaded tools (``tool_names`` — agent.py passes the
# search/MCP/custom tool list), so the scrub works for any deployment's naming scheme.
# The legacy ``get_*`` family is always matched as a fallback. PROSE SAFETY: only
# snake_case names (containing "_") are scrubbed — a plain-word tool name like
# "screener" is a real English word, and stripping it would damage prose, so it is
# deliberately ignored here (the prompt rules remain its only guard).
_GET_TOKEN = r"get_[a-z0-9_]+"


def _scrub_token(tool_names: tuple[str, ...]) -> str:
    """Alternation regex matching any scrubbable tool name (longest first, so a name
    that prefixes another can't shadow it), plus the ``get_*`` fallback family."""
    names = sorted(
        {n for n in tool_names if "_" in n and re.fullmatch(r"[A-Za-z0-9_]+", n)},
        key=len, reverse=True)
    return "(?:" + "|".join([*(re.escape(n) for n in names), _GET_TOKEN]) + ")"


class _Patterns(NamedTuple):
    paren: re.Pattern         # 1. parenthetical tool list: "Data Provider (get_x, get_y)"
    list_suffix: re.Pattern   # 2. separator-introduced list to EOL: "Data Provider — get_x, get_y"
    tool_id: re.Pattern       # 3. bare inline call / backticked name: "get_x(date)", "`get_x`"
    bare: re.Pattern          # lint: any tool name in the report body


@lru_cache(maxsize=8)  # one entry per distinct run tool-set; tiny
def _compile(tool_names: tuple[str, ...]) -> _Patterns:
    tok = _scrub_token(tool_names)
    return _Patterns(
        paren=re.compile(rf"\s*\([^()]*\b{tok}\b[^()]*\)"),
        list_suffix=re.compile(
            rf"(?m)\s*[—–:\-]\s*`?{tok}\b`?(?:\s*\([^()]*\))?"
            rf"(?:\s*,\s*`?{tok}\b`?(?:\s*\([^()]*\))?)*\s*$"),
        tool_id=re.compile(rf"`?\b{tok}\b`?(?:\s*\([^()]*\))?"),
        bare=re.compile(rf"\b{tok}\b"),
    )


def _patterns(tool_names) -> _Patterns:
    """Compiled patterns for a run's tool-set. Takes ANY iterable and normalizes it to
    the hashable, order-independent cache key, so callers just pass their tool names."""
    return _compile(tuple(sorted(set(tool_names or ()))))


# Stray implementation adjective.
_SERVER_SIDE = re.compile(r"\s*\bserver-side\b")
# Artifacts left by the removals above.
_EMPTY_PAREN = re.compile(r"\(\s*\)")
_DANGLING_SEP = re.compile(r"(?m)[ \t]*[—–:]+[ \t]*$")
_SPACE_BEFORE_PUNCT = re.compile(r" +([.,;:])")
_MULTISPACE = re.compile(r"[ \t]{2,}")

_CITE = re.compile(r"\[(\d+)\]")
_SOURCES_HEADING = re.compile(r"(?im)^\s{0,3}#{1,6}\s*sources\b.*$")
# A backticked field/identifier (snake_case) — machinery that must not appear in the
# report body. (Bare tool names are matched by the per-run ``_patterns().bare``.)
_BACKTICK_FIELD = re.compile(r"`[^`\n]*[a-z]+_[a-z]+[^`\n]*`")
# A Sources bullet: "- [1] Label" / "- [1][2] Label" → captures the label after the numbers.
_SRC_LABEL = re.compile(r"^\s*-?\s*(?:\[\d+\])+\s*(.+?)\s*$")
# An EMPTY Sources bullet: numbers with nothing after them ("- [1]") — a citation that
# points nowhere. Seen live from a mid-tier writer: ten bare [n] bullets shipped because
# every other check only counts numbers, not whether an entry names its source.
_SRC_EMPTY = re.compile(r"^\s*-\s*(?:\s*\[\d+\])+\s*$")
# ---- Raw time series in the report body ------------------------------------------------
# The shape rejected live as "completely unreadable": consecutive lines each opening with a
# date/timestamp followed by numbers — `- 2026-09-01T09:00:00.000Z: bearish=0.0544,
# bullish=0.3833, neutral=0.5622` for 24 hours, `| 2026-09-01 | 0.05 | 0.38 |` table rows,
# or `2026-06-04    19.189` for 90 days. Detection is per LINE and only a RUN of such lines
# counts, so a dated fact in prose or a short dated list is left alone. Blank lines inside a
# run do not end it (a table pasted with spacing is still a table).
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_TS = (
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"  # ISO
    r"|\d{4}/\d{2}/\d{2}"                        # 2026/06/04
    r"|\d{1,2}[./]\d{1,2}[./]\d{4}"              # 04.06.2026, 6/4/2026
    rf"|{_MONTHS} \d{{1,2}},? \d{{4}}"           # Jun 4, 2026
    rf"|\d{{1,2}} {_MONTHS} \d{{4}}"             # 4 Jun 2026
)
_SERIES_ROW = re.compile(
    r"^\s*(?:[-*+]\s*|\d+[.)]\s*|\|\s*)?"  # optional bullet / list number / table cell
    rf"\**`?({_TS})`?\**"                  # the timestamp
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z]{2,}")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_VALUE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")   # the first value column, for the collapse note
# Fewer consecutive timestamped rows than this is a dated list, not a series.
MIN_SERIES_ROWS = 5


def _series_row_ts(line: str) -> str | None:
    """The timestamp if ``line`` is a data row of a series (timestamp + numbers, little
    prose), else None."""
    m = _SERIES_ROW.match(line)
    if not m:
        return None
    rest = m.group("rest")
    nums, words = len(_NUM.findall(rest)), len(_WORD.findall(rest))
    # `bearish=0.05, bullish=0.38` → 2+ numbers; `| 0.05 |` → 1 number, 0 words. A dated
    # headline ("2026-08-01: launch raised $5M in Series A") has one number and prose.
    if nums >= 2 or (nums == 1 and words <= 3):
        return m.group(1)
    return None


def _row_value(line: str) -> float | None:
    """The first numeric column of a series row (``$79,038`` → 79038.0), else None."""
    m = _SERIES_ROW.match(line)
    v = _VALUE.search(m.group("rest")) if m else None
    if not v:
        return None
    try:
        return float(v.group(0).replace(",", ""))
    except ValueError:
        return None


def series_runs(md: str, min_rows: int = MIN_SERIES_ROWS) -> list[tuple[int, int, str, str]]:
    """Runs of >= ``min_rows`` timestamped data rows in ``md`` (blank lines between rows
    allowed), as ``(first_line, end_line_exclusive, first_timestamp, last_timestamp)`` over
    ``md.splitlines()`` indices. Empty when the text transcribes no series."""
    if not md:
        return []
    lines = md.splitlines()
    runs: list[tuple[int, int, str, str]] = []
    i = 0
    while i < len(lines):
        ts = _series_row_ts(lines[i])
        if ts is None:
            i += 1
            continue
        j, first, last, rows = i, ts, ts, 0
        while j < len(lines):
            t = _series_row_ts(lines[j])
            if t is not None:
                last, rows, j = t, rows + 1, j + 1
                continue
            if not lines[j].strip():             # blank lines inside a run don't end it
                k = j
                while k < len(lines) and not lines[k].strip():
                    k += 1
                if k < len(lines) and _series_row_ts(lines[k]) is not None:
                    j = k
                    continue
            break
        if rows >= min_rows:
            runs.append((i, j, first, last))
        i = j
    return runs


def series_row_count(md: str, start: int, end: int) -> int:
    """Data rows inside a run (blank lines excluded)."""
    return sum(1 for l in md.splitlines()[start:end] if _series_row_ts(l) is not None)


def _fmt_num(v: float) -> str:
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.4g}"


def _run_summary(rows: list[str]) -> str:
    """One sentence of statistics over a run's first value column — the collapse keeps the
    information the rows carried, not the rows. '' when the rows carry no consistent value."""
    data = [(t, v) for t, v in ((_series_row_ts(l), _row_value(l)) for l in rows)
            if t is not None and v is not None]
    if len(data) < 2:
        return ""
    vals = [v for _, v in data]
    imin = min(range(len(vals)), key=vals.__getitem__)
    imax = max(range(len(vals)), key=vals.__getitem__)
    multi = any(len(_NUM.findall(_SERIES_ROW.match(l).group("rest"))) > 1
                for l in rows if _series_row_ts(l) is not None)
    col = "first value column" if multi else "values"
    return (f" Summary of the {col}: first {_fmt_num(vals[0])}, last {_fmt_num(vals[-1])}, "
            f"min {_fmt_num(vals[imin])} at {data[imin][0]}, max {_fmt_num(vals[imax])} at "
            f"{data[imax][0]}, mean {_fmt_num(statistics.fmean(vals))}.")


def collapse_series(md: str) -> str:
    """Last-mile fallback: replace every raw series run with ONE line saying what was
    dropped, plus the statistics the rows carried. Idempotent (the replacement is not itself
    a series row). Normally the quality gate has already made the model rewrite the series as
    a summary; this fires — on the live `submit_report` emit and on the persisted report —
    when the rewrite still shipped the rows."""
    runs = series_runs(md or "")
    if not runs:
        return md
    lines = md.splitlines(keepends=True)
    for start, end, first, last in reversed(runs):
        rows = [l.rstrip("\n") for l in lines[start:end]]
        n = sum(1 for l in rows if _series_row_ts(l) is not None)
        note = (f"*(Raw series of {n} timestamped rows, {first} to {last}, omitted — a report "
                f"describes a series, it never lists it.{_run_summary(rows)})*\n")
        lines[start:end] = [note]
    return "".join(lines)


# A fully-bolded Sources bullet ("- **[12] Santiment Quantitative Data**") — renders as a
# shouting pseudo-heading in the report card; de-bold it (scrub), content unchanged.
_SRC_BOLD = re.compile(r"(?m)^(\s*-\s*)\*\*((?:\[\d+\])+[^*\n]*)\*\*\s*$")


def scrub_report(md: str, tool_names=()) -> str:
    """Remove leaked data-layer machinery from report markdown. Idempotent and prose-safe.
    ``tool_names`` is the run's loaded tool list (snake_case names are scrubbed exactly;
    the ``get_*`` family always matches as a fallback)."""
    if not md:
        return md
    pats = _patterns(tool_names)
    out = pats.paren.sub("", md)  # (get_a, get_b)
    out = pats.list_suffix.sub("", out)  # — get_a, get_b   /   : get_a, get_b
    out = pats.tool_id.sub(
        "the underlying data", out
    )  # bare get_a(args) / `get_a` left in prose
    out = _SERVER_SIDE.sub("", out)
    out = _SRC_BOLD.sub(r"\1\2", out)  # de-bold "- **[12] Label**" source bullets
    out = _EMPTY_PAREN.sub("", out)
    out = _DANGLING_SEP.sub("", out)
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    out = _MULTISPACE.sub(" ", out)
    return out


def lint_citations(md: str) -> dict:
    """Inline-``[n]`` vs ``## Sources``-``[n]`` consistency. Returns ``orphans`` (listed but
    never cited) and ``danglers`` (cited but never listed) — the report's own CITATIONS rule
    forbids both. Counts are over distinct citation numbers."""
    if not md:
        return {"orphans": [], "danglers": [], "inline": 0, "listed": 0}
    m = _SOURCES_HEADING.search(md)
    body, sources = (md[: m.start()], md[m.start() :]) if m else (md, "")
    inline = set(_CITE.findall(body))
    listed = set(_CITE.findall(sources))
    return {
        "orphans": sorted(listed - inline, key=int),
        "danglers": sorted(inline - listed, key=int),
        "inline": len(inline),
        "listed": len(listed),
    }


def _fmt_cites(nums: list[str]) -> str:
    return ", ".join(f"[{n}]" for n in nums)


def _duplicate_source_label(sources: str) -> str | None:
    """An internal (non-URL) data source listed on more than one Sources line — the CITATIONS
    rule requires one line per source, grouping its [n]. Returns the first offending label."""
    labels: list[str] = []
    for line in sources.splitlines():
        mm = _SRC_LABEL.match(line)
        if not mm:
            continue
        label = mm.group(1)
        if (
            label.startswith("[") or "http" in label
        ):  # a web source (markdown link) — skip
            continue
        labels.append(label)
    for label, n in Counter(labels).items():
        if n > 1:
            return label
    return None


def report_problems(md: str, tool_names=()) -> list[str]:
    """Presentation-contract violations a research report must NOT ship with — limited to the
    ones the AUTHORING model can fix because it knows which claim maps to which source (inline
    citations, source grouping) or how to reword machinery (field/tool names). Returns a list
    of plain-language fixes; empty means the report passes. The report quality gate uses this
    to bounce a report back for one revision. Run on the SCRUBBED markdown so leaks the scrub
    already removes don't trigger a needless revision."""
    if not md or not md.strip():
        return []
    m = _SOURCES_HEADING.search(md)
    body, sources = (md[: m.start()], md[m.start() :]) if m else (md, "")
    probs: list[str] = []

    cite = lint_citations(md)
    if cite["listed"] and cite["inline"] == 0:
        probs.append(
            f"the report lists {cite['listed']} sources but cites NONE inline — interleave [n] "
            "markers in the text next to the claims they support"
        )
    else:
        if cite["orphans"]:
            probs.append(
                f"sources {_fmt_cites(cite['orphans'])} are listed but never cited inline — "
                "cite them in the body or drop them"
            )
        if cite["danglers"]:
            probs.append(
                f"{_fmt_cites(cite['danglers'])} are cited in the body but missing from the "
                "Sources list — add them"
            )

    empty = sorted(
        {n for line in sources.splitlines() if _SRC_EMPTY.match(line)
         for n in _CITE.findall(line)},
        key=int)
    if empty:
        probs.append(
            f"Sources entries {_fmt_cites(empty)} are EMPTY — a bare [n] that points nowhere. "
            "Every Sources line must NAME its source: '[n] [Title](URL)' for a web source, or "
            "the internal data source's name. If you no longer have a source's title/URL, "
            "REMOVE that entry and its inline [n] citations instead of leaving it blank"
        )

    dup = _duplicate_source_label(sources)
    if dup:
        probs.append(
            f"the data source {dup!r} is split across multiple Sources lines — list it ONCE "
            f"and group its numbers (e.g. '[1][2][3] {dup}')"
        )

    if _patterns(tool_names).bare.search(body):
        probs.append("remove tool/function names from the report body")

    runs = series_runs(body)
    if runs:
        start, end, first, last = runs[0]
        probs.append(
            f"the body transcribes a raw time series ({series_row_count(body, start, end)} "
            "consecutive timestamped "
            f"rows, {first} to {last}"
            + (f"; {len(runs)} such blocks" if len(runs) > 1 else "")
            + ") — a report never lists buckets. Replace each such block with a summary: "
            "first and last value, peak and trough (with when), average, and direction"
        )

    fields = list(dict.fromkeys(_BACKTICK_FIELD.findall(body)))
    if fields:
        probs.append(
            f"remove raw field names from the body (e.g. {', '.join(fields)[:120]}) — describe "
            "them in plain business terms"
        )

    return probs
