"""Deterministic last-mile report hygiene — the guarantee the prompt rules can't give.

Two pure helpers applied to the final report markdown:
  - ``scrub_report``: strip data-layer machinery that leaks despite the prompt rules. The
    persistent failure is a ``(get_x, get_y, …)`` tool list appended to the
    internal-data Sources line; bare inline tool calls/names are the rarer fallback. Only
    high-confidence, prose-safe rewrites — it never changes the meaning of a sentence.
  - ``lint_citations``: report inline-``[n]`` vs ``## Sources`` ``[n]`` mismatches (orphans /
    danglers) for observability. DETECTION only — auto-pruning a source the model merely
    forgot to cite would lose a real source, so this warns rather than edits.
  - ``series_runs`` / ``collapse_series``: runs of timestamped data rows (a raw time series).
    ``report_problems`` flags them so the quality gate bounces the report; ``collapse_series``
    drops any that still ship, leaving a one-line note with the statistics.
  - ``delimited_runs`` / ``collapse_delimited``: the same for CSV-shaped blocks without
    timestamps (``telegram_room_a,4821``).
  - ``collapse_data_blocks``: what the delivery paths call — both collapses plus the fence,
    header line and "## CSV 1:" heading that wrapped a dropped block.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from functools import lru_cache
from typing import Any, NamedTuple

from .series import SERIES_RULE, _num, find_series, fmt_num, points_of, summary_block

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
# A sandbox file path ("/workspace/data/social_messages-6debc408.json"): where a result was
# stored is machinery the reader never needs. Scrubbed to the same neutral phrase as a tool
# name; the gate then flags whatever machinery sentence was built around it.
_SANDBOX_PATH = re.compile(
    r"(?<![\w/])(?:/workspace|/skills|/large_tool_results)(?:/[\w\-]+(?:\.[\w\-]+)*)*")
# A sentence whose point is where a file lives ("The file is saved at /workspace/x.csv with
# two columns…") tells a reader who cannot open it nothing — it goes whole. A path inside a
# sentence that carries content is only neutralized. The storage words are checked with the
# path removed, so ".csv" in the path itself does not count.
# A sentence boundary is a terminator followed by whitespace or the end — "1.5%" and
# "x.json" are not boundaries. Leaves the space before the sentence, takes its own newline.
_SENT_CHAR = r"(?:[^.!?\n]|[.!?](?=\S))"
_PATH_SENTENCE = re.compile(
    rf"[^\s.!?]{_SENT_CHAR}*(?:/workspace|/skills|/large_tool_results)(?:/[\w\-]+(?:\.[\w\-]+)*)*"
    rf"{_SENT_CHAR}*[.!?]?[ \t]*\n?")
# The path as the OBJECT of a storage phrase: "saved at <path>", "written to <path>",
# "here's the CSV file — it's saved at <path>". A path used as a data reference inside a
# claim ("the 12,400 files in <path> are 1.5%") is not a location statement.
_LOCATION_PHRASE = re.compile(
    r"\b(?:saved|stored|written|wrote|exported|persisted|located|available|found|lives?|"
    r"is|it's|file)\b[^.!?\n]{0,40}?\b(?:at|to|in|under|as)\b[^.!?\n]{0,12}?`?"
    r"(?:/workspace|/skills|/large_tool_results)/",
    re.IGNORECASE)


def _scrub_path_sentence(m: re.Match) -> str:
    sentence = m.group(0)
    if _LOCATION_PHRASE.search(sentence):
        return ""
    return _SANDBOX_PATH.sub("the underlying data", sentence)


# Artifacts left by the removals above.
_EMPTY_PAREN = re.compile(r"\(\s*\)")
_DANGLING_SEP = re.compile(r"(?m)[ \t]*[—–:]+[ \t]*$")
_SPACE_BEFORE_PUNCT = re.compile(r" +([.,;:])")
_MULTISPACE = re.compile(r"[ \t]{2,}")

_CITE = re.compile(r"\[(\d+)\]")
# A placed chart artifact: the UI renders the `chart` event with this id where a line is
# exactly this token (CRLF tolerated). Inside a code fence it is text, not a placement.
CHART_REF = re.compile(r"^[ \t]*\[chart:([0-9a-f]{8})\][ \t]*\r?$", re.MULTILINE)
_FENCED = re.compile(r"```.*?```", re.DOTALL)


def chart_refs(md: str) -> list[str]:
    """Chart ids a report places (own-line tokens outside code fences), in order, once."""
    return list(dict.fromkeys(CHART_REF.findall(_FENCED.sub("", md or ""))))
_SOURCES_HEADING = re.compile(r"(?im)^\s{0,3}#{1,6}\s*sources\b.*$")
# A backticked field/identifier (snake_case) — machinery that must not appear in the
# report body. (Bare tool names are matched by the per-run ``_patterns().bare``.)
_BACKTICK_FIELD = re.compile(r"`[^`\n]*[a-z]+_[a-z]+[^`\n]*`")
# Data-layer machinery in prose that no regex can rewrite safely, so the gate bounces it:
# the reader must never learn that files exist, where they live, or what code produced a
# number. Seen live: "Source: R.price_levels(d) on /workspace/data/…json" and a whole
# "Offloaded Files" section listing paths with "still needs text extraction".
_MACHINERY = re.compile(
    r"(?<![\w/])(?:/workspace|/skills|/large_tool_results)(?:/[\w\-]+(?:\.[\w\-]+)*)*"  # a sandbox path
    r"|(?<![\w/.:])[\w\-]+\.(?:json|py|csv|parquet|pkl)\b"                  # a data/script file name
    r"|\boffloaded?\s+(?:result\s+)?files?\b|\bresult\s+files?\b"          # "offloaded files"
    r"|\b[A-Za-z_]\w*\.(?!(?:com|org|net|io|co|ai|xyz)\()[a-z_]+\([^()\n]*\)"  # a call: R.card(d), json.load(f)
    r"|\b(?:research|extract|coding)-subagents?\b|\bsub-?agents?\b",
    re.IGNORECASE,
)
# A Sources bullet: "- [1] Label" / "- [1][2] Label" → captures the label after the numbers.
_SRC_LABEL = re.compile(r"^\s*-?\s*(?:\[\d+\])+\s*(.+?)\s*$")
# An EMPTY Sources bullet: numbers with nothing after them ("- [1]") — a citation that
# points nowhere. Seen live from a mid-tier writer: ten bare [n] bullets shipped because
# every other check only counts numbers, not whether an entry names its source.
_SRC_EMPTY = re.compile(r"^\s*-\s*(?:\s*\[\d+\])+\s*$")
# ---- Raw time series in the report body ------------------------------------------------
# A line opening with a timestamp followed by numbers and little prose is a data row; only
# a run of >= MIN_SERIES_ROWS such lines counts (blank lines inside a run don't end it).
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_TS = (
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"  # ISO
    r"|\d{4}/\d{2}/\d{2}"                        # 2026/06/04
    r"|\d{1,2}[./]\d{1,2}[./]\d{4}"              # 04.06.2026, 6/4/2026
    rf"|{_MONTHS} \d{{1,2}},? \d{{4}}"           # Jun 4, 2026
    rf"|\d{{1,2}} {_MONTHS} \d{{4}}"             # 4 Jun 2026
)
_SERIES_ROW = re.compile(
    # optional bullet / list number / line number or DataFrame index / table cell
    r"^\s*(?:[-*+]\s*|\d+[.)]\s*|\d+[\t ]+|\|\s*)?"
    rf"\**`?({_TS})`?\**"                  # the timestamp
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z]{2,}")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_VALUE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")   # the first value column, for the collapse note
# Fewer consecutive timestamped rows than this is a dated list, not a series.
MIN_SERIES_ROWS = 5
# A column-header line ("date,price_usd", "   date  price_usd", numbered by `cat -n` or
# not); taken along when it sits directly above a collapsed run.
_CSV_HEADER = re.compile(
    r"^\s*(?:\d+[\t ]+)?[A-Za-z_][\w.%$-]*(?:(?:\s*[,;\t]\s*|\s{2,})[A-Za-z_][\w.%$-]*)+\s*$")


# A markdown table's header and its |---|---| separator row.
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
_TABLE_HEADER = re.compile(r"^\s*\|.*\|\s*$")


def _with_header(lines: list[str], start: int) -> int:
    """``start`` moved up over the column header of the run below — a CSV header line, or a
    markdown table header plus its separator row."""
    if start > 0 and _CSV_HEADER.match(lines[start - 1].rstrip("\n")):
        return start - 1
    if (start > 1 and _TABLE_SEP.match(lines[start - 1].rstrip("\n"))
            and _TABLE_HEADER.match(lines[start - 2].rstrip("\n"))
            and not _NUM.search(lines[start - 2])):
        return start - 2
    return start


def _series_row_ts(line: str) -> str | None:
    """The timestamp if ``line`` is a data row of a series (timestamp + numbers, little
    prose), else None."""
    m = _SERIES_ROW.match(line)
    if not m:
        return None
    rest = m.group("rest")
    nums, words = len(_NUM.findall(rest)), len(_WORD.findall(rest))
    # `bearish=0.05, bullish=0.38` → 2 numbers, 2 words; `| 0.05 |` → 1 number, 0 words. A
    # dated headline ("2026-08-01: ETF inflows hit $1.2B, 3rd largest day") is prose.
    if nums >= 1 and words <= max(3, nums + 1):
        return m.group(1)
    return None


def _row_value(line: str) -> float | None:
    """The first numeric column of a series row (``$79,038`` → 79038.0), else None."""
    m = _SERIES_ROW.match(line)
    v = _VALUE.search(m.group("rest")) if m else None
    return _num(v.group(0)) if v else None


# A timestamped point wherever it sits — same line or its own. `series_runs` is line-based
# (it locates runs to rewrite in a report); a model that pastes a series into ONE findings
# field (`2026-06-05,-0.20; 2026-06-06,-0.20; …`) produces no such lines, so counting points
# is what catches it. Deliberately requires a separator before the number, so prose like
# "on 2026-06-05 it fell 4%" does not count.
_DATED_POINT = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[ T][\d:]+)?\s*[,;:|\t]\s*-?\d[\d,]*(?:\.\d+)?")
# Three dated pairs is a sentence quoting values; more is a transcription of the series.
MAX_QUOTED_POINTS = 3


def dated_points(text: str) -> int:
    """How many timestamped `date<sep>number` points ``text`` transcribes, line breaks
    irrelevant. Above ``MAX_QUOTED_POINTS`` the text is listing a series, not citing it."""
    return len(_DATED_POINT.findall(text or ""))


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


# ---- CSV-shaped blocks without timestamps ----------------------------------------------
# A row is a short label, a delimiter, then numbers only. A run needs MIN_DELIM_ROWS
# consecutive rows with the same delimiter and field count; a markdown table row (leading
# pipe) never matches.
_DELIM_ROW = re.compile(
    r"^\s*(?P<label>[^\s,;\t|][^,;\t|]{0,60}?)\s*(?P<delim>[,;\t])\s*"
    r"(?P<nums>-?\d[\d,]*(?:\.\d+)?(?:\s*[,;\t]\s*-?\d[\d,]*(?:\.\d+)?)*)\s*$"
)
MIN_DELIM_ROWS = 5


def _delim_row(line: str) -> tuple[str, str, int, float] | None:
    """``(label, delimiter, field_count, first_value)`` for a delimited data row, else
    None. Timestamped rows belong to ``series_runs``."""
    m = _DELIM_ROW.match(line)
    if not m or _series_row_ts(line) is not None:
        return None
    nums = _VALUE.findall(m.group("nums"))
    value = _num(nums[0]) if nums else None
    if value is None:
        return None
    return m.group("label"), m.group("delim"), len(nums) + 1, value


def delimited_runs(md: str, min_rows: int = MIN_DELIM_ROWS) -> list[tuple[int, int]]:
    """Runs of >= ``min_rows`` delimited data rows, as ``(first_line, end_line_exclusive)``
    over ``md.splitlines()``; a run breaks when the delimiter or field count changes."""
    if not md:
        return []
    lines = md.splitlines()
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        head = _delim_row(lines[i])
        if head is None:
            i += 1
            continue
        _, delim, fields, _ = head
        j, rows = i, 0
        while j < len(lines):
            row = _delim_row(lines[j])
            if row is not None and row[1] == delim and row[2] == fields:
                rows, j = rows + 1, j + 1
                continue
            break
        if rows >= min_rows:
            runs.append((i, j))
        i = max(j, i + 1)
    return runs


def _delim_summary(rows: list[str]) -> str:
    """Statistics over a delimited run's first numeric column; '' when under 2 rows."""
    data = [(r[0], r[3]) for r in (_delim_row(ln) for ln in rows) if r is not None]
    if len(data) < 2:
        return ""
    vals = [v for _, v in data]
    imin = min(range(len(vals)), key=vals.__getitem__)
    imax = max(range(len(vals)), key=vals.__getitem__)
    return (f" Summary of the first value column: min {fmt_num(vals[imin])} "
            f"({data[imin][0]}), max {fmt_num(vals[imax])} ({data[imax][0]}), "
            f"total {fmt_num(sum(vals))} across {len(vals)} rows.")


def collapse_delimited(md: str) -> str:
    """``collapse_series`` for CSV-shaped blocks. Idempotent."""
    runs = delimited_runs(md or "")
    if not runs:
        return md
    lines = md.splitlines(keepends=True)
    for start, end in reversed(runs):
        rows = [ln.rstrip("\n") for ln in lines[start:end]]
        n = sum(1 for ln in rows if _delim_row(ln) is not None)
        note = (f"*(Raw data block of {n} rows omitted — a report states its figures, it "
                f"never pastes a table.{_delim_summary(rows)})*\n")
        lines[_with_header(lines, start):end] = [note]
    return "".join(lines)


def collapse_data_blocks(md: str) -> str:
    """Every raw-data shape collapsed to a note — a whole-text JSON series, timestamped
    runs, CSV-shaped blocks — plus the fence and dump heading around a dropped block.
    Idempotent."""
    text = md or ""
    json_note = _collapse_json_series(text)
    if json_note is not None:
        return json_note
    out = collapse_delimited(collapse_series(text))
    return _drop_dump_headings(_unwrap_collapsed_fences(out))


def _only_series(obj: Any) -> bool:
    """True when ``obj`` is a series, or a container holding only series (a
    ``{"data": {"bitcoin": [...]}}`` result) — no scalar fields alongside."""
    if isinstance(obj, list):
        return points_of(obj) is not None
    if isinstance(obj, dict) and obj:
        return all(_only_series(v) for v in obj.values())
    return False


def _collapse_json_series(text: str) -> str | None:
    """Note for text that IS a JSON series (``print(json.dumps(rows))``, ``cat x.json``):
    the row detector sees no lines there. Only when the points make up the bulk of the JSON,
    so a stats object that merely contains a curve keeps its stats. None otherwise."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    series = find_series(stripped)
    n = sum(len(pts) for pts in series.values())
    if not n:
        return None
    # Only when the JSON is nothing but series: a stats object that carries a curve next
    # to scalar figures keeps its figures.
    try:
        obj = json.loads(stripped)
    except ValueError:
        return None
    if not _only_series(obj):
        return None
    return (f"*(Raw series of {n} points omitted — a report describes a series, it never "
            f"lists it. {SERIES_RULE})*\n{summary_block(series)}\n")


# A fence whose whole body is collapse notes: keep the notes, drop the fence.
_COLLAPSED_FENCE = re.compile(
    r"(?m)^[ \t]*```[^\n]*\n((?:[ \t]*\*\(Raw [^\n]*\)\*\n)+)[ \t]*```[ \t]*$")


def _unwrap_collapsed_fences(md: str) -> str:
    return _COLLAPSED_FENCE.sub(lambda m: m.group(1).strip() + "\n", md or "")


# A heading that only labels a dump ("## CSV 1: …"); dropped when its section collapsed
# to notes. A heading that merely contains "csv" in prose does not match.
_DUMP_HEADING = re.compile(r"(?im)^#{1,6}[ \t]*(?:csv|table|raw data)[ \t]*\d*[ \t]*[:\-—–]")
_COLLAPSE_NOTE = re.compile(r"^[ \t]*\*\(Raw [^\n]*\)\*[ \t]*$")


def _drop_dump_headings(md: str) -> str:
    """Remove a dump-labeling heading whose first content line is a collapse note."""
    lines = (md or "").splitlines(keepends=True)
    for i, line in enumerate(lines):
        if not _DUMP_HEADING.match(line):
            continue
        for nxt in lines[i + 1:]:
            if not nxt.strip():
                continue
            if _COLLAPSE_NOTE.match(nxt.rstrip("\n")):
                lines[i] = ""
            break
    return "".join(lines)


def series_row_count(md: str, start: int, end: int) -> int:
    """Data rows inside a run (blank lines excluded)."""
    return sum(1 for ln in md.splitlines()[start:end] if _series_row_ts(ln) is not None)


def _run_summary(rows: list[str]) -> str:
    """Statistics over a run's first value column; '' when fewer than 2 rows carry one."""
    data = [(t, v) for t, v in ((_series_row_ts(ln), _row_value(ln)) for ln in rows)
            if t is not None and v is not None]
    if len(data) < 2:
        return ""
    vals = [v for _, v in data]
    imin = min(range(len(vals)), key=vals.__getitem__)
    imax = max(range(len(vals)), key=vals.__getitem__)
    multi = any(len(_NUM.findall(_SERIES_ROW.match(ln).group("rest"))) > 1
                for ln in rows if _series_row_ts(ln) is not None)
    col = "first value column" if multi else "values"
    return (f" Summary of the {col}: first {fmt_num(vals[0])}, last {fmt_num(vals[-1])}, "
            f"min {fmt_num(vals[imin])} at {data[imin][0]}, max {fmt_num(vals[imax])} at "
            f"{data[imax][0]}, mean {fmt_num(statistics.fmean(vals))}.")


def collapse_series(md: str) -> str:
    """Replace every raw series run with one line saying what was dropped plus its
    statistics. Idempotent."""
    runs = series_runs(md or "")
    if not runs:
        return md
    lines = md.splitlines(keepends=True)
    for start, end, first, last in reversed(runs):
        rows = [ln.rstrip("\n") for ln in lines[start:end]]
        n = sum(1 for ln in rows if _series_row_ts(ln) is not None)
        note = (f"*(Raw series of {n} timestamped rows, {first} to {last}, omitted — a report "
                f"describes a series, it never lists it.{_run_summary(rows)})*\n")
        lines[_with_header(lines, start):end] = [note]
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
    out = _PATH_SENTENCE.sub(_scrub_path_sentence, out)  # a sandbox path left in prose
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

    blocks = delimited_runs(body)
    if blocks:
        start, end = blocks[0]
        rows = sum(1 for ln in body.splitlines()[start:end] if _delim_row(ln) is not None)
        probs.append(
            f"the body pastes a raw data block ({rows} delimited rows"
            + (f"; {len(blocks)} such blocks" if len(blocks) > 1 else "")
            + ") — a report never ships a CSV or a pasted table. State the figures that "
            "matter (totals, top-N with their counts, the direction) in prose instead"
        )

    fields = list(dict.fromkeys(_BACKTICK_FIELD.findall(body)))
    if fields:
        probs.append(
            f"remove raw field names from the body (e.g. {', '.join(fields)[:120]}) — describe "
            "them in plain business terms"
        )

    machinery = list(dict.fromkeys(m.group(0) for m in _MACHINERY.finditer(body)))
    if machinery:
        probs.append(
            f"the body mentions files, paths, code or agents (e.g. {', '.join(machinery)[:120]}) — "
            "the reader never sees how a number was produced or where data was stored: delete "
            "every file/path/'offloaded' mention and any list of files or remaining processing "
            "steps, and state each figure with its data source only"
        )

    return probs
