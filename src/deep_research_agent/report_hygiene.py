"""Deterministic last-mile report hygiene — the guarantee the prompt rules can't give.

Two pure helpers applied to the final report markdown:
  - ``scrub_report``: strip data-layer machinery that leaks despite the prompt rules. The
    persistent failure is a ``(get_x, get_y, …)`` tool list appended to the
    internal-data Sources line; bare inline tool calls/names are the rarer fallback. Only
    high-confidence, prose-safe rewrites — it never changes the meaning of a sentence.
  - ``lint_citations``: report inline-``[n]`` vs ``## Sources`` ``[n]`` mismatches (orphans /
    danglers) for observability. DETECTION only — auto-pruning a source the model merely
    forgot to cite would lose a real source, so this warns rather than edits.
"""

from __future__ import annotations

import re
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

    fields = list(dict.fromkeys(_BACKTICK_FIELD.findall(body)))
    if fields:
        probs.append(
            f"remove raw field names from the body (e.g. {', '.join(fields)[:120]}) — describe "
            "them in plain business terms"
        )

    return probs
