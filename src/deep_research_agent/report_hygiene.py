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

# TODO: Do not expect all tools to be named get_x

# Data-layer tool names all share the `get_*` prefix, which never occurs in real prose — so
# stripping them is safe. They leak in several shapes; handle each, since the model varies the
# delimiter (parentheses one run, an em-dash list the next).
#
# 1. A parenthetical tool list: "Data Provider (get_x, get_y)".
_TOOL_PAREN = re.compile(r"\s*\([^()]*\bget_[a-z0-9_]+[^()]*\)")
# 2. A tool list introduced by a separator (—, –, :, -) running to end of line:
#    "Data Provider — get_x, get_y, get_z".
_TOOL_LIST_SUFFIX = re.compile(
    r"(?m)\s*[—–:\-]\s*`?get_[a-z0-9_]+`?(?:\s*\([^()]*\))?"
    r"(?:\s*,\s*`?get_[a-z0-9_]+`?(?:\s*\([^()]*\))?)*\s*$"
)
# 3. A bare inline tool call or backticked name left in prose: "get_records(date)",
#    "`get_records`". Neutralized to a readable phrase (rare; the prompt handles the body).
_TOOL_ID = re.compile(r"`?\bget_[a-z0-9_]+`?(?:\s*\([^()]*\))?")
# Stray implementation adjective.
_SERVER_SIDE = re.compile(r"\s*\bserver-side\b")
# Artifacts left by the removals above.
_EMPTY_PAREN = re.compile(r"\(\s*\)")
_DANGLING_SEP = re.compile(r"(?m)[ \t]*[—–:]+[ \t]*$")
_SPACE_BEFORE_PUNCT = re.compile(r" +([.,;:])")
_MULTISPACE = re.compile(r"[ \t]{2,}")

_CITE = re.compile(r"\[(\d+)\]")
_SOURCES_HEADING = re.compile(r"(?im)^\s{0,3}#{1,6}\s*sources\b.*$")
# A bare data-layer tool name left in prose, and a backticked field/identifier (snake_case)
# — both are machinery that must not appear in the report body.
_BARE_TOOL = re.compile(r"\bget_[a-z0-9_]+")
_BACKTICK_FIELD = re.compile(r"`[^`\n]*[a-z]+_[a-z]+[^`\n]*`")
# A Sources bullet: "- [1] Label" / "- [1][2] Label" → captures the label after the numbers.
_SRC_LABEL = re.compile(r"^\s*-?\s*(?:\[\d+\])+\s*(.+?)\s*$")


def scrub_report(md: str) -> str:
    """Remove leaked data-layer machinery from report markdown. Idempotent and prose-safe."""
    if not md:
        return md
    out = _TOOL_PAREN.sub("", md)  # (get_a, get_b)
    out = _TOOL_LIST_SUFFIX.sub("", out)  # — get_a, get_b   /   : get_a, get_b
    out = _TOOL_ID.sub(
        "the underlying data", out
    )  # bare get_a(args) / `get_a` left in prose
    out = _SERVER_SIDE.sub("", out)
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


def report_problems(md: str) -> list[str]:
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

    dup = _duplicate_source_label(sources)
    if dup:
        probs.append(
            f"the data source {dup!r} is split across multiple Sources lines — list it ONCE "
            f"and group its numbers (e.g. '[1][2][3] {dup}')"
        )

    if _BARE_TOOL.search(body):
        probs.append("remove tool/function names (get_*) from the report body")

    fields = list(dict.fromkeys(_BACKTICK_FIELD.findall(body)))
    if fields:
        probs.append(
            f"remove raw field names from the body (e.g. {', '.join(fields)[:120]}) — describe "
            "them in plain business terms"
        )

    return probs
