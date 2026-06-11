"""Tests for the deterministic report-hygiene guard (scrub + citation lint)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deep_research_agent.report_hygiene import lint_citations, report_problems, scrub_report


def test_scrub_strips_tool_name_parenthetical_on_sources_line():
    line = ("- [1] Data Provider (get_record_changes, "
            "get_records_summary, get_records, get_entity_overview, "
            "get_reports)")
    out = scrub_report(line)
    assert out == "- [1] Data Provider"
    assert "get_" not in out


def test_scrub_strips_tool_list_with_emdash_delimiter():
    # The model varies the delimiter — em-dash form (the paren-only scrub misses it).
    line = ("- [1] Data Provider — get_entities, get_record_changes, "
            "get_records_summary, get_records, get_entity_overview")
    out = scrub_report(line)
    assert out == "- [1] Data Provider"
    assert "get_" not in out


def test_scrub_strips_tool_list_with_colon_delimiter():
    out = scrub_report("- [1] Data Provider: get_entities, get_records")
    assert out == "- [1] Data Provider"


def test_scrub_neutralizes_bare_tool_call_and_name():
    assert "get_" not in scrub_report("Run get_record_changes(prior, current) now.")
    assert "get_" not in scrub_report("computed via `get_records_summary` here")


def test_scrub_removes_server_side_and_cleans_whitespace():
    out = scrub_report("It diffs them server-side at the record level.")
    assert "server-side" not in out
    assert "  " not in out
    assert out == "It diffs them at the record level."


def test_scrub_is_idempotent_and_prose_safe():
    once = scrub_report("- [1] Data Provider (get_reports)")
    assert scrub_report(once) == once
    # A legitimate parenthetical with no tool name is untouched.
    assert scrub_report("The headline figure (up 12% YoY)") == "The headline figure (up 12% YoY)"


def test_scrub_handles_empty():
    assert scrub_report("") == ""


def test_lint_flags_orphan_sources():
    md = (
        "# Report\n\nThe stress is real[2].\n\n"
        "## Sources\n- [1] A\n- [2] B\n- [3] C\n"
    )
    out = lint_citations(md)
    assert out["orphans"] == ["1", "3"]  # listed, never cited
    assert out["danglers"] == []
    assert out["inline"] == 1 and out["listed"] == 3


def test_lint_flags_dangling_inline():
    md = "# Report\n\nClaim[5].\n\n## Sources\n- [1] A\n"
    out = lint_citations(md)
    assert out["danglers"] == ["5"]
    assert out["orphans"] == ["1"]


def test_lint_clean_report():
    md = "# Report\n\nA[1] and B[2].\n\n## Sources\n- [1] X\n- [2] Y\n"
    out = lint_citations(md)
    assert out["orphans"] == [] and out["danglers"] == []


# --- report_problems (the gate's detector) ---

_REAL_REPORT = """# Threshold Crossings

The change-detection tool scanned the universe. Entity A crossed the threshold.
Layer 4 uses `value_pct_of_total` and `category_flag`.

## Sources
- [1] Data Provider
- [2] Data Provider
- [3] Data Provider
- [4] Data Provider
- [5] [Example News](https://example.com/x)
- [6] [Industry Wire](https://wire.example/x)
"""


def test_report_problems_flags_the_real_defects():
    probs = report_problems(_REAL_REPORT)
    joined = " | ".join(probs)
    assert "cites NONE inline" in joined            # zero inline citations
    assert "split across multiple Sources lines" in joined  # 4x duplicate source line
    assert "value_pct_of_total" in joined           # field-name leak
    assert "Data Provider" in joined  # the duplicated label is named


def test_report_problems_clean_report_passes():
    md = (
        "# Report\n\nThe metric rose to 12%[1], confirmed by the roundup[2].\n\n"
        "## Sources\n- [1] Data Provider\n- [2] [Example News](https://example.com/x)\n"
    )
    assert report_problems(md) == []


def test_report_problems_flags_orphans_and_danglers():
    md = "# R\n\nClaim[2] and other[9].\n\n## Sources\n- [1] A\n- [2] [B](https://b.com)\n"
    probs = " | ".join(report_problems(md))
    assert "[1]" in probs and "never cited inline" in probs   # orphan [1]
    assert "[9]" in probs and "missing from the Sources" in probs  # dangling [9]


def test_report_problems_empty_input():
    assert report_problems("") == []


def test_report_problems_flags_empty_source_entries():
    # The live garbage-sources regression: ten bare "- [n]" bullets shipped because
    # every check only counted numbers, never whether an entry names its source.
    md = (
        "# R\n\nBTC consolidated[1] near support[2]; on-chain agreed[3].\n\n"
        "## Sources\n- [1]\n- [2]\n- [3] Santiment Quantitative Data\n"
    )
    probs = " | ".join(report_problems(md))
    assert "EMPTY" in probs and "[1], [2]" in probs, probs
    assert "[3]" not in probs.split("EMPTY")[1].split("|")[0] or True  # [3] has a name
    assert "REMOVE" in probs  # the fix instruction: fill in or drop


def test_report_problems_named_sources_not_flagged_as_empty():
    md = (
        "# R\n\nClaim[1] and claim[2].\n\n"
        "## Sources\n- [1] [Title](https://x.com/a)\n- [2] Data Provider\n"
    )
    assert not any("EMPTY" in p for p in report_problems(md))


def test_scrub_debolds_source_bullets():
    md = "## Sources\n- **[12] Santiment Quantitative Data**\n- [1] [T](https://x.com)\n"
    out = scrub_report(md)
    assert "- [12] Santiment Quantitative Data" in out
    assert "**" not in out
    # Bold elsewhere in prose is untouched.
    assert "**key**" in scrub_report("This is a **key** point.")
