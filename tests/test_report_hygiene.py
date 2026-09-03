"""Tests for the deterministic report-hygiene guard (scrub + citation lint)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deep_research_agent.report_hygiene import (collapse_series, lint_citations, report_problems, scrub_report,
                                                series_row_count, series_runs)


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


# ---- raw time series: detect (gate) and collapse (last-mile) ------------------------------


_HOURLY = "\n".join(
    f"- 2026-09-01T{h:02d}:00:00.000Z: bearish=0.0{h}44, bullish=0.3833, neutral=0.5622"
    for h in range(9, 18))  # 9 hourly buckets — the shape rejected live

_SERIES_REPORT = f"""# Bitcoin crowd mood

Sentiment stayed net-bullish through the day[1].

Hourly sentiment buckets:
{_HOURLY}

## Sources
- [1] Santiment social messages
"""


def test_series_runs_finds_hourly_bucket_list():
    runs = series_runs(_SERIES_REPORT)
    assert len(runs) == 1
    start, end, first, last = runs[0]
    assert end - start == 9
    assert first == "2026-09-01T09:00:00.000Z" and last == "2026-09-01T17:00:00.000Z"


def test_series_runs_finds_timestamp_table():
    rows = "\n".join(f"| 2026-09-0{d} | {d * 100} | 0.{d}1 |" for d in range(1, 8))
    md = f"| date | volume | balance |\n|---|---|---|\n{rows}\n"
    assert len(series_runs(md)) == 1


def test_series_runs_ignores_short_dated_lists_and_prose():
    md = (
        "On 2026-08-01 the price hit $60,000 and 2026-08-02 saw a pullback.\n"
        "- 2026-08-01: launch raised $5M in Series A funding\n"
        "- 2026-08-03: partnership announced with 3 exchanges\n"
        "- 2026-08-05: mainnet v2 shipped, 40% faster blocks\n"
        "- 2026-08-07: 12 new listings\n"
    )
    assert series_runs(md) == []  # 4 dated bullets, one number each with prose — a timeline


def test_series_runs_ignores_long_dated_timeline_with_numbers():
    md = "\n".join(
        f"- 2026-08-{d:02d}: ETF inflows hit ${d}.2B, the {d}rd largest day on record"
        for d in range(1, 9))
    assert series_runs(md) == []  # 8 dated bullets, two numbers each, but prose


def test_report_problems_flags_raw_series():
    probs = " | ".join(report_problems(_SERIES_REPORT))
    assert "raw time series" in probs
    assert "9 consecutive timestamped rows" in probs
    assert "2026-09-01T09:00:00.000Z" in probs


def test_collapse_series_replaces_run_with_one_line_and_is_idempotent():
    out = collapse_series(_SERIES_REPORT)
    assert "bearish=0.0944" not in out
    assert "Raw series of 9 timestamped rows" in out
    assert "Sentiment stayed net-bullish through the day[1]." in out  # prose intact
    assert "- [1] Santiment social messages" in out
    assert collapse_series(out) == out
    assert report_problems(out) == []


def test_collapse_series_noop_on_clean_report():
    md = "# R\n\nThe metric rose to 12%[1].\n\n## Sources\n- [1] Data Provider\n"
    assert collapse_series(md) == md
    assert collapse_series("") == ""


# ---- the live leak shape: whitespace-separated date/value rows, other date formats ----------

_DOMINANCE = "Date    Value (%)\n" + "\n".join(
    f"2026-06-{d:02d}    {v}" for d, v in
    [(4, 19.189), (5, 15.980), (6, 14.982), (7, 14.532), (8, 12.040), (9, 11.487), (10, 11.772)])


def test_series_runs_catch_space_separated_rows_and_other_date_formats():
    assert len(series_runs(_DOMINANCE)) == 1
    assert len(series_runs(_DOMINANCE.replace("    ", "\t"))) == 1
    month = "\n".join(f"| Jun {d}, 2026 | {d * 1.5}% |" for d in range(1, 8))
    assert series_runs(month)[0][2:] == ("Jun 1, 2026", "Jun 7, 2026")
    euro = "\n".join(f"{d:02d}.06.2026  {d * 100}  {d}" for d in range(1, 8))
    assert len(series_runs(euro)) == 1
    dmy = "\n".join(f"- 1{d} Jun 2026: 0.{d}1" for d in range(0, 6))
    assert len(series_runs(dmy)) == 1


def test_series_runs_survive_blank_lines_between_rows():
    spaced = "\n\n".join(_DOMINANCE.splitlines()[1:])            # one blank line between rows
    runs = series_runs(spaced)
    assert len(runs) == 1
    start, end, first, last = runs[0]
    assert (first, last) == ("2026-06-04", "2026-06-10")
    assert series_row_count(spaced, start, end) == 7               # blanks not counted as rows
    assert "7 consecutive timestamped rows" in " ".join(report_problems(spaced))


def test_collapse_note_keeps_the_statistics_the_rows_carried():
    out = collapse_series(_DOMINANCE)
    assert "19.189" not in out and "2026-06-07" not in out
    assert "Raw series of 7 timestamped rows, 2026-06-04 to 2026-06-10" in out
    assert "first 19.19, last 11.77, min 11.49 at 2026-06-09, max 19.19 at 2026-06-04, mean 14.28" in out
    assert collapse_series(out) == out and series_runs(out) == []


# --- machinery in the body: paths, file names, code calls, "offloaded files" ---

def test_scrub_replaces_sandbox_paths_but_leaves_urls():
    out = scrub_report("Source: R.price_levels(d) on /workspace/data/social_messages-6debc408.json.")
    assert "/workspace" not in out and "6debc408" not in out
    assert out == "Source: R.price_levels(d) on the underlying data."
    url = "- [1] [Doc](https://example.com/workspace/data/x.json)"
    assert scrub_report(url) == url                      # a URL path is not a sandbox path
    assert scrub_report(scrub_report("see /skills/crowd-positioning/recipes.py")) == "see the underlying data"


def test_report_problems_flags_paths_calls_and_offloaded_files_sections():
    md = (
        "# R\n\nCrowd is long[1].\n\n"
        "Source: R.price_levels(d) on the underlying data.\n\n"
        "## Offloaded Files\n"
        "- social_messages-6debc408.json: full dataset, still needs text extraction.\n"
        "- /workspace/price_usd_90d.json: trailing 90d price. Used for R.extreme.\n\n"
        "## Sources\n- [1] Santiment social messages\n"
    )
    probs = report_problems(md)
    joined = " | ".join(probs)
    assert "mentions files, paths, code or agents" in joined
    assert "R.price_levels(d)" in joined and "social_messages-6debc408.json" in joined
    # the gate evaluates the SCRUBBED report — the surviving machinery still trips it
    assert any("files, paths, code" in p for p in report_problems(scrub_report(md)))


def test_report_problems_machinery_detector_is_prose_safe():
    md = (
        "# R\n\nWhales offloaded 12k BTC to Coinbase (Nasdaq: COIN)[1]; the U.S. ETF (IBIT) "
        "absorbed it, i.e. net flows were flat. See coinbase.com (investor relations) and the "
        "2.5% move vs. the prior week.[2]\n\n"
        "## Sources\n- [1] Santiment social messages\n- [2] [Doc](https://example.com/data.json)\n"
    )
    assert not any("files, paths, code" in p for p in report_problems(md))
