"""Report hygiene follows the RUN's real tool names, not a hardcoded prefix.

``scrub_report`` / ``report_problems`` take the run's loaded tool list (agent.py passes
search + MCP + custom names) and scrub exactly those names when they leak into a report
— plus the legacy ``get_*`` family as an always-on fallback. PROSE SAFETY is the
invariant: only snake_case names are ever scrubbed; a plain-word tool name is a real
English word and stripping it would damage prose.

Runs with plain Python (``python tests/test_scrub_tool_names.py``) — no pytest needed —
and is also pytest-discoverable. No network, no API keys.
"""

from __future__ import annotations

from deep_research_agent.report_hygiene import report_problems, scrub_report

_NAMES = ("fetch_holdings", "screener", "loan_yield_history")


def test_custom_name_paren_list_scrubbed():
    out = scrub_report("Data Provider (fetch_holdings, loan_yield_history).", _NAMES)
    assert "fetch_holdings" not in out and "loan_yield_history" not in out


def test_custom_name_suffix_list_scrubbed():
    out = scrub_report("- [1][2] Data Provider — fetch_holdings, loan_yield_history", _NAMES)
    assert "fetch_holdings" not in out and "loan_yield_history" not in out
    assert "Data Provider" in out


def test_custom_name_inline_call_neutralized():
    out = scrub_report("We ran `fetch_holdings(quarter)` for each firm.", _NAMES)
    assert "fetch_holdings" not in out
    assert "the underlying data" in out


def test_plain_word_tool_name_never_scrubbed():
    # "screener" is an English word — scrubbing it would damage prose. Deliberate no-op.
    text = "The screener shows 14 firms above the threshold."
    assert scrub_report(text, _NAMES) == text


def test_get_fallback_still_scrubbed_alongside_custom_names():
    out = scrub_report("See get_records(2025) and `fetch_holdings`.", _NAMES)
    assert "get_records" not in out and "fetch_holdings" not in out


def test_longer_name_not_shadowed_by_its_prefix():
    names = ("load_data", "load_data_v2")
    out = scrub_report("Then load_data_v2(x) ran.", names)
    assert "load_data" not in out and "_v2" not in out


def test_names_accept_any_iterable_and_order():
    a = scrub_report("x fetch_holdings y", ["fetch_holdings", "screener"])
    b = scrub_report("x fetch_holdings y", ("screener", "fetch_holdings"))
    assert a == b


def test_report_problems_flags_custom_name_in_body():
    md = "# T\nfetch_holdings shows growth[1].\n## Sources\n- [1] [A](https://a.example)"
    probs = report_problems(md, _NAMES)
    assert any("tool/function names" in p for p in probs)


def test_report_problems_ignores_plain_word_name():
    md = "# T\nThe screener shows growth[1].\n## Sources\n- [1] [A](https://a.example)"
    assert not any("tool/function names" in p for p in report_problems(md, _NAMES))


def test_no_names_behaves_like_legacy_get_only():
    text = "Custom fetch_holdings stays; get_x goes."
    out = scrub_report(text)
    assert "fetch_holdings" in out and "get_x" not in out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all scrub-tool-names tests passed")
