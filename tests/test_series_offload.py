"""A time-series tool result never reaches the model as rows: ``instrument_tool`` offloads it
(whatever its size) and shows a computed summary; ``submit_report`` deletes any series rows on
its live emit. HTTP and sandbox are stubbed."""

from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel

from deep_research_agent import events
from deep_research_agent.events import instrument_tool
from deep_research_agent.tools.report import build_submit_report_tool
from tests.test_series import santiment


class _EmptyArgs(BaseModel):
    pass


class _Tool:
    def __init__(self, result, name="fetch_metric_data"):
        self.name, self.description, self.args_schema, self._result = name, "fake", _EmptyArgs, result

    async def ainvoke(self, _kwargs):
        return self._result


class _Sink:
    def __init__(self):
        self.files: list[tuple[str, bytes]] = []

    def upload_files(self, files):
        self.files += files
        return [type("Up", (), {"path": p, "error": None})() for p, _ in files]


def _run(result, **kw):
    return asyncio.run(instrument_tool(_Tool(result), kind="mcp", **kw).ainvoke({}))


def test_small_series_is_offloaded_with_a_summary_stub():
    raw = santiment()
    sink = _Sink()
    out = _run(raw, offload_sink=sink, offload_dir="/workspace/data")
    assert out.startswith("[Time series saved to a file")
    assert "file: /workspace/data/fetch_metric_data-" in out
    assert "series: bitcoin: 8 points" in out
    assert "first 79,000, last 76,900 (-2.7%)" in out and "flat (last third vs first third)" in out
    assert '"datetime"' not in out and "2026-08-03" not in out     # no row reaches the model
    assert len(sink.files) == 1 and sink.files[0][1] == raw.encode()   # file holds the verbatim result


def test_non_series_small_result_passes_through_unchanged():
    ranking = json.dumps([{"slug": f"c{i}", "value": i} for i in range(20)])
    sink = _Sink()
    assert _run(ranking, offload_sink=sink) == ranking and sink.files == []


def test_series_without_sandbox_gets_summary_and_rule_prepended():
    raw = santiment()
    out = _run(raw)
    assert out.startswith("[Time series. A time series is NEVER listed")
    assert "bitcoin: 8 points" in out and out.endswith(raw)          # rows stay: nothing to offload to


def test_big_series_gets_the_series_stub_not_the_generic_one():
    rows = [{"datetime": f"2026-07-{d:02d}T00:00:00Z", "value": d} for d in range(1, 31)]
    sink = _Sink()
    out = _run(rows, offload_sink=sink, max_result_rows=5)
    assert "Time series saved" in out and "summary:" in out and "30 points" in out
    assert "preview (first" not in out


def test_submit_report_live_emit_collapses_series_rows():
    md = ("# BTC\n\nDominance fell[1].\n\nDate    Value (%)\n"
          + "\n".join(f"2026-06-{d:02d}    {v}" for d, v in
                      [(4, 19.189), (5, 15.980), (6, 14.982), (7, 14.532), (8, 12.040), (9, 11.487)])
          + "\n\n## Sources\n- [1] Santiment\n")
    captured: list[dict] = []
    orig = events._writer
    events._writer = lambda: captured.append
    try:
        reply = asyncio.run(build_submit_report_tool(()).ainvoke({"report_markdown": md}))
    finally:
        events._writer = orig
    assert "delivered" in reply.lower()
    report = [e for e in captured if e["type"] == "report"][0]["markdown"]
    assert "19.189" not in report and "2026-06-05" not in report
    assert "Raw series of 6 timestamped rows" in report
    assert "first 19.19, last 11.49, min 11.49 at 2026-06-09" in report
    assert "Dominance fell[1]." in report and "- [1] Santiment" in report
