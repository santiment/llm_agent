"""A time-series tool result never reaches the model as rows: ``instrument_tool`` offloads it
(whatever its size) and shows a computed summary; ``submit_report`` deletes any series rows on
its live emit. HTTP and sandbox are stubbed."""

from __future__ import annotations

import asyncio
import json

from deep_research_agent.events import instrument_tool
from deep_research_agent.tools.report import build_submit_report_tool
from test_series import santiment
from conftest import EmptyArgs


class _Tool:
    def __init__(self, result, name="fetch_metric_data"):
        self.name, self.description, self.args_schema, self._result = name, "fake", EmptyArgs, result

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


def test_structured_series_without_sandbox_is_summarized_too():
    rows = [{"datetime": f"2026-07-{d:02d}T00:00:00Z", "value": d} for d in range(1, 9)]
    out = _run(rows)                                                  # a list, not JSON text
    assert out.startswith("[Time series. A time series is NEVER listed")
    assert "8 points" in out and out.endswith(json.dumps(rows))


def test_big_series_gets_the_series_stub_not_the_generic_one():
    rows = [{"datetime": f"2026-07-{d:02d}T00:00:00Z", "value": d} for d in range(1, 31)]
    sink = _Sink()
    out = _run(rows, offload_sink=sink, max_result_rows=5)
    assert "Time series saved" in out and "summary:" in out and "30 points" in out
    assert "preview (first" not in out


def test_submit_report_live_emit_collapses_series_rows(capture_events):
    md = ("# BTC\n\nDominance fell[1].\n\nDate    Value (%)\n"
          + "\n".join(f"2026-06-{d:02d}    {v}" for d, v in
                      [(4, 19.189), (5, 15.980), (6, 14.982), (7, 14.532), (8, 12.040), (9, 11.487)])
          + "\n\n## Sources\n- [1] Santiment\n")
    reply = asyncio.run(build_submit_report_tool(()).ainvoke({"report_markdown": md}))
    assert "delivered" in reply.lower()
    report = [e for e in capture_events if e["type"] == "report"][0]["markdown"]
    assert "19.189" not in report and "2026-06-05" not in report
    assert "Raw series of 6 timestamped rows" in report
    assert "first 19.19, last 11.49, min 11.49 at 2026-06-09" in report
    assert "Dominance fell[1]." in report and "- [1] Santiment" in report


# --- MCP content-block envelope: the shape langchain-mcp-adapters actually returns -------

def _mcp_blocks(json_text: str):
    """What an MCP tool returns through langchain-mcp-adapters: content blocks, the real
    JSON buried in a text block's `text` (response_format=content_and_artifact)."""
    return [{"type": "text", "text": json_text}]


def test_series_inside_mcp_content_blocks_is_offloaded():
    # The regression: fetch_metric_data returned its series wrapped in MCP text blocks, so
    # find_series saw a [{type,text}] list (no dated rows) and never offloaded it — the
    # rows reached the model, which hand-transcribed them into a file for the recipes.
    raw = santiment()
    sink = _Sink()
    out = _run(_mcp_blocks(raw), offload_sink=sink, offload_dir="/workspace/data")
    assert out.startswith("[Time series saved to a file")
    assert "series: bitcoin: 8 points" in out
    assert '"datetime"' not in out and "2026-08-03" not in out
    assert len(sink.files) == 1 and sink.files[0][1] == raw.encode()   # file holds clean JSON, not blocks


def test_series_in_content_and_artifact_tuple_is_offloaded():
    raw = santiment()
    sink = _Sink()
    out = _run((_mcp_blocks(raw), {"structured": True}), offload_sink=sink)
    assert out.startswith("[Time series saved to a file") and "bitcoin: 8 points" in out
    assert sink.files[0][1] == raw.encode()


def test_plain_text_mcp_block_is_flattened_to_its_string():
    from deep_research_agent.events import unwrap_tool_result
    assert unwrap_tool_result([{"type": "text", "text": "hello"}]) == "hello"
    assert unwrap_tool_result({"type": "text", "text": "hi"}) == "hi"
    assert unwrap_tool_result([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    # a non-text block present → keep the blocks intact (nothing lost)
    blocks = [{"type": "text", "text": "x"}, {"type": "image", "base64": "..."}]
    assert unwrap_tool_result(blocks) == blocks
    # non-MCP shapes pass through untouched
    assert unwrap_tool_result("plain") == "plain"
    assert unwrap_tool_result([{"slug": "btc", "value": 1}]) == [{"slug": "btc", "value": 1}]


def test_small_non_series_mcp_block_still_passes_through_inline():
    # A short discovery/list result stays inline — no needless file + round-trip.
    ranking = json.dumps([{"slug": f"c{i}", "value": i} for i in range(10)])
    sink = _Sink()
    assert _run(_mcp_blocks(ranking), offload_sink=sink) == ranking and sink.files == []
