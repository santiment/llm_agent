"""The `submit_report` tool — the single, explicit way the agent delivers its answer.

Making the final report a deliberate tool call (rather than "whatever the last chat
message happened to be") means the answer is written exactly once: no `#`-heading
heuristics, and no nudge-induced rewrites. The tool's argument IS the report.
"""

from __future__ import annotations

import logging

from langchain_core.tools import StructuredTool

from ..events import emit
from ..report_hygiene import scrub_report

log = logging.getLogger("deep_research_agent.report")

# A report longer than this is almost certainly a raw-row dump, not a synthesis (the
# 20k-row blow-up was >1M chars). ~50k chars ≈ a very long but legitimate report.
_MAX_REPORT_CHARS = 50_000


def build_submit_report_tool() -> StructuredTool:
    async def submit_report(report_markdown: str) -> str:
        """Deliver the FINAL report to the user. Call this EXACTLY ONCE, when research
        is complete, with the full report as Markdown (begin with a single '# ' title).
        This is the ONLY way to deliver your answer — never write the report as a normal
        message. After it returns, stop; do not repeat or rewrite the report."""
        md = report_markdown if isinstance(report_markdown, str) else str(report_markdown)
        # Last-mile guard: strip any data-layer machinery (tool names / call syntax) that
        # leaked past the prompt rules, so the user never sees `get_*` in the report.
        md = scrub_report(md)
        # Backstop against a raw-row dump: hard-truncate past a generous ceiling so a
        # pathological dump can't reach the user; the prompt rules are the primary guard,
        # this guarantees termination.
        if len(md) > _MAX_REPORT_CHARS:
            log.warning("REPORT TRUNCATED: %d chars > cap %d — likely a raw-row dump",
                        len(md), _MAX_REPORT_CHARS)
            md = md[:_MAX_REPORT_CHARS] + (
                "\n\n> _[Report truncated — exceeded the length cap. Summarize and aggregate "
                "findings (totals, counts, top-N); do not transcribe raw rows.]_\n"
            )
        emit({"type": "report", "markdown": md})
        return (
            "Report delivered to the user. You are DONE — end your turn now. Do not "
            "repeat, restate, or rewrite the report."
        )

    return StructuredTool.from_function(
        coroutine=submit_report,
        name="submit_report",
        description=(
            "Deliver the FINAL report to the user as complete Markdown (begins with a "
            "single '# ' title). Call exactly once when research is complete — it is the "
            "ONLY way to deliver your answer. Never also write the report as a chat message. "
            "Write it as a market research note for a professional analyst: lead with "
            "the findings and the numbers, and NEVER mention tools, function names, 'MCP', "
            "queries, sub-agents, or how the data was retrieved."
        ),
    )
