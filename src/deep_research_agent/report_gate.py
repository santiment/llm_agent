"""Report quality gate — the judgment-level complement to ``report_hygiene.scrub_report``.

``scrub_report`` deterministically strips machinery the regex can recognize (tool names). The
remaining contract violations — uncited sources, an internal source split across many Sources
lines, raw field names in prose — need to know which claim maps to which source, and only the
authoring model has that. So this middleware intercepts ``submit_report`` BEFORE it delivers,
and if the (scrubbed) report still violates the contract, bounces it back to the model ONCE
with specific fixes instead of guessing. Capped by ``max_revisions`` so it can never loop:
once exhausted, the report is delivered as-is.

``make_graph`` builds middleware fresh per run, so the per-instance revision counter is scoped
to a single turn.
"""

from __future__ import annotations

import logging

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from .events import emit
from .report_hygiene import report_problems, scrub_report

log = logging.getLogger("deep_research_agent.report_gate")

_REVISE = (
    "Your report was NOT delivered to the user yet. Fix the following and call `submit_report` "
    "again with the COMPLETE corrected report. Change ONLY the presentation — do NOT alter any "
    "finding, number, or figure:\n{problems}\n"
    "Then resubmit the full report. (Cite claims inline with [n] matching the Sources list; "
    "list each internal data source on ONE line grouping its [n]; keep tool and field names "
    "out of the report body.)"
)


class ReportQualityGateMiddleware(AgentMiddleware):
    def __init__(self, *, max_revisions: int = 1) -> None:
        super().__init__()
        self.max_revisions = max_revisions
        self._revisions = 0

    def _call_of(self, request):
        # langchain renamed this field `call` -> `tool_call`; read both so a version
        # bump can't silently turn this gate into a no-op (it did exactly that once).
        call = getattr(request, "tool_call", None) or getattr(request, "call", None) or {}
        if isinstance(call, dict):
            return call.get("name", ""), (call.get("args") or {}), call.get("id", "")
        return (getattr(call, "name", ""), getattr(call, "args", {}) or {},
                getattr(call, "id", ""))

    async def awrap_tool_call(self, request, handler):
        name, args, call_id = self._call_of(request)
        if name != "submit_report":
            return await handler(request)
        # Evaluate the SCRUBBED report so leaks the scrub already fixes don't force a revision.
        raw = args.get("report_markdown") if isinstance(args, dict) else ""
        problems = report_problems(scrub_report(raw or ""))
        if not problems:
            return await handler(request)
        if self._revisions >= self.max_revisions:
            log.warning("REPORT GATE: delivering despite issues (revisions exhausted): %s",
                        problems)
            return await handler(request)
        self._revisions += 1
        log.warning("REPORT GATE: revision %d/%d — bouncing report to fix: %s",
                    self._revisions, self.max_revisions, problems)
        emit({"type": "status", "state": "revising", "reason": "report_quality",
              "detail": "; ".join(problems)})
        bullets = "\n".join(f"- {p}" for p in problems)
        return ToolMessage(content=_REVISE.format(problems=bullets),
                           tool_call_id=call_id or "", name="submit_report")
