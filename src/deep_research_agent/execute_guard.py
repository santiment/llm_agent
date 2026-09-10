"""Collapse raw rows in the orchestrator's ``execute`` results.

``execute`` stays with the orchestrator for one-line arithmetic, which makes it the one tool
through which row data can still enter its context (``cat`` a CSV, ``print(df)``). The same
collapse the report gets runs over every result: series and CSV-shaped blocks become a
stats note; anything else — a number, ``wc -l``, a 3-row ``head`` — passes untouched.
Orchestrator only: for the extract sub-agent ``execute`` is the reading channel.
"""

from __future__ import annotations

import logging

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from .report_hygiene import collapse_data_blocks
from .turn import tool_call_of

log = logging.getLogger("deep_research_agent.execute_guard")

_NOTE = (
    "\n\n[Raw rows were removed from this output: the orchestrator never holds row data. "
    "For numbers, compute and print only the figures; for text, delegate the file to a "
    "sub-agent with `task`.]"
)


class ExecuteResultGuardMiddleware(AgentMiddleware):
    def __init__(self, role: str = "orchestrator") -> None:
        super().__init__()
        self.role = role

    def wrap_tool_call(self, request, handler):
        return self._guard(request, handler(request))

    async def awrap_tool_call(self, request, handler):
        return self._guard(request, await handler(request))

    def _guard(self, request, result):
        name, _, _ = tool_call_of(request)
        if name != "execute" or not isinstance(result, ToolMessage):
            return result
        text = result.content if isinstance(result.content, str) else None
        if text is None:
            return result
        clean = collapse_data_blocks(text)
        if clean == text:
            return result
        log.warning("EXECUTE GUARD (%s): raw rows in an `execute` result collapsed "
                    "(%d -> %d chars)", self.role, len(text), len(clean))
        return result.model_copy(update={"content": clean.rstrip() + _NOTE})
