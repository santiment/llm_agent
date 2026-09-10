"""Hide tools from a model without unmounting the middleware that provides them — and
refuse them if the model names one anyway.

The extract-subagent (utility model) must work ONLY through ``execute`` — bounded Python
slices over offloaded JSON. With ``read_file``/``grep`` on offer it paged a 1 MB one-line
JSON file 80k chars at a time and produced nothing. The coding-subagent keeps the file tools
(it writes, runs and edits scripts) but loses ``grep`` — over a one-line JSON file it is the
same flood — and ``write_todos``, which only invites narration. Place this AFTER the
tool-injecting middleware in a spec's ``middleware`` list.

Hiding is not disabling: a filtered tool stays registered and deepagents' filesystem prompt
still names it, so a model that calls it from memory gets it executed (an orchestrator read
a whole CSV through a hidden ``read_file``). Excluded names are therefore also refused at
call time, and the handler never runs.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from .turn import tool_call_of

log = logging.getLogger("deep_research_agent.tool_filter")

_REFUSED = (
    "`{name}` is not available to this role, so this call did nothing. Files are read and "
    "written by sub-agents: delegate with the `task` tool (extract-subagent for a data file, "
    "coding-subagent for a script) and work from what it returns. Do not call `{name}` again."
)

EXTRACT_EXCLUDED_TOOLS: frozenset[str] = frozenset({
    "ls", "read_file", "write_file", "edit_file", "glob", "grep", "write_todos",
})
CODING_EXCLUDED_TOOLS: frozenset[str] = frozenset({"grep", "write_todos"})
# The orchestrator plans, delegates, verifies and synthesizes; it never reads or writes
# files itself (sub-agents do, skills included). Hiding the filesystem tools also drops
# ~1.7k tokens of descriptions plus their schemas from EVERY orchestrator step.
ORCHESTRATOR_EXCLUDED_TOOLS: frozenset[str] = frozenset({
    "ls", "read_file", "write_file", "edit_file", "glob", "grep",
})


def _tool_name(tool: Any) -> str | None:
    name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def filter_tools(tools: list, excluded: frozenset[str]) -> list:
    return [t for t in tools if _tool_name(t) not in excluded]


class ExcludeToolsMiddleware(AgentMiddleware):
    def __init__(self, excluded: frozenset[str] | set[str]) -> None:
        super().__init__()
        self.excluded = frozenset(excluded)

    def wrap_model_call(self, request, handler):
        return handler(request.override(tools=filter_tools(request.tools, self.excluded)))

    async def awrap_model_call(self, request, handler):
        return await handler(request.override(tools=filter_tools(request.tools, self.excluded)))

    def wrap_tool_call(self, request, handler):
        refused = self._refuse(request)
        return refused if refused is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        refused = self._refuse(request)
        return refused if refused is not None else await handler(request)

    def _refuse(self, request) -> ToolMessage | None:
        name, _, call_id = tool_call_of(request)
        if name not in self.excluded:
            return None
        log.warning("TOOL FILTER: refused %r (hidden from this role); nothing was executed", name)
        return ToolMessage(content=_REFUSED.format(name=name), tool_call_id=call_id or "",
                           name=name)
