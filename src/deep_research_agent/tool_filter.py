"""Hide tools from a model without unmounting the middleware that provides them.

The extract-subagent (utility model) must work ONLY through ``execute`` — bounded Python
slices over offloaded JSON. With ``read_file``/``grep`` on offer it paged a 1 MB one-line
JSON file 80k chars at a time and produced nothing. The coding-subagent keeps the file tools
(it writes, runs and edits scripts) but loses ``grep`` — over a one-line JSON file it is the
same flood — and ``write_todos``, which only invites narration. Place this AFTER the
tool-injecting middleware in a spec's ``middleware`` list.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware

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
