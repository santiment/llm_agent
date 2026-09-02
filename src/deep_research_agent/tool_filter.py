"""Hide tools from a model without unmounting the middleware that provides them.

Used for the extract-subagent (utility model): deepagents' ``FilesystemMiddleware`` always
mounts ``ls / read_file / write_file / edit_file / glob / grep / execute`` together, but the
cheap extractor must work ONLY through ``execute`` — bounded Python slices over the offloaded
JSON. Seen live: with ``read_file`` on offer it paged a one-line 1 MB JSON file 80k chars at
a time, then ``jq .``-dumped it, then paged the evicted dump from ``/large_tool_results``,
burning ~380k tokens and producing nothing. Filtering the tool list at the model request is
the deterministic fix; the prompt rule alone did not hold on the utility model.

Place it AFTER the tool-injecting middleware in a spec's ``middleware`` list (deepagents
appends a spec's middleware after its default filesystem stack, so that ordering holds on
both the orchestrator→extract and the research-subagent→extract paths).
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware

# Everything the extract-subagent may NOT see. `execute` is the one built-in it keeps.
EXTRACT_EXCLUDED_TOOLS: frozenset[str] = frozenset({
    "ls", "read_file", "write_file", "edit_file", "glob", "grep", "write_todos",
})


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        name = tool.get("name")
    else:
        name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def filter_tools(tools: list, excluded: frozenset[str]) -> list:
    """The tools the model may see: ``tools`` minus any whose name is in ``excluded``."""
    return [t for t in tools if _tool_name(t) not in excluded]


class ExcludeToolsMiddleware(AgentMiddleware):
    """Strip ``excluded`` tool names from every model request of the agent it's attached to."""

    def __init__(self, excluded: frozenset[str] | set[str]) -> None:
        super().__init__()
        self.excluded = frozenset(excluded)

    def wrap_model_call(self, request, handler):
        return handler(request.override(tools=filter_tools(request.tools, self.excluded)))

    async def awrap_model_call(self, request, handler):
        return await handler(request.override(tools=filter_tools(request.tools, self.excluded)))
