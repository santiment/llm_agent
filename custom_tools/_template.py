"""COPY-PASTE TEMPLATE for a new custom tool. NOT loaded itself.

The leading ``_`` makes the loader skip this file, so it lives here as a living
example without ever being registered. To add a real tool:

    cp custom_tools/_template.py custom_tools/my_tool.py
    # edit name / description / run, delete what you don't need, restart the agent.

See ``docs/CUSTOM_TOOLS.md`` for the full guide.
"""

from __future__ import annotations

import json
import os

from deep_research_agent.tools.custom import CustomTool


class MyTool(CustomTool):
    """A custom tool. Subclass CustomTool; that's the whole contract.

    The loader auto-discovers this class, constructs it with the run config, and
    exposes it to BOTH the orchestrator and the research sub-agents. Define more
    than one CustomTool subclass in a file and all of them are picked up.
    """

    # REQUIRED. snake_case, stable, unique — this is what the model calls.
    name = "my_tool"

    # REQUIRED. What it does, WHEN to use it, and how to cite its data. The model
    # reads ONLY this to decide whether/how to call the tool — be specific and put
    # citation guidance here (e.g. "cite as 'My Source'").
    description = (
        "One-line summary of what this returns and when to use it. "
        "Cite results as 'My Source'."
    )

    async def run(self, query: str, limit: int = 10) -> str:
        """The tool body.

        Declare typed parameters (``query: str``, ``limit: int = 10``) — their
        names, types, and defaults become the arg schema the model fills in.
        Avoid ``**kwargs`` (it produces no schema). May be sync ``def`` or
        ``async def``.

        ``self.cfg`` is the ResearchConfig for this run — read API keys / URLs /
        flags off it or the environment.

        RETURN FORMAT — the model always ends up seeing a STRING. You may return:
          - a ``str`` (recommended). For structured data, ``json.dumps(...)`` it
            yourself, like the hardcoded example below.
          - a ``list`` (ideally of dicts) — auto JSON-encoded; ALSO the best shape
            for large tabular results: the agent offloads it to a file (row count,
            columns, head) the ``execute`` tool reads back instead of flooding the
            context.
          - any other JSON-serializable value (``dict`` etc.) — auto JSON-encoded.
        Put numbers/facts the model should cite right in the returned text.
        """
        # token = os.environ.get("MY_API_KEY")
        # ... do the real work (hit an API, query a DB, etc.) ...

        # A perfectly valid hardcoded result — swap this for the real call. Returning
        # a JSON string keeps structured fields legible to the model.
        return json.dumps({
            "query": query,
            "limit": limit,
            "results": [
                {"title": "Example result A", "score": 0.92},
                {"title": "Example result B", "score": 0.81},
            ],
            "source": "My Source (cite this)",
        })

    @classmethod
    def enabled(cls, cfg) -> bool:
        """OPTIONAL. Return False to skip loading when a prerequisite is missing.
        Delete this method to always load. Example: require an env var."""
        return bool(os.environ.get("MY_API_KEY"))


# ──────────────────────────────────────────────────────────────────────────────
# ESCAPE HATCH (rarely needed). For dynamic cases — building N tools from config,
# or returning an existing LangChain BaseTool you got from elsewhere — define a
# factory instead of / in addition to the class above. Delete if unused.
#
# from langchain_core.tools import StructuredTool
#
# def build_tools(cfg) -> list:        # or build_tool(cfg) -> single BaseTool
#     async def echo(text: str) -> str:
#         """Echo the input back."""
#         return text
#     return [StructuredTool.from_function(
#         coroutine=echo, name="echo", description="Echo the input back.")]
