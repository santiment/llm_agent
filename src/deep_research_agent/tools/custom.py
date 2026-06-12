"""Drop-in loader for deployment-specific tools.

Keeps the agent generic: any environment-specific tool lives as a ``*.py`` file
in the gitignored ``custom_tools/`` directory (``cfg.custom_tools_dir``) instead
of being hard-coded here. Adding a tool needs NO change to config / agent /
prompts — drop a file in, restart.

There are two ways to declare a tool in a plugin file. Pick whichever fits.

1. Subclass ``CustomTool`` — the easy, structured path (RECOMMENDED). Set
   ``name`` / ``description``, implement ``run``. The loader finds every subclass
   in the file automatically; no factory boilerplate::

       from deep_research_agent.tools.custom import CustomTool

       class Echo(CustomTool):
           name = "echo"
           description = "Echo the input back."

           async def run(self, text: str) -> str:
               return text

   - ``run`` may be sync or ``async def``. Declare typed params (``text: str``,
     ``limit: int = 10``) — they become the tool's arg schema the model sees.
   - ``self.cfg`` is the live ``ResearchConfig`` (read env / settings off it).
   - Override ``enabled(cls, cfg) -> bool`` to skip loading when a prerequisite
     is missing (e.g. an env var is unset). Default: always on.

2. Define a factory ``build_tools(cfg)`` (or ``build_tool(cfg)``) returning raw
   LangChain ``BaseTool`` instance(s) — the escape hatch for dynamic cases
   (build N tools from config, reuse an existing tool object, etc.).

A file may use either path (or both); the loader collects from both. Returned
tools are appended to the agent's tool list and the model sees each tool's own
``name`` / ``description`` — so put usage and citation guidance right there.

Files whose name starts with ``_`` are skipped (use them for templates / shared
helpers). Import, build, or factory errors are logged and skipped — one bad
plugin never takes down the agent or the other plugins.

See ``docs/CUSTOM_TOOLS.md`` for the full guide and ``custom_tools/_template.py``
for a copy-paste starting point.
"""

from __future__ import annotations

import abc
import importlib.util
import inspect
import logging
import os
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

log = logging.getLogger("deep_research_agent.custom_tools")

_FACTORY_NAMES = ("build_tools", "build_tool")


class CustomTool(abc.ABC):
    """Base class for a deployment-specific tool. Subclass it, set ``name`` and
    ``description``, implement ``run`` — that is the entire contract.

    The loader auto-discovers every concrete subclass defined in a plugin file,
    constructs it with the run config, and converts it to a LangChain tool. No
    registration, no factory function.

    Minimal example::

        class WeatherNow(CustomTool):
            name = "weather_now"
            description = "Current weather for a city. Cite as 'OpenWeather'."

            async def run(self, city: str) -> str:
                ...                       # self.cfg is available here

    Notes:
      - ``run`` is the tool body. Declare typed parameters (``city: str``,
        ``limit: int = 10``) — their names / types / defaults become the arg
        schema the model fills in. Avoid ``**kwargs`` (it yields no schema).
      - ``run`` may be sync ``def`` or ``async def``; both work.
      - ``self.cfg`` is the ``ResearchConfig`` for the run — read API keys, URLs,
        or feature flags off it / the environment.
      - Return a string (preferred) or any JSON-serializable value.
    """

    #: Tool name the model invokes — snake_case, stable, unique. REQUIRED.
    name: str = ""
    #: What the tool does, WHEN to use it, and how to cite its data. The model
    #: reads this to decide whether and how to call the tool. Be specific.
    #: REQUIRED.
    description: str = ""

    def __init__(self, cfg: object) -> None:
        self.cfg = cfg

    @classmethod
    def enabled(cls, cfg: object) -> bool:
        """Whether this tool should be loaded for the given run config. Override
        to gate on a prerequisite — e.g. ``return bool(os.environ.get("API_KEY"))``.
        Returning ``False`` silently skips the tool. Default: always enabled."""
        return True

    @abc.abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """The tool body — override with your own typed signature. See class docstring."""
        raise NotImplementedError


def load_custom_tools(cfg: object) -> list[BaseTool]:
    """Import every plugin in ``cfg.custom_tools_dir`` and collect its tools."""
    directory = getattr(cfg, "custom_tools_dir", "") or ""
    if not directory or not os.path.isdir(directory):
        return []

    out: list[BaseTool] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".py") or filename.startswith("_"):
            continue
        path = os.path.join(directory, filename)
        try:
            tools = _load_one(path, cfg)
        except Exception:
            log.exception("custom tool plugin failed to load: %s", path)
            continue
        for tool in tools:
            if isinstance(tool, BaseTool):
                out.append(tool)
            else:
                log.warning("custom tool plugin %s returned a non-tool (%r) — skipped",
                            filename, type(tool))
    if out:
        log.info("custom tools loaded from %s: %s", directory, [t.name for t in out])
    return out


def _load_one(path: str, cfg: object) -> list[BaseTool]:
    mod_name = f"dra_custom_tool_{os.path.splitext(os.path.basename(path))[0]}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        log.warning("could not create import spec for %s", path)
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base = os.path.basename(path)
    classes = _custom_tool_classes(module)
    factory = _find_factory(module)

    tools: list[BaseTool] = _tools_from_classes(classes, cfg, base)
    if factory is not None:
        tools.extend(_call_factory(factory, cfg))

    # Warn only when the file declares NO tool at all. A class that is gated off by
    # enabled() / fails to build, or a factory that legitimately returns [] (its own
    # gating), is NOT a misconfigured plugin — don't cry wolf on every run.
    if not classes and factory is None:
        log.warning("custom tool plugin %s defines no CustomTool subclass and no %s — "
                    "skipped", base, " or ".join(_FACTORY_NAMES))
    return tools


def _custom_tool_classes(module: object) -> list[type]:
    """Concrete ``CustomTool`` subclasses DEFINED in ``module`` — not the imported
    base, not abstract ones, not classes imported from elsewhere."""
    out: list[type] = []
    for obj in vars(module).values():
        if not (isinstance(obj, type) and issubclass(obj, CustomTool)):
            continue
        if obj is CustomTool or inspect.isabstract(obj):
            continue
        if getattr(obj, "__module__", None) != getattr(module, "__name__", None):
            continue  # imported into the file, not defined here
        out.append(obj)
    return out


def _tools_from_classes(classes: list[type], cfg: object, plugin: str) -> list[BaseTool]:
    """Gate, instantiate, and convert each class. One bad class is skipped, not fatal."""
    out: list[BaseTool] = []
    for obj in classes:
        try:
            if not obj.enabled(cfg):
                log.info("custom tool %s.%s disabled by enabled() — skipped", plugin, obj.__name__)
                continue
            out.append(_tool_from_instance(obj(cfg)))
        except Exception:
            log.exception("custom tool %s.%s failed to build — skipped", plugin, obj.__name__)
    return out


def _tool_from_instance(instance: CustomTool) -> BaseTool:
    """Convert a ``CustomTool`` instance into a LangChain ``StructuredTool``. The
    arg schema is inferred from ``run``'s signature (``self`` excluded)."""
    if not instance.name:
        raise ValueError(f"{type(instance).__name__}.name is required (set the class attribute)")
    if not instance.description:
        raise ValueError(f"{type(instance).__name__}.description is required (set the class attribute)")
    run = instance.run
    if inspect.iscoroutinefunction(run):
        return StructuredTool.from_function(
            coroutine=run, name=instance.name, description=instance.description)
    return StructuredTool.from_function(
        func=run, name=instance.name, description=instance.description)


def _find_factory(module: object):
    """The ``build_tools`` / ``build_tool`` factory the file defines, or ``None``."""
    return next(
        (getattr(module, n) for n in _FACTORY_NAMES if callable(getattr(module, n, None))),
        None)


def _call_factory(factory, cfg: object) -> list[BaseTool]:
    """Run a factory and normalize its result to a list (``None`` -> ``[]``)."""
    result = factory(cfg)
    if result is None:
        return []
    return list(result) if isinstance(result, (list, tuple)) else [result]
