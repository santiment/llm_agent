"""The generic custom-tools drop-in loader.

Pins the contract that keeps the agent generic: any ``*.py`` in
``cfg.custom_tools_dir`` is auto-loaded if it either subclasses ``CustomTool``
(the easy path) or defines a ``build_tools(cfg)`` / ``build_tool`` factory (the
escape hatch). ``_*`` files are skipped; a broken plugin is logged and skipped
without taking down the others. Self-contained (writes throwaway plugins to a
tmp dir) so it never depends on the real ``custom_tools/`` contents.

Runs with plain Python (``python tests/test_custom_tools.py``) or pytest.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

from deep_research_agent.tools.custom import load_custom_tools

_GOOD = '''
from langchain_core.tools import StructuredTool

def build_tools(cfg):
    async def alpha(x: str) -> str:
        "Alpha tool."
        return x
    return [StructuredTool.from_function(coroutine=alpha, name="alpha", description="Alpha.")]
'''

_SINGLE = '''
from langchain_core.tools import StructuredTool

def build_tool(cfg):
    async def beta(x: str) -> str:
        "Beta tool."
        return x
    return StructuredTool.from_function(coroutine=beta, name="beta", description="Beta.")
'''

# Class path: two CustomTool subclasses in one file (both picked up), an async
# and a sync run, cfg reachable via self.cfg, plus an imported base that must NOT
# be registered as a tool.
_CLASS = '''
from deep_research_agent.tools.custom import CustomTool

class Gamma(CustomTool):
    name = "gamma"
    description = "Gamma."
    async def run(self, x: str) -> str:
        return f"{x}:{self.cfg.tag}"

class Delta(CustomTool):
    name = "delta"
    description = "Delta."
    def run(self, x: str, n: int = 2) -> str:   # sync run, typed default
        return x * n
'''

# enabled() gating: only one of the two should load.
_GATED = '''
from deep_research_agent.tools.custom import CustomTool

class On(CustomTool):
    name = "on"
    description = "On."
    @classmethod
    def enabled(cls, cfg):
        return True
    async def run(self) -> str:
        return "on"

class Off(CustomTool):
    name = "off"
    description = "Off."
    @classmethod
    def enabled(cls, cfg):
        return False
    async def run(self) -> str:
        return "off"
'''

# A subclass missing the required `name` — must be skipped, not fatal.
_BAD_CLASS = '''
from deep_research_agent.tools.custom import CustomTool

class NoName(CustomTool):
    description = "missing name"
    async def run(self) -> str:
        return "x"
'''

_BROKEN = "raise RuntimeError('boom at import')\n"

_NO_FACTORY = "VALUE = 1\n"


def _write(d: Path, name: str, body: str) -> None:
    (d / name).write_text(body)


def test_loads_skips_and_survives_breakage():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, "good.py", _GOOD)          # build_tools -> [alpha]
        _write(d, "single.py", _SINGLE)      # build_tool -> beta
        _write(d, "klass.py", _CLASS)        # CustomTool subclasses -> gamma, delta
        _write(d, "_ignored.py", _GOOD)      # underscore -> skipped
        _write(d, "broken.py", _BROKEN)      # import error -> skipped, not fatal
        _write(d, "badclass.py", _BAD_CLASS)  # missing name -> skipped, not fatal
        _write(d, "nofactory.py", _NO_FACTORY)  # no class/factory -> skipped

        cfg = SimpleNamespace(custom_tools_dir=str(d), tag="T")
        tools = {t.name: t for t in load_custom_tools(cfg)}

        assert sorted(tools) == ["alpha", "beta", "delta", "gamma"]
        # class tool reaches cfg via self.cfg, async run works
        assert asyncio.run(tools["gamma"].ainvoke({"x": "z"})) == "z:T"
        # sync run + typed default surfaces in the arg schema
        assert tools["delta"].args["n"]["default"] == 2
        assert asyncio.run(tools["delta"].ainvoke({"x": "ab", "n": 3})) == "ababab"


def test_enabled_gating_filters_classes():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, "gated.py", _GATED)
        cfg = SimpleNamespace(custom_tools_dir=str(d))
        names = sorted(t.name for t in load_custom_tools(cfg))
        assert names == ["on"]


def test_missing_dir_is_noop():
    cfg = SimpleNamespace(custom_tools_dir="/no/such/dir")
    assert load_custom_tools(cfg) == []
    assert load_custom_tools(SimpleNamespace(custom_tools_dir="")) == []


if __name__ == "__main__":
    test_loads_skips_and_survives_breakage()
    test_enabled_gating_filters_classes()
    test_missing_dir_is_noop()
    print("ok")
