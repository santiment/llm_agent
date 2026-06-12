"""The generic custom-tools drop-in loader.

Pins the contract that keeps the agent generic: any ``*.py`` in
``cfg.custom_tools_dir`` defining ``build_tools(cfg)`` (or ``build_tool``) is
auto-loaded; ``_*`` files are skipped; a broken plugin is logged and skipped
without taking down the others. Self-contained (writes throwaway plugins to a
tmp dir) so it never depends on the gitignored ``custom_tools/`` contents.

Runs with plain Python (``python tests/test_custom_tools.py``) or pytest.
"""

from __future__ import annotations

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

_BROKEN = "raise RuntimeError('boom at import')\n"

_NO_FACTORY = "VALUE = 1\n"


def _write(d: Path, name: str, body: str) -> None:
    (d / name).write_text(body)


def test_loads_skips_and_survives_breakage():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, "good.py", _GOOD)          # build_tools -> [alpha]
        _write(d, "single.py", _SINGLE)      # build_tool -> beta
        _write(d, "_ignored.py", _GOOD)      # underscore -> skipped
        _write(d, "broken.py", _BROKEN)      # import error -> skipped, not fatal
        _write(d, "nofactory.py", _NO_FACTORY)  # no factory -> skipped

        cfg = SimpleNamespace(custom_tools_dir=str(d))
        names = sorted(t.name for t in load_custom_tools(cfg))

        assert names == ["alpha", "beta"]


def test_missing_dir_is_noop():
    cfg = SimpleNamespace(custom_tools_dir="/no/such/dir")
    assert load_custom_tools(cfg) == []
    assert load_custom_tools(SimpleNamespace(custom_tools_dir="")) == []


if __name__ == "__main__":
    test_loads_skips_and_survives_breakage()
    test_missing_dir_is_noop()
    print("ok")
