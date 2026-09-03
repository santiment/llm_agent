"""Shared pytest fixtures and test helpers for the deep_research_agent test suite.

A handful of test modules are also runnable as plain Python scripts (``python
tests/test_x.py``) and call their ``test_*`` functions with zero arguments, so the
event-capture helper is exposed both as a plain context manager (``capture_events_cm``,
usable with or without pytest) and as a pytest fixture (``capture_events``) built on
top of it, for modules with no such standalone-script constraint.
"""

from __future__ import annotations

import asyncio
import contextlib
import os

import pytest
from pydantic import BaseModel

import deep_research_agent.agent as agent_mod
import deep_research_agent.events as events
from deep_research_agent.config import ResearchConfig


@contextlib.contextmanager
def capture_events_cm():
    """Route ``events.emit()`` output into a list for the duration of the ``with`` block, restoring the writer after."""
    captured: list[dict] = []
    orig = events._writer
    events._writer = lambda: captured.append
    try:
        yield captured
    finally:
        events._writer = orig


@pytest.fixture
def capture_events():
    """Pytest fixture yielding the list that ``events.emit()`` writes captured events into during the test."""
    with capture_events_cm() as captured:
        yield captured


def make_state(*messages) -> dict:
    """Build a minimal agent state dict (``{"messages": [...]}``) from a sequence of messages."""
    return {"messages": list(messages)}


def is_nudge(update, name) -> bool:
    """True if a middleware `update` dict includes a message named `name` (a nudge)."""
    msgs = (update or {}).get("messages") or []
    return any(getattr(m, "name", None) == name for m in msgs)


class EmptyArgs(BaseModel):
    """Pydantic ``args_schema`` for a fake tool that takes no arguments."""


class FakeTool:
    """Minimal async tool stand-in exposing only what ``instrument_tool`` reads."""

    def __init__(self, behavior, name="fake_mcp_tool"):
        self.name = name
        self.description = "fake"
        self.args_schema = EmptyArgs
        self._behavior = behavior
        self.calls = 0

    async def ainvoke(self, _kwargs):
        self.calls += 1
        return await self._behavior()


class StubGraph:
    """Stand-in returned by a stubbed ``create_deep_agent()``; absorbs ``make_graph``'s ``.with_config()``."""

    def with_config(self, *args, **kwargs):
        return self


def make_graph_capture(monkeypatch, config: dict) -> dict:
    """Run ``agent_mod.make_graph(config)`` with ``create_deep_agent`` stubbed; return its captured kwargs."""
    captured: dict = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return StubGraph()

    monkeypatch.setattr(agent_mod, "create_deep_agent", fake_create_deep_agent)
    asyncio.run(agent_mod.make_graph(config))
    return captured


def build_config(env_keys, env: dict[str, str] | None = None, **configurable) -> ResearchConfig:
    """Build a ResearchConfig with `env_keys` masked from the environment, optionally setting some via `env`."""
    saved = {k: os.environ.pop(k, None) for k in env_keys}
    os.environ.update(env or {})
    try:
        return ResearchConfig.from_runnable_config({"configurable": configurable})
    finally:
        for k in env_keys:
            os.environ.pop(k, None)
        os.environ.update({k: v for k, v in saved.items() if v is not None})
