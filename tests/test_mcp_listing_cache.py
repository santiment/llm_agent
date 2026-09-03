"""MCP tool listings are reused across graph builds within the TTL; failures never are."""

from __future__ import annotations

import asyncio

import deep_research_agent.tools.mcp as mcp_mod
from conftest import EmptyArgs, build_config


class _Tool:
    def __init__(self, name):
        self.name = name
        self.description = "fake"
        self.args_schema = EmptyArgs

    async def ainvoke(self, _kwargs):
        return "ok"


class _FakeClient:
    listings = 0
    fail = False

    def __init__(self, connections):
        self.connections = connections

    async def get_tools(self, server_name):
        type(self).listings += 1
        if type(self).fail:
            raise RuntimeError("connection refused")
        return [_Tool("get_metric"), _Tool("get_trends")]


def _cfg(ttl, url="http://127.0.0.1:8765/mcp"):
    return build_config(("DRA_MCP_URL", "DRA_MCP_SERVERS", "DRA_MCP_TOOLS_TTL"),
                        openai_api_key="k", mcp_tools_ttl=ttl,
                        mcp_servers=[{"name": "data", "label": "Data", "url": url}])


def _load(cfg):
    return asyncio.run(mcp_mod.load_mcp_tools(cfg))


def _install(monkeypatch):
    import langchain_mcp_adapters.client as client_mod
    _FakeClient.listings = 0
    _FakeClient.fail = False
    monkeypatch.setattr(client_mod, "MultiServerMCPClient", _FakeClient)
    mcp_mod.clear_listing_cache()


def test_second_build_reuses_the_listing(monkeypatch, capture_events) -> None:
    _install(monkeypatch)
    cfg = _cfg(600)
    first, second = _load(cfg), _load(cfg)
    assert _FakeClient.listings == 1
    assert [t.name for t in first] == [t.name for t in second] == ["get_metric", "get_trends"]
    # Instrumentation is per build: distinct wrapper objects around the cached listing.
    assert first[0] is not second[0]
    assert [e["state"] for e in capture_events if e.get("type") == "status"] == ["mcp_ready", "mcp_ready"]


def test_ttl_zero_lists_every_build(monkeypatch) -> None:
    _install(monkeypatch)
    cfg = _cfg(0)
    _load(cfg)
    _load(cfg)
    assert _FakeClient.listings == 2


def test_different_server_is_a_different_entry(monkeypatch) -> None:
    _install(monkeypatch)
    _load(_cfg(600))
    _load(_cfg(600, url="http://127.0.0.1:8766/mcp"))
    assert _FakeClient.listings == 2


def test_expired_entry_is_relisted(monkeypatch) -> None:
    _install(monkeypatch)
    cfg = _cfg(600)
    now = [1000.0]
    monkeypatch.setattr(mcp_mod.time, "monotonic", lambda: now[0])
    _load(cfg)
    _load(cfg)                       # still inside the window
    now[0] = 10 ** 9                 # far future → expired
    _load(cfg)
    assert _FakeClient.listings == 2


def test_failures_are_not_cached(monkeypatch, capture_events) -> None:
    _install(monkeypatch)
    cfg = _cfg(600)
    _FakeClient.fail = True
    assert _load(cfg) == []
    assert any(e.get("state") == "mcp_error" for e in capture_events)
    _FakeClient.fail = False
    assert [t.name for t in _load(cfg)] == ["get_metric", "get_trends"]
    assert _FakeClient.listings == 2
