"""The Santiment ``social_messages`` custom tool (custom_tools/social_messages.py):
invalid ``sources`` are rejected client-side with the valid list; a project slug that
matches nothing falls back to a free-text search for the same word; a truly empty window
carries an explicit "no crowd data" note so the model does not paper over it with web
search. HTTP is stubbed — nothing here talks to metrics-hub."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "custom_tools" / "social_messages.py"


def _load(monkeypatch, responses: dict[str, dict]):
    """Import the plugin fresh with a canned ``_post_json``; returns (tool, calls)."""
    monkeypatch.setenv("DRA_METRICS_HUB_URL", "http://metrics-hub.invalid:3000")
    spec = importlib.util.spec_from_file_location("social_messages_under_test", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls: list[dict] = []

    def fake_post(url: str, body: dict) -> dict:
        calls.append(body)
        key = "slug" if "slug" in body else "search_text"
        return responses[key]

    monkeypatch.setattr(mod, "_post_json", fake_post)
    (tool,) = mod.build_tools(None)
    return tool, calls, mod


def _payload(total: int, n_msgs: int) -> dict:
    return {"data": {"stats": {"total_matching": total, "sampled": n_msgs},
                     "messages": [{"text": f"m{i}", "stratum": "random"} for i in range(n_msgs)]}}


def _run(tool, **kwargs) -> str:
    return asyncio.run(tool.coroutine(**kwargs))


def test_slug_with_data_is_a_single_request(monkeypatch) -> None:
    tool, calls, _ = _load(monkeypatch, {"slug": _payload(1249, 3)})
    out = json.loads(_run(tool, asset="bitcoin", from_timestamp="utc_now-24h"))
    assert len(calls) == 1 and calls[0]["slug"] == "bitcoin"
    assert calls[0]["from_timestamp"] == "now-24h"  # Santiment date math normalized
    assert out["stats"]["total_matching"] == 1249 and len(out["messages"]) == 3
    assert "note" not in out


def test_empty_slug_falls_back_to_free_text(monkeypatch) -> None:
    tool, calls, _ = _load(monkeypatch, {"slug": _payload(0, 0), "search_text": _payload(42, 2)})
    out = json.loads(_run(tool, asset="santiment", sources="telegram, reddit"))
    assert [("slug" in c, c.get("search_text")) for c in calls] == [(True, None), (False, "santiment")]
    assert calls[1]["sources"] == "telegram,reddit" and "slug" not in calls[1]
    assert out["stats"]["total_matching"] == 42
    assert out["stats"]["query_mode"] == "search_text"
    assert "santiment" in out["stats"]["note"]


def test_truly_empty_window_carries_no_data_note(monkeypatch) -> None:
    tool, calls, _ = _load(monkeypatch, {"slug": _payload(0, 0), "search_text": _payload(0, 0)})
    out = json.loads(_run(tool, asset="santiment"))
    assert len(calls) == 2
    assert out["messages"] == []
    assert "do not fill the gap with web search" in out["note"]


def test_invalid_sources_rejected_before_any_request(monkeypatch) -> None:
    tool, calls, mod = _load(monkeypatch, {"slug": _payload(1, 1)})
    out = _run(tool, asset="bitcoin", sources="twitter, telegram")
    assert calls == []
    assert out.startswith("social_messages: unknown source(s) twitter.")
    assert ", ".join(mod.VALID_SOURCES) in out
    assert "twitter_crypto" in mod._DESCRIPTION
