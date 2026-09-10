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

import pytest

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


def _load_raw(monkeypatch):
    """Import the plugin with the real ``_post_json`` (its own error handling under test)."""
    monkeypatch.setenv("DRA_METRICS_HUB_URL", "http://metrics-hub.invalid:3000")
    spec = importlib.util.spec_from_file_location("social_messages_raw", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_failed_free_text_retry_is_not_reported_as_no_crowd_data(monkeypatch) -> None:
    """A dead retry must not turn an empty slug result into an authoritative silence claim."""
    tool, _, mod = _load(monkeypatch, {})
    calls: list[dict] = []

    def flaky(url: str, body: dict) -> dict:
        calls.append(body)
        if "slug" not in body:
            raise RuntimeError(f"metrics-hub unreachable at {url}: [Errno 61] Connection refused")
        return _payload(0, 0)

    monkeypatch.setattr(mod, "_post_json", flaky)
    out = _run(tool, asset="santiment")
    assert len(calls) == 2 and "search_text" in calls[1]
    assert "UNKNOWN whether crowd data exists" in out
    assert "Errno 61" in out            # the real cause reaches the model
    assert mod._NO_DATA_NOTE not in out  # and the silence claim does not


def test_http_error_body_beats_the_status_line(monkeypatch) -> None:
    """urlopen raises on 5xx and the exception IS the response; its body must survive."""
    import io
    import urllib.error
    mod = _load_raw(monkeypatch)
    exc = urllib.error.HTTPError(
        "http://metrics-hub.invalid:3000/sample_documents", 500, "INTERNAL SERVER ERROR", {},
        io.BytesIO(json.dumps({"error": "bad_window", "trace": "T" * 4000}).encode()))
    # `error` kept, 4k `trace` dropped rather than truncated into the model's context
    assert mod._error_detail(exc) == "bad_window"
    assert str(exc) == "HTTP Error 500: INTERNAL SERVER ERROR"  # what it used to be alone


def test_transport_failures_name_the_host(monkeypatch) -> None:
    import urllib.error
    mod = _load_raw(monkeypatch)
    url = "http://metrics-hub.invalid:3000/sample_documents"

    def raising(exc):
        monkeypatch.setattr(mod.urllib.request, "urlopen",
                            lambda req, timeout=None: (_ for _ in ()).throw(exc))

    raising(urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")))
    with pytest.raises(RuntimeError, match=r"unreachable at http://metrics-hub\.invalid:3000"):
        mod._post_json(url, {})

    raising(TimeoutError("timed out"))  # a body-read timeout arrives unwrapped
    with pytest.raises(RuntimeError, match=r"timed out after 60s at http://metrics-hub\.invalid"):
        mod._post_json(url, {})


def test_non_json_200_shows_the_body(monkeypatch) -> None:
    """A wrong port or a proxy answers 200 with HTML; a bare JSONDecodeError says nothing."""
    mod = _load_raw(monkeypatch)

    class _Resp:
        def read(self): return b"<html>502 Bad Gateway</html>"
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req, timeout=None: _Resp())
    with pytest.raises(RuntimeError, match="non-JSON response.*Bad Gateway"):
        mod._post_json("http://metrics-hub.invalid:3000/sample_documents", {})


def test_invalid_sources_rejected_before_any_request(monkeypatch) -> None:
    tool, calls, mod = _load(monkeypatch, {"slug": _payload(1, 1)})
    out = _run(tool, asset="bitcoin", sources="twitter, telegram")
    assert calls == []
    assert out.startswith("social_messages: unknown source(s) twitter.")
    assert ", ".join(mod.VALID_SOURCES) in out
    assert "twitter_crypto" in mod._DESCRIPTION



def test_capitalized_slug_is_a_slug_not_a_text_search(monkeypatch):
    tool, calls, _ = _load(monkeypatch, {"slug": _payload(3, 1)})
    out = json.loads(asyncio.run(tool.coroutine(asset="  Bitcoin ")))
    assert calls[0]["slug"] == "bitcoin" and "search_text" not in calls[0]
    assert out["stats"]["total_matching"] == 3


def test_empty_asset_is_refused_before_any_request(monkeypatch):
    tool, calls, _ = _load(monkeypatch, {})
    out = asyncio.run(tool.coroutine(asset="   "))
    assert "`asset` is required" in out and calls == []


def test_missing_stats_block_is_an_error_not_no_crowd_data(monkeypatch):
    tool, _, _ = _load(monkeypatch, {"slug": {"data": {"stats": None, "messages": []}}})
    out = asyncio.run(tool.coroutine(asset="bitcoin"))
    assert out.startswith("social_messages: response carries no stats block")
    assert "no crowd data" not in out.lower() and "No social messages matched" not in out
