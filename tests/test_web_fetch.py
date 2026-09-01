"""Pin the web_fetch tool (tools/fetch.py): the stdlib HTML→text converter, the
model-supplied-URL SSRF policy (initial URL and every redirect hop), and the
happy path / content-type gate over a mocked HTTP transport.

Runs with plain Python (``python tests/test_web_fetch.py``) — no pytest needed — and
is also pytest-discoverable. No network.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx

from deep_research_agent.tools import fetch as fetch_mod
from deep_research_agent.tools.fetch import (build_fetch_tool, fetch_url_blocked,
                                             html_to_text)

# build_fetch_tool only reads request_timeout off the config.
_CFG = SimpleNamespace(request_timeout=5.0)

# --- SSRF policy --------------------------------------------------------------

def test_public_urls_pass() -> None:
    assert fetch_url_blocked("https://example.com/page") is None
    assert fetch_url_blocked("http://sub.example.com:8080/a?b=c") is None


def test_private_and_special_targets_are_blocked() -> None:
    for url in (
        "ftp://example.com/x",             # scheme
        "file:///etc/passwd",              # scheme
        "https:///nopath",                 # missing host
        "http://localhost:8080/admin",
        "http://127.0.0.1/",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",  # AWS metadata
        "http://metadata.google.internal/computeMetadata/",
        "http://[::1]/",
        "http://0.0.0.0/",
    ):
        assert fetch_url_blocked(url) is not None, url


# --- HTML → text ---------------------------------------------------------------

_HTML = """<html><head><title> Page Title </title>
<script>var secret = 1;</script><style>.x{color:red}</style></head>
<body><h1>Main Heading</h1><p>First paragraph.</p>
<h2>Sub</h2><ul><li>One</li><li>Two</li></ul>
<p>See <a href="https://example.com/ref">the reference</a> and
<a href="/relative">local link</a>.</p></body></html>"""


def test_html_to_text_readable_output() -> None:
    title, text = html_to_text(_HTML)
    assert title == "Page Title"
    assert "# Main Heading" in text and "## Sub" in text
    assert "- One" in text and "- Two" in text
    assert "the reference (https://example.com/ref)" in text  # absolute link kept
    assert "local link" in text and "(/relative)" not in text  # relative href dropped
    assert "secret" not in text and "color:red" not in text    # script/style gone


# --- the tool over a mocked transport -------------------------------------------

def _fetch(url: str, handler) -> str:
    """Run the tool against a MockTransport by patching httpx.AsyncClient in the module."""
    real = httpx.AsyncClient
    fetch_mod.httpx.AsyncClient = (
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    try:
        tool = build_fetch_tool(_CFG)
        return asyncio.run(tool.coroutine(url=url))
    finally:
        fetch_mod.httpx.AsyncClient = real


def _fetch_error(url: str, handler) -> str:
    """The tagged error message a failing fetch raises (instrument_tool converts these
    to model-facing text in production — see events.instrument_tool)."""
    try:
        _fetch(url, handler)
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError("fetch was expected to raise")


def test_fetch_returns_page_text() -> None:
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"},
                              content=_HTML.encode())

    out = _fetch("https://example.com/article", handler)
    assert out.startswith("# Page Title")
    assert "URL: https://example.com/article" in out
    assert "# Main Heading" in out


def test_fetch_refuses_redirect_to_private_address() -> None:
    def handler(request):
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/latest"})
        return httpx.Response(200, headers={"content-type": "text/html"},
                              content=b"<p>internal</p>")

    err = _fetch_error("https://example.com/go", handler)
    assert "[permanent]" in err and "refused after redirect" in err
    assert "internal" not in err


def test_fetch_skips_binary_content_types() -> None:
    def handler(request):
        return httpx.Response(200, headers={"content-type": "application/pdf"},
                              content=b"%PDF-1.7")

    err = _fetch_error("https://example.com/file.pdf", handler)
    assert "[permanent]" in err and "not readable text" in err


def test_fetch_http_errors_raise_tagged() -> None:
    def handler(request):
        codes = {"missing": 404, "flaky": 503}
        return httpx.Response(codes[request.url.path.strip("/")], content=b"nope")

    err = _fetch_error("https://example.com/missing", handler)
    assert "[permanent]" in err and "HTTP 404" in err
    err = _fetch_error("https://example.com/flaky", handler)
    assert "[transient]" in err and "HTTP 503" in err


def test_blocked_url_never_hits_the_network() -> None:
    def handler(request):  # a request here means the SSRF gate failed
        raise AssertionError("network was reached for a blocked URL")

    err = _fetch_error("http://169.254.169.254/latest/meta-data/", handler)
    assert "[permanent]" in err and "refused" in err


if __name__ == "__main__":
    test_public_urls_pass()
    test_private_and_special_targets_are_blocked()
    test_html_to_text_readable_output()
    test_fetch_returns_page_text()
    test_fetch_refuses_redirect_to_private_address()
    test_fetch_skips_binary_content_types()
    test_fetch_http_errors_raise_tagged()
    test_blocked_url_never_hits_the_network()
    print("OK — web_fetch converter, SSRF policy and transport behavior verified.")
