"""``web_fetch``: one URL → its readable text (HTML converted with the stdlib parser,
JSON/text passed through). Registered through ``events.instrument_tool`` like every data
tool, so an oversized page offloads to the sandbox instead of flooding context.

SSRF policy is ``config.url_blocked`` with ``allow_private=False`` — the URL is
model-supplied. Redirects are followed by hand so every hop is vetted BEFORE it is
requested. A DNS name resolving to a private address is the documented residual gap.
"""

from __future__ import annotations

import asyncio
import re
from html.parser import HTMLParser

import httpx
from langchain_core.tools import StructuredTool

from ..config import ResearchConfig, url_blocked
from ..events import source_events

MAX_FETCH_BYTES = 2_000_000
MAX_REDIRECTS = 5
_UA = "Mozilla/5.0 (compatible; deep-research-agent/1.0)"

_TEXT_CTYPES = {"application/json", "application/xml", "application/xhtml+xml"}
_HTML_CTYPES = {"text/html", "application/xhtml+xml"}


def fetch_url_blocked(url: str) -> str | None:
    return url_blocked(url, allow_private=False)


# --- HTML → readable text --------------------------------------------------------

_DROP = {"script", "style", "noscript", "template", "svg", "iframe"}
_BLOCK = {"p", "div", "section", "article", "header", "footer", "main", "aside",
          "nav", "table", "tr", "ul", "ol", "blockquote", "figure", "form", "br",
          "pre", "hr"}
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


class _HTMLToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._drop_depth = 0
        self._in_title = False
        self._href: str | None = None
        self._link_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _DROP:
            self._drop_depth += 1
            return
        if self._drop_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag in _HEADINGS:
            self.parts.append("\n\n" + "#" * _HEADINGS[tag] + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "a" and self._href is None:
            self._href = dict(attrs).get("href") or ""
            self._link_text = []
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _DROP:
            self._drop_depth = max(0, self._drop_depth - 1)
            return
        if self._drop_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag == "a" and self._href is not None:
            text = "".join(self._link_text).strip()
            href = self._href
            self._href = None
            if text:
                # Absolute links only — the model needs citable URLs.
                keep = href.startswith(("http://", "https://")) and href != text
                self.parts.append(f"{text} ({href})" if keep else text)
        elif tag in _HEADINGS or tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._drop_depth:
            return
        if self._in_title:
            self.title += data
        elif self._href is not None:
            self._link_text.append(data)
        else:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    """``(title, readable_text)`` from an HTML document."""
    parser = _HTMLToText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 — a malformed page degrades, never raises
        pass
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return parser.title.strip(), text


# --- the tool ----------------------------------------------------------------------

def _refuse(reason: str) -> RuntimeError:
    return RuntimeError(f"[permanent] Fetch refused ({reason}) — this URL cannot be fetched.")


def build_fetch_tool(cfg: ResearchConfig) -> StructuredTool:
    # One client per run for connection pooling; httpx cleans it up at GC. Redirects are
    # followed manually so each Location is vetted before it is requested.
    client = httpx.AsyncClient(follow_redirects=False, timeout=cfg.request_timeout,
                               headers={"User-Agent": _UA})

    async def web_fetch(url: str) -> str:
        # Failures RAISE tagged errors: instrument_tool meters them as failures, memoizes
        # [permanent] ones per-args and turns them into model-facing text.
        blocked = fetch_url_blocked(url)
        if blocked:
            raise _refuse(blocked)
        try:
            for hop in range(MAX_REDIRECTS + 1):
                async with client.stream("GET", url) as resp:
                    if resp.next_request is not None:
                        if hop == MAX_REDIRECTS:
                            raise RuntimeError("[permanent] Fetch failed: too many redirects.")
                        url = str(resp.next_request.url)
                        blocked = fetch_url_blocked(url)
                        if blocked:
                            raise RuntimeError(f"[permanent] Fetch refused after redirect ({blocked}).")
                        continue
                    if resp.status_code >= 400:
                        tag = ("transient" if resp.status_code in (408, 429)
                               or resp.status_code >= 500 else "permanent")
                        raise RuntimeError(f"[{tag}] Fetch failed: HTTP {resp.status_code} for {url}.")
                    ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                    if ctype and not ctype.startswith("text/") and ctype not in _TEXT_CTYPES:
                        raise RuntimeError(f"[permanent] Fetch skipped: content type {ctype!r} is "
                                           "not readable text (only HTML / text / JSON pages can "
                                           "be fetched).")
                    buf = bytearray()
                    truncated = False
                    async for chunk in resp.aiter_bytes():
                        buf += chunk
                        if len(buf) >= MAX_FETCH_BYTES:
                            truncated = True
                            break
                    enc = resp.encoding or "utf-8"
                    break
        except httpx.HTTPError as exc:
            raise RuntimeError(f"[transient] Fetch failed ({type(exc).__name__}): {exc}") from exc

        def _decode_and_extract() -> tuple[str, str]:
            body = bytes(buf[:MAX_FETCH_BYTES]).decode(enc, errors="replace")
            if ctype in _HTML_CTYPES or (not ctype and body.lstrip()[:1] == "<"):
                return html_to_text(body)
            return "", body

        # Decoding + parsing a 2 MB page is CPU-bound; keep it off the event loop.
        title, text = await asyncio.to_thread(_decode_and_extract)
        if not text.strip():
            return f"Fetched {url} but found no readable text on the page."

        source_events([{"title": title or url, "url": url}])
        header = f"# {title}\n" if title else ""
        note = "\n\n[Page truncated at the fetch size limit.]" if truncated else ""
        return f"{header}URL: {url}\n\n{text}{note}"

    return StructuredTool.from_function(
        coroutine=web_fetch,
        name="web_fetch",
        description=(
            "Fetch ONE web page by URL and return its full readable text (HTML is "
            "converted to plain text; JSON/text returned as-is). Use it to READ a page "
            "whose search snippet is not enough — never cite a page for substantive "
            "claims when you have only seen its snippet. Not a search tool (use "
            "web_search to find pages first)."
        ),
    )
