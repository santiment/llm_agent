"""Typed streaming-event protocol — THE contract any frontend renders.

Events are emitted on LangGraph's ``custom`` stream channel via
``get_stream_writer()``. They are plain JSON dicts (so any HTTP/SSE client can
consume them) with a ``type`` discriminator. The agent core is the only
producer; your app is just a consumer — this is what keeps the agent portable.

Render mapping (Claude / Gemini deep-research UIs):
  - ``phase``          -> collapsible section header ("Structuring the Investigation")
  - ``search_query``   -> the globe row ("how to analyze key metrics")
  - ``search_results`` -> the favicon + title grid ("7 results")
  - ``tool_call`` /
    ``tool_result``    -> MCP call rows
  - ``source``         -> registered citation (for the live source list)
  - ``skill``          -> a skill being applied ("Skill: data-provider")
  - ``report``         -> final markdown answer (also persisted in state)
  - ``status``         -> lifecycle: researching | writing | done | error

Assistant *reasoning* prose (the italic narration between steps) is NOT a custom
event — it streams on the ``messages`` channel as normal AI tokens, so the UI
puts it in the "show thinking process" pane.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from contextlib import nullcontext
from typing import Any, Iterable
from urllib.parse import urlparse

from langchain_core.tools import BaseTool, StructuredTool

log = logging.getLogger("deep_research_agent.events")


def new_id() -> str:
    """Short correlation id linking a *_call event to its *_result event."""
    return uuid.uuid4().hex[:8]


def domain_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").removeprefix("www.")
    except Exception:
        return ""


def _writer():
    """Stream writer if we're inside a streamed run, else None (tests/CLI)."""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:
        return None


def emit(event: dict[str, Any]) -> None:
    """Push one protocol event onto the ``custom`` stream channel (no-op offline)."""
    w = _writer()
    if w is not None:
        try:
            w(event)
        except Exception:
            # Streaming is best-effort observability — never break the run for it.
            pass


def _summarize(value: Any, limit: int = 280) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


_RETRY_AFTER = re.compile(r"retry[- ]after[:\s]+(\d+(?:\.\d+)?)", re.IGNORECASE)


def _is_rate_limited(msg: str) -> bool:
    """Best-effort detection of an upstream "slow down" signal in a (lowercased)
    error message. MCP tool errors surface as plain exceptions whose message carries
    the HTTP status / text, so we match on the few unambiguous markers rather than
    the exception type."""
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


# A failed tool call must NOT kill the run (one mistyped metric name used to abort the
# whole research) — the error text goes back to the model as the tool result so it can
# self-correct. Classification decides the guidance: PERMANENT (validation /
# unknown-name — the same arguments can never succeed) vs TRANSIENT (worth one retry).
# Servers that know best can tag explicitly: a message starting with "[permanent]" or
# "[transient]" wins over the marker heuristics.
_PERMANENT_MARKERS = (
    "not supported", "not found", "unknown", "invalid", "unsupported",
    "does not exist", "must be one of", "mistyped", "missing required",
    "can contain at most", "not available for", "bad request",
)
_TRANSIENT_MARKERS = (
    "timeout", "timed out", "connection", "temporarily", "unavailable",
    "internal server error", "500", "502", "503", "504",
)


def classify_tool_error(msg: str) -> str:
    """``"permanent"`` | ``"transient"`` | ``"unknown"`` for a tool error message."""
    low = msg.strip().lower()
    if low.startswith("[permanent]"):
        return "permanent"
    if low.startswith("[transient]"):
        return "transient"
    if any(m in low for m in _PERMANENT_MARKERS):
        return "permanent"
    if any(m in low for m in _TRANSIENT_MARKERS):
        return "transient"
    return "unknown"


def _strip_class_tag(msg: str) -> str:
    low = msg.lstrip().lower()
    for tag in ("[permanent]", "[transient]"):
        if low.startswith(tag):
            return msg.lstrip()[len(tag):].lstrip()
    return msg


_ERROR_GUIDANCE = {
    "permanent": (
        "This error is PERMANENT for these arguments — repeating the identical call WILL "
        "fail again. Fix the arguments (e.g. resolve valid names with a discovery tool) "
        "or proceed without this data and note the gap."
    ),
    "transient": (
        "This error is likely TRANSIENT. You may retry this call ONCE; if it fails "
        "again, proceed without this data and note the gap."
    ),
    "unknown": (
        "If a retry with the SAME arguments fails again, treat the error as permanent: "
        "fix the arguments or proceed without this data and note the gap."
    ),
}


def tool_error_text(tool_name: str, msg: str, classification: str) -> str:
    """The model-facing tool result for a failed call: the error + how to proceed."""
    return (f"TOOL ERROR ({tool_name}, {classification}): "
            f"{_summarize(_strip_class_tag(msg), 1000)}\n"
            + _ERROR_GUIDANCE[classification])


def _retry_after_seconds(msg: str) -> float | None:
    """Honor a server-provided ``Retry-After`` hint if it leaked into the message."""
    m = _RETRY_AFTER.search(msg)
    return float(m.group(1)) if m else None


def cap_result(result: Any, *, max_chars: int = 0, max_rows: int = 0) -> tuple[Any, str | None]:
    """Bound a tool result's size BEFORE it enters the model's context.

    Returns ``(possibly_smaller_result, note_or_None)``; each limit is disabled when 0.
    This is the source-level guard for the failure where many medium-sized MCP results —
    each under the per-message eviction threshold — silently piled up until the context
    blew past the model's limit. Capping here also defuses the ``read_file`` re-inflation
    path, since there is no longer a huge offloaded result to read back. The appended note
    nudges the model toward narrowing the query / using an aggregate tool, not paging more.
    """
    # Row cap for list-shaped results (e.g. holdings rows).
    if max_rows and isinstance(result, list) and len(result) > max_rows:
        note = f"{len(result) - max_rows} of {len(result)} rows omitted"
        capped = list(result[:max_rows])
        capped.append({"_truncated": note,
                       "_hint": "Result capped — add filters or request fewer rows."})
        return capped, note
    # Char cap for string results (the common MCP shape), or anything large once stringified.
    if max_chars and isinstance(result, str) and len(result) > max_chars:
        note = f"{len(result) - max_chars} of {len(result)} chars omitted"
        return (
            result[:max_chars]
            + f"\n\n[truncated: {note}. Add filters or request fewer rows.]"
        ), note
    return result, None


def _offload_result(
    result: Any,
    *,
    sink: Any,
    offload_dir: str,
    tool_name: str,
    call_id: str,
    head_rows: int = 5,
) -> tuple[str | None, str | None]:
    """Persist a large tool result to a file in the sandbox and return a compact stub
    the model can act on, INSTEAD of truncating and discarding rows.

    The full result lands at ``{offload_dir}/{tool}-{call_id}.json`` inside the
    container's persistent /workspace; the stub carries the path, row count, column
    list and a small head, plus an instruction to process the file with ``execute``.
    Returns ``(stub, note)`` on success, or ``(None, None)`` if anything went wrong —
    the caller then falls back to ``cap_result`` so a flaky sandbox never loses data
    silently or breaks the run.
    """
    try:
        rows: list | None = None
        if isinstance(result, list):
            rows = result
            payload = json.dumps(result, default=str)
        elif isinstance(result, str):
            payload = result
            try:
                parsed = json.loads(result)
                rows = parsed if isinstance(parsed, list) else None
            except (ValueError, TypeError):
                rows = None
        else:
            payload = json.dumps(result, default=str)

        path = f"{offload_dir.rstrip('/')}/{tool_name}-{call_id}.json"
        resp = sink.upload_files([(path, payload.encode("utf-8"))])
        if resp and getattr(resp[0], "error", None):
            log.warning("offload upload failed (%s): %s", tool_name, resp[0].error)
            return None, None
    except Exception as exc:  # never lose data silently / break the run on offload failure
        log.warning("offload failed (%s): %s", tool_name, exc)
        return None, None

    n = len(rows) if isinstance(rows, list) else None
    columns = ""
    head = ""
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        columns = ", ".join(map(str, list(rows[0].keys())[:40]))
        head = json.dumps(rows[:head_rows], default=str)[:2000]
    note = f"offloaded to {path} ({n if n is not None else '?'} rows, {len(payload)} bytes)"
    stub = (
        "[Large result saved to a file to keep the context small — NOT shown inline.]\n"
        f"file: {path}\n"
        f"format: {'JSON array of objects' if n is not None else 'JSON'}\n"
        + (f"rows: {n}\n" if n is not None else "")
        + (f"columns: {columns}\n" if columns else "")
        + (f"preview (first {head_rows} rows):\n{head}\n" if head else "")
        + "\nThis file holds the COMPLETE result. To use it, call the `execute` tool to load "
        "and analyze the file (Python/pandas or duckdb over the JSON) — compute aggregates, "
        "joins, or filters there. Do NOT re-call this tool to page the same rows."
    )
    return stub, note


def instrument_tool(
    tool: BaseTool,
    kind: str = "tool",
    *,
    semaphore: asyncio.Semaphore | None = None,
    rate_limit_max_wait: float = 0.0,
    base_delay: float = 0.5,
    max_delay: float = 20.0,
    max_result_chars: int = 0,
    max_result_rows: int = 0,
    meter: Any = None,
    offload_sink: Any = None,
    offload_dir: str = "/workspace/data",
) -> BaseTool:
    """Wrap any tool so each invocation emits ``{kind}_call`` / ``{kind}_result``.

    Preserves the original name / description / args schema so the model is
    unaware of the wrapper. Used for MCP tools; the web-search tool emits its
    own richer events instead.

    ``semaphore`` bounds how many wrapped tools may run *at once* — a shared one
    acts as a fixed-size queue across the orchestrator and all parallel
    sub-researchers, so the agent's fan-out can't open unbounded MCP connections.

    On a rate-limit signal the call does NOT fail — it releases its slot and waits
    (honoring ``Retry-After`` when present, else capped exponential backoff), then
    retries. It keeps waiting until the cumulative backoff would exceed
    ``rate_limit_max_wait`` seconds, a budget that stops a permanently-throttled
    server from hanging the run forever. ``0`` disables retry (fail on first 429).
    Connection / fd errors are never retried — that would just hammer a struggling
    server.
    """

    # Permanent failures remembered per (args) so an identical retry is answered
    # locally instead of hammering the server. Per-run state: make_graph builds the
    # wrappers fresh for every run.
    failed_permanently: dict[str, str] = {}

    async def _run(**kwargs: Any) -> Any:
        call_id = new_id()
        emit({
            "type": f"{kind}_call",
            "id": call_id,
            "tool": tool.name,
            "args": {k: _summarize(v, 120) for k, v in kwargs.items()},
        })
        args_key = json.dumps(kwargs, sort_keys=True, default=str)
        if args_key in failed_permanently:
            emit({"type": f"{kind}_result", "id": call_id, "tool": tool.name,
                  "ok": False, "error_class": "permanent", "repeated": True,
                  "summary": _summarize(failed_permanently[args_key])})
            return (f"TOOL ERROR ({tool.name}, permanent, REPEATED CALL): you already "
                    f"called this tool with these EXACT arguments and it failed: "
                    f"{_summarize(failed_permanently[args_key], 500)}\n"
                    "Do NOT repeat this call. " + _ERROR_GUIDANCE["permanent"])
        attempt = 0
        waited = 0.0
        while True:
            try:
                # The semaphore (a shared, fixed-size queue) is held only for the
                # duration of the call — released on exit so we never hold a slot
                # while backing off on a 429.
                async with (semaphore or nullcontext()):
                    result = await tool.ainvoke(kwargs)
            except Exception as exc:
                # A failed call is a RESULT, not a run-ending event: the error text is
                # returned to the model (with retry guidance) so it can self-correct —
                # raising here would abort the whole research over one bad argument.
                msg = str(exc)
                low = msg.lower()
                # Wait out a rate-limit signal rather than failing — but only within
                # the budget. Compute the delay only on this (rate-limited) path.
                if rate_limit_max_wait > 0 and _is_rate_limited(low):
                    delay = _retry_after_seconds(low) or min(max_delay, base_delay * (2 ** attempt))
                    if waited + delay <= rate_limit_max_wait:
                        attempt += 1
                        waited += delay
                        await asyncio.sleep(delay)
                        continue
                classification = classify_tool_error(msg)
                if classification == "permanent":
                    if len(failed_permanently) >= 128:  # bound per-run memory
                        failed_permanently.pop(next(iter(failed_permanently)))
                    failed_permanently[args_key] = _strip_class_tag(msg)
                if meter is not None:
                    meter.record_tool_result(ok=False)
                emit({"type": f"{kind}_result", "id": call_id, "tool": tool.name,
                      "ok": False, "error_class": classification,
                      "summary": _summarize(msg)})
                log.warning("TOOL ERROR (%s, %s): %s", tool.name, classification,
                            _summarize(msg, 500))
                return tool_error_text(tool.name, msg, classification)
            # Observability + source-level cap: record the RAW size (before capping) so the
            # run log shows what the tool actually returned, then bound it for the context.
            raw_rows = len(result) if isinstance(result, (list, tuple)) else None
            raw_bytes = len(result) if isinstance(result, str) else len(repr(result))
            # Is this result too big for context? (over EITHER the row or char threshold).
            too_big = bool(
                (max_result_rows and raw_rows is not None and raw_rows > max_result_rows)
                or (max_result_chars and raw_bytes > max_result_chars)
            )
            capped: str | None = None
            # Prefer OFFLOAD to a sandbox file over truncation: keeps the full data
            # available (the model reads it back with `execute`) instead of dropping rows.
            if too_big and offload_sink is not None:
                stub, note = _offload_result(
                    result, sink=offload_sink, offload_dir=offload_dir,
                    tool_name=tool.name, call_id=call_id)
                if stub is not None:
                    result, capped = stub, note
                    log.info("RESULT OFFLOADED (%s): %s [raw: %d bytes, rows=%s]",
                             tool.name, note, raw_bytes, raw_rows)
            if capped is None:
                # No sandbox (or offload failed) → fall back to the truncation caps.
                result, capped = cap_result(
                    result, max_chars=max_result_chars, max_rows=max_result_rows)
                if capped:
                    log.info("RESULT CAPPED (%s): %s [raw: %d bytes, rows=%s]",
                             tool.name, capped, raw_bytes, raw_rows)
            if meter is not None:
                meter.record_tool_result(ok=True, result_bytes=raw_bytes,
                                         result_rows=raw_rows, capped=bool(capped))
            emit({"type": f"{kind}_result", "id": call_id, "tool": tool.name,
                  "ok": True, "summary": _summarize(result),
                  "bytes": raw_bytes, "rows": raw_rows, "capped": bool(capped)})
            return result

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_run,
    )


def source_events(results: Iterable[dict[str, Any]]) -> None:
    """Emit one ``source`` event per result for the live citation list."""
    for r in results:
        emit({"type": "source", "title": r.get("title", ""),
              "url": r.get("url", ""), "domain": domain_of(r.get("url", ""))})
