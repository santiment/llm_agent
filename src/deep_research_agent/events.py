"""Typed streaming-event protocol — THE contract any frontend renders.

Events are emitted on LangGraph's ``custom`` stream channel via
``get_stream_writer()``. They are plain JSON dicts (so any HTTP/SSE client can
consume them) with a ``type`` discriminator. The agent core is the only
producer; your app is just a consumer — this is what keeps the agent portable.

Render mapping (Claude / Gemini deep-research UIs) — one line per registered type,
in ``EVENT_SCHEMAS`` order:
  - ``run_start``      -> protocol handshake; pin/verify before rendering anything else
                          (``started_at``, UTC: the anchor for run time if a run dies early)
  - ``search_query``   -> the globe row ("how to analyze key metrics")
  - ``search_results`` -> the favicon + title grid ("7 results")
  - ``source``         -> registered citation (for the live source list)
  - ``mcp_call`` /
    ``mcp_result``     -> MCP call rows (legacy aliases of the two below)
  - ``tool_call`` /
    ``tool_result``    -> tool call rows
  - ``skill``          -> a skill being applied ("Skill: data-provider")
  - ``report``         -> final markdown answer (also persisted in state)
  - ``status``         -> lifecycle: mcp_ready | mcp_error | budget_soft | budget_halt |
                          revising | compacting | compacted | loop_detected | loop_halt |
                          subagent_start | subagent_done (``role`` + ``model`` of the
                          sub-agent) | done | error (the last two are the run's end-state,
                          classified in citations.py; ``reason`` carries the specific code,
                          ``elapsed_s`` / ``elapsed`` the run time, also appended to ``detail``)
  - ``clarification``  -> the questions the agent needs answered before it can proceed
  - ``usage``          -> end-of-run tool-call / token counters against their limits,
                          plus run time (``elapsed_s``, ``elapsed``, ``started_at``, ``finished_at``)
  - ``subagent_findings`` -> one research unit's summary, findings and gaps

Assistant *reasoning* prose (the italic narration between steps) is NOT a custom
event — it streams on the ``messages`` channel as normal AI tokens, so the UI
puts it in the "show thinking process" pane.

THE CONTRACT IS CODE, not just this docstring: ``EVENT_SCHEMAS`` below registers every
event type and its required keys, and ``emit`` warns (never raises) when an event
misses its shape or uses an unregistered type — so drift shows up in this repo's logs
and tests, not in a consumer's broken UI. Versioning for consumers: every run opens
with a ``run_start`` event carrying ``protocol_version`` (bump it on any BREAKING shape
change — a removed/renamed key or type; additions are compatible and don't bump) and
``engine_version`` (the installed package version), so a frontend can pin what it
understands and detect mismatch at run start instead of failing mid-render.
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

from .series import MAX_SCAN_BYTES, SERIES_RULE, find_series, summary_block

log = logging.getLogger("deep_research_agent.events")

# Bump ONLY on a breaking change to a shipped event's shape (removed/renamed key or
# type). Additive keys and new event types are backward-compatible — no bump.
PROTOCOL_VERSION = 1


def engine_version() -> str:
    """Installed package version for the run_start handshake ('unknown' in odd
    environments — the event must never fail over metadata lookup)."""
    try:
        from importlib.metadata import version

        return version("deep-research-agent")
    except Exception:
        return "unknown"


# type -> keys every event of that type MUST carry (beyond "type"). Optional keys are
# deliberately not listed — consumers must tolerate extras. ``emit`` checks each event
# against this registry and WARNS on violation; it never raises (observability must not
# break a run). Adding an event type without registering it here is itself a warning.
EVENT_SCHEMAS: dict[str, frozenset[str]] = {
    "run_start": frozenset({"protocol_version", "engine_version", "started_at"}),
    "search_query": frozenset({"id", "query"}),
    "search_results": frozenset({"id", "query", "ok", "results"}),
    "source": frozenset({"title", "url", "domain"}),
    "mcp_call": frozenset({"id", "tool", "args"}),
    "mcp_result": frozenset({"id", "tool", "ok"}),
    "tool_call": frozenset({"id", "tool", "args"}),
    "tool_result": frozenset({"id", "tool", "ok"}),
    "skill": frozenset({"name", "path", "state"}),
    "report": frozenset({"markdown"}),
    "status": frozenset({"state"}),
    "clarification": frozenset({"questions"}),
    "usage": frozenset({"tool_calls", "input_tokens", "output_tokens",
                        "total_tokens", "limits", "elapsed_s"}),
    "subagent_findings": frozenset({"unit", "summary", "findings", "gaps"}),
}

# Every ``state`` a status event may carry; ``_check_shape`` warns on an unregistered one.
STATUS_STATES = frozenset({
    "mcp_ready", "mcp_error", "budget_soft", "budget_halt", "revising",
    "compacting", "compacted", "loop_detected", "loop_halt", "done", "error",
    "subagent_start", "subagent_done",  # carry ``role`` + ``model``
})


def _check_shape(event: dict[str, Any]) -> None:
    etype = event.get("type")
    required = EVENT_SCHEMAS.get(etype or "")
    if required is None:
        log.warning("EVENT PROTOCOL: unregistered event type %r — register it in "
                    "EVENT_SCHEMAS", etype)
        return
    missing = required - event.keys()
    if missing:
        log.warning("EVENT PROTOCOL: %r event missing required keys %s",
                    etype, sorted(missing))
    if etype == "status" and event.get("state") not in STATUS_STATES:
        log.warning("EVENT PROTOCOL: unregistered status state %r — register it in "
                    "STATUS_STATES", event.get("state"))


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
    _check_shape(event)
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


# The explicit server-side tags, written once: classify_tool_error reads them and
# _strip_class_tag removes them, so the two can't disagree on the spelling.
_CLASS_TAGS = ("permanent", "transient")


def _explicit_class(msg: str) -> str | None:
    """The classification a server tagged the message with (``[permanent]`` /
    ``[transient]`` prefix), or None when it did not tag it."""
    low = msg.lstrip().lower()
    return next((t for t in _CLASS_TAGS if low.startswith(f"[{t}]")), None)


def classify_tool_error(msg: str) -> str:
    """``"permanent"`` | ``"transient"`` | ``"unknown"`` for a tool error message."""
    explicit = _explicit_class(msg)
    if explicit:
        return explicit
    low = msg.strip().lower()
    if any(m in low for m in _PERMANENT_MARKERS):
        return "permanent"
    if any(m in low for m in _TRANSIENT_MARKERS):
        return "transient"
    return "unknown"


def _strip_class_tag(msg: str) -> str:
    tag = _explicit_class(msg)
    return msg.lstrip()[len(tag) + 2:].lstrip() if tag else msg


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
    series: dict | None = None,
) -> tuple[str | None, str | None]:
    """Persist a large tool result to a file in the sandbox and return a compact stub
    the model can act on, INSTEAD of truncating and discarding rows.

    ``series`` — the result's time series (``series.find_series``) if the caller already
    detected them; computed here otherwise. A series result gets a stub with the file path
    and a computed summary instead of any rows.

    The full result lands at ``{offload_dir}/{tool}-{call_id}.json`` inside the
    container's persistent /workspace; the stub carries the path, row count, column
    list and a small head, plus an instruction to process the file with ``execute``.
    Returns ``(stub, note)`` on success, or ``(None, None)`` if anything went wrong —
    the caller then falls back to ``cap_result`` so a flaky sandbox never loses data
    silently or breaks the run.

    BLOCKING on purpose: ``sink.upload_files`` is a sync HTTP call and ``json.dumps``
    of a multi-MB result is CPU-bound. The caller runs this whole function through
    ``asyncio.to_thread`` — call it from async code ONLY that way, or one big offload
    stalls every concurrent tool call in the run.
    """
    try:
        # A string result is written through verbatim; anything else is serialized. Rows
        # (for the stub's count/columns/head) are the result itself when it is already a
        # list, or the parse of a string that happens to hold a JSON array.
        payload = result if isinstance(result, str) else json.dumps(result, default=str)
        rows: list | None = result if isinstance(result, list) else None
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except (ValueError, TypeError):
                parsed = None
            rows = parsed if isinstance(parsed, list) else None

        path = f"{offload_dir.rstrip('/')}/{tool_name}-{call_id}.json"
        resp = sink.upload_files([(path, payload.encode("utf-8"))])
        if resp and getattr(resp[0], "error", None):
            log.warning("offload upload failed (%s): %s", tool_name, resp[0].error)
            return None, None
    except Exception as exc:  # never lose data silently / break the run on offload failure
        log.warning("offload failed (%s): %s", tool_name, exc)
        return None, None

    if series is None:
        series = find_series(result) if len(payload) <= MAX_SCAN_BYTES else {}
    if series:
        counts = ", ".join(f"{label or 'series'}: {len(pts)} points" for label, pts in series.items())
        note = f"time series offloaded to {path} ({counts}, {len(payload)} bytes)"
        stub = (
            f"[Time series saved to a file — NOT shown inline. {SERIES_RULE}]\n"
            f"file: {path}\n"
            f"format: JSON exactly as the tool returned it ({len(payload)} bytes)\n"
            f"series: {counts}\n"
            "summary:\n" + summary_block(series) + "\n"
            "\nNumeric work over the points (percentile or z-score of a window, correlation with "
            "another series, sums, custom windows): `json.load(open(path))` in `execute` and "
            "compute; the points are where the tool put them (e.g. `data.<slug>`, a list of "
            "{datetime, value}). Report the computed numbers only — never print the points. "
            "Do NOT re-call this tool to page the same rows."
        )
        return stub, note

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
        + "\nThis file holds the COMPLETE result. Numeric aggregates / joins / filters: "
        "compute with the `execute` tool (Python + pandas/numpy over the JSON). READING "
        "the text inside (topics, sentiment, claims): you MUST delegate that to "
        "`extract-subagent` via the `task` tool (file path + question + source label) — "
        "do NOT print the text into your own context; `execute` on this file is for "
        "numeric computation and structure checks only. Do NOT re-call this tool to "
        "page the same rows."
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

        def result_event(ok: bool, **extra: Any) -> None:
            """The three result emits below differ only in their extras — the correlation
            keys are written once here so a *_result can't drift from its *_call."""
            emit({"type": f"{kind}_result", "id": call_id, "tool": tool.name,
                  "ok": ok, **extra})

        args_key = json.dumps(kwargs, sort_keys=True, default=str)
        if args_key in failed_permanently:
            result_event(False, error_class="permanent", repeated=True,
                         summary=_summarize(failed_permanently[args_key]))
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
                result_event(False, error_class=classification, summary=_summarize(msg))
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
            # A time series leaves the context whatever its size (shown the rows, the model
            # transcribes them into a table). A too-big result is offloaded regardless and
            # `_offload_result` detects the series itself.
            series: dict = {}
            if not too_big:
                series = await asyncio.to_thread(find_series, result)
            # Prefer OFFLOAD to a sandbox file over truncation: keeps the full data
            # available (the model reads it back with `execute`) instead of dropping rows.
            if (too_big or series) and offload_sink is not None:
                # to_thread: the offload serializes megabytes and uploads them over a
                # SYNC HTTP client. Called inline it would freeze the event loop — and
                # with it every other tool call in flight — for the whole upload.
                stub, note = await asyncio.to_thread(
                    _offload_result,
                    result, sink=offload_sink, offload_dir=offload_dir,
                    tool_name=tool.name, call_id=call_id, series=series or None)
                if stub is not None:
                    result, capped = stub, note
                    log.info("RESULT OFFLOADED (%s): %s [raw: %d bytes, rows=%s]",
                             tool.name, note, raw_bytes, raw_rows)
            elif series and isinstance(result, str):
                # No sandbox: the rows stay, but the summary and the rule lead.
                result = (f"[Time series. {SERIES_RULE}]\nsummary:\n{summary_block(series)}"
                          f"\n\n{result}")
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
            result_event(True, summary=_summarize(result), bytes=raw_bytes,
                         rows=raw_rows, capped=bool(capped))
            return result

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_run,
    )


def result_handling(cfg: Any, meter: Any = None, offload_sink: Any = None) -> dict[str, Any]:
    """The ``instrument_tool`` kwargs every instrumented tool shares — size bounds,
    the run meter, and where oversized results are offloaded.

    Both call sites (MCP tools in ``tools/mcp.py``, custom tools in ``agent.py``) wrap
    with IDENTICAL result handling and differ only in the transport knobs (``kind``,
    ``semaphore``, ``rate_limit_max_wait``). Building the shared half here means a new
    size knob is wired ONCE instead of having to be remembered in both places."""
    return {
        "max_result_chars": cfg.max_result_chars,
        "max_result_rows": cfg.max_result_rows,
        "meter": meter,
        "offload_sink": offload_sink,
        "offload_dir": cfg.offload_dir,
    }


def source_events(results: Iterable[dict[str, Any]]) -> None:
    """Emit one ``source`` event per result for the live citation list."""
    for r in results:
        emit({"type": "source", "title": r.get("title", ""),
              "url": r.get("url", ""), "domain": domain_of(r.get("url", ""))})
