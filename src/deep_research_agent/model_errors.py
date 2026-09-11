"""Provider failures on model calls: wait them out, and never let one dead sub-agent
take the run down.

Two things ended a run over a *temporary* condition (a 429 from every provider in
OpenRouter's pool for the sub-agent model — "temporarily rate-limited upstream"):

1. The OpenAI SDK's own retries (``max_retries``: 0.5 s → 8 s exponential, ~3.5 s in
   total for the default 3) were spent inside four seconds — far shorter than a
   shared-pool throttle lasts — and the exception escaped the model call.
2. It then bubbled out of the sub-agent, through the orchestrator's ``task`` tool call,
   and killed the WHOLE run: LangGraph's ToolNode re-raises anything that is not an
   argument-validation error, so one unit's model outage discarded every other unit's
   finished work and the report with it. The host showed "An internal error occurred".

``ModelBackoffMiddleware`` answers (1): on a retryable model error (429 / 5xx / timeout /
connection — ``langchain_core.exceptions.ModelError.is_retryable``, or a 429 marker in the
message) it waits — ``Retry-After`` when the provider says, else capped exponential
backoff — and calls again, until the cumulative wait would exceed ``max_wait`` seconds
(``model_rate_limit_max_wait`` / ``DRA_MODEL_RATE_LIMIT_MAX_WAIT``; 0 disables). The same
budgeted-backoff contract as the MCP tool wrapper's 429 handling (``events.instrument_tool``),
so the two throttle knobs read alike. Anything else re-raises untouched — an auth or
bad-request error gets no better by retrying. Listed LAST in every role's middleware, so a
retry re-runs only the model call, not the request rewrites above it.

``SubagentFailureMiddleware`` answers (2): an exception out of ``task`` becomes an error
``ToolMessage`` telling the caller the unit was NOT researched and how to proceed (retry
once, else report the gap) — a failed delegation is a RESULT, the same stance the MCP
wrapper takes for a failed data call. Attached to whoever holds a ``task`` tool: the
orchestrator and the research-subagent (which nests the extract / coding sub-agents).
LangGraph control flow (``GraphBubbleUp``: interrupts, parent commands) always propagates.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.exceptions import ModelError
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from .events import _is_rate_limited, _retry_after_seconds, emit, exception_message
from .turn import tool_call_of

log = logging.getLogger("deep_research_agent.model_errors")

# How much of a provider error reaches the log line, the status event and the tool result.
# OpenRouter's 429 body is a nested JSON of every provider it tried; the first few hundred
# characters carry the message and the first provider's reason.
_DETAIL_CHARS = 300


def is_transient_model_error(exc: BaseException) -> bool:
    """A model-call failure that an identical later call may survive: the integration says
    so (``ModelError.is_retryable`` — 429, 5xx, timeout, connection), or the message carries
    a rate-limit marker (a provider path the integration never classified)."""
    if isinstance(exc, ModelError):
        return bool(exc.is_retryable)
    return _is_rate_limited(str(exc).lower())


def retry_after_seconds(exc: BaseException) -> float | None:
    """The provider's ``Retry-After`` hint: the response header when the SDK kept the
    response (seconds form only), else a hint in the message text; None when neither."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    raw = headers.get("retry-after") if headers is not None else None
    if raw is not None:
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            seconds = -1.0  # the HTTP-date form: not worth parsing, the backoff covers it
        if seconds >= 0:
            return seconds
    return _retry_after_seconds(str(exc).lower())


def error_detail(exc: BaseException) -> str:
    """``Type: message`` for humans and the model, bounded."""
    return f"{type(exc).__name__}: {exception_message(exc)[:_DETAIL_CHARS]}"


class ModelBackoffMiddleware(AgentMiddleware):
    """Wait out transient provider errors on this role's model calls within a budget.

    One instance per role (it names the role and model in its log lines and ``status``
    events). Stateless across calls: attempts and waited time live in the call frame, so
    one instance serves every parallel sub-agent run of that role."""

    def __init__(self, role: str, model: str = "", *, max_wait: float = 120.0,
                 base_delay: float = 2.0, max_delay: float = 30.0, jitter: bool = True) -> None:
        super().__init__()
        self.role = role
        self.model = model or ""
        self.max_wait = max(0.0, float(max_wait))
        self.base_delay = base_delay
        self.max_delay = max_delay
        # ±20% on the computed delay so parallel sub-agents throttled together don't
        # all come back in the same instant. A provider's Retry-After is honored as-is.
        self.jitter = jitter

    # Indirection so tests can observe the waits without patching the event loop.
    _sleep = staticmethod(time.sleep)
    _asleep = staticmethod(asyncio.sleep)

    def wrap_model_call(self, request, handler):
        attempt, waited = 0, 0.0
        while True:
            try:
                return handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                delay = self._before_retry(exc, attempt, waited)
                if delay is None:
                    raise
                self._sleep(delay)
                attempt, waited = attempt + 1, waited + delay

    async def awrap_model_call(self, request, handler):
        attempt, waited = 0, 0.0
        while True:
            try:
                return await handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                delay = self._before_retry(exc, attempt, waited)
                if delay is None:
                    raise
                await self._asleep(delay)
                attempt, waited = attempt + 1, waited + delay

    def _delay(self, exc: BaseException, attempt: int) -> float:
        hinted = retry_after_seconds(exc)
        if hinted is not None:
            return hinted
        delay = min(self.max_delay, self.base_delay * (2 ** attempt))
        return delay * random.uniform(0.8, 1.2) if self.jitter else delay

    def _before_retry(self, exc: BaseException, attempt: int, waited: float) -> float | None:
        """Seconds to sleep before calling again, or None to let ``exc`` propagate: not a
        transient failure, or the budget can't cover the next wait. Logs and emits the
        ``status`` event for whichever it is — the UI's only word on why nothing moves."""
        if not is_transient_model_error(exc):
            return None  # not ours to soften; the run's own error path reports it
        detail = error_detail(exc)
        delay = self._delay(exc, attempt)
        if self.max_wait <= 0 or waited + delay > self.max_wait:
            log.error("MODEL UNAVAILABLE role=%s model=%s: %s — giving up after %d retr%s / "
                      "%.0fs of backoff (budget %.0fs; DRA_MODEL_RATE_LIMIT_MAX_WAIT)",
                      self.role, self.model or "?", detail, attempt,
                      "y" if attempt == 1 else "ies", waited, self.max_wait)
            emit({"type": "status", "state": "model_unavailable", "role": self.role,
                  "model": self.model, "detail": detail, "retries": attempt,
                  "waited_s": round(waited, 1), "budget_s": self.max_wait})
            return None
        log.warning("MODEL BACKOFF role=%s model=%s: %s — retry %d in %.1fs (waited %.0f/%.0fs)",
                    self.role, self.model or "?", detail, attempt + 1, delay, waited,
                    self.max_wait)
        emit({"type": "status", "state": "rate_limited", "role": self.role, "model": self.model,
              "detail": detail, "retry": attempt + 1, "wait_s": round(delay, 1),
              "waited_s": round(waited, 1), "budget_s": self.max_wait})
        return delay


# The same "TOOL ERROR (…)" shape the MCP wrapper uses, so the model reads both alike.
_TASK_FAILED = (
    "TOOL ERROR (task → {role}): the sub-agent failed before returning its findings — "
    "{error}. Its unit was NOT researched and nothing from it is available. Retry that "
    "task ONCE with the same brief; if it fails again, continue with the other units and "
    "list this unit under gaps in the report instead of guessing at its data."
)


class SubagentFailureMiddleware(AgentMiddleware):
    """A sub-agent that dies mid-``task`` is reported to its caller as an error tool
    result — the caller decides (retry the unit, or deliver with a gap) — instead of the
    exception unwinding the whole run."""

    TOOL = "task"

    def wrap_tool_call(self, request, handler):
        name, args, call_id = tool_call_of(request)
        if name != self.TOOL:
            return handler(request)
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return self._failed(args, call_id, exc)

    async def awrap_tool_call(self, request, handler):
        name, args, call_id = tool_call_of(request)
        if name != self.TOOL:
            return await handler(request)
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return self._failed(args, call_id, exc)

    def _failed(self, args: dict[str, Any], call_id: str, exc: BaseException) -> ToolMessage:
        role = str(args.get("subagent_type") or "sub-agent")
        error = error_detail(exc)
        log.error("SUBAGENT FAILED role=%s: %s — returned to its caller as a tool error; "
                  "the run continues", role, error, exc_info=exc)
        emit({"type": "status", "state": "subagent_failed", "role": role, "detail": error})
        return ToolMessage(content=_TASK_FAILED.format(role=role, error=error),
                           tool_call_id=call_id, name=self.TOOL, status="error")
