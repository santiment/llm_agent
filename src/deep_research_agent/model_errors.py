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

``ProviderRoutingFallbackMiddleware`` answers a third, deterministic one: OpenRouter
refusing the call over OUR routing object — ``Error code: 404 … No endpoints found that
satisfy the max price`` (``provider_routing.py``'s cap, anchored on the public feed's
cheapest endpoint, which the account's own tier/data-policy filters had already removed).
No retry as-is can help, so the middleware retries at once with less of the object —
``provider_routing.relax``: the cap, then the ignore list, then nothing — and remembers the
level that worked per model for the routing TTL. A model we name can be slow or pricey for
a while; it can no longer be unreachable because of a preference of ours. Listed right
BEFORE (outside) the backoff, so a relaxed retry still gets its throttle handling. The two
model calls that happen outside an agent's model step — the triage router (the FIRST call of
every run, so what it learns every later role starts from) and the compaction summarizer —
go through ``invoke_with_routing_fallback`` for the same ladder.

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
import re
import time
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.exceptions import ModelError, ModelNotFoundError
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from .events import _is_rate_limited, _retry_after_seconds, emit, exception_message
from .provider_routing import (MAX_RELAX_LEVEL, next_relax_level, relax, relaxed_level,
                               remember_relaxed)
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


_NO_ENDPOINTS = "no endpoints found"
_ROUTING_STEP = re.compile(r"failed_routing_step['\"]?\s*:\s*['\"]([^'\"]+)['\"]")


def routing_rejection(exc: BaseException) -> str | None:
    """OpenRouter refused the call over the request's provider-routing object — a 404 whose
    message reads "No endpoints found …" (that satisfy the max price / the ignore list) —
    and the routing step that emptied the pool (``metadata.failed_routing_step``, e.g.
    "Filter by Max Price"; "unknown" when the body names none). None for any other error:
    a wrong slug, an auth failure or a throttle is not fixed by routing less."""
    status = getattr(exc, "status_code", None)
    if status is None and isinstance(exc, ModelNotFoundError):
        status = 404
    if status != 404:
        return None
    text = str(exc)
    if _NO_ENDPOINTS not in text.lower():
        return None
    body = getattr(exc, "body", None)
    err = body.get("error", body) if isinstance(body, dict) else {}
    meta = err.get("metadata") if isinstance(err, dict) else None
    step = str(meta.get("failed_routing_step") or "") if isinstance(meta, dict) else ""
    if not step:
        m = _ROUTING_STEP.search(text)
        step = m.group(1) if m else ""
    return step or "unknown"


def routing_body(model) -> dict[str, Any] | None:
    """The OpenAI-compatible ``extra_body`` a model sends (``provider`` rides in it), looking
    through Runnable bindings (``with_config``, ``bind``) to the chat model; None when there
    is none — a fake, or a model off OpenRouter."""
    m = model
    for _ in range(6):
        if m is None:
            break
        body = getattr(m, "extra_body", None)
        if body is not None:
            return dict(body)
        if hasattr(m, "extra_body"):
            return {}
        m = getattr(m, "bound", None)
    return None


def model_slug(model) -> str:
    """The model id behind ``model`` (through bindings), "" when unknown."""
    m = model
    for _ in range(6):
        if m is None:
            return ""
        slug = getattr(m, "model_name", None) or getattr(m, "model", None)
        if isinstance(slug, str) and slug:
            return slug
        m = getattr(m, "bound", None)
    return ""


def _relaxed_body(body: dict[str, Any] | None, original: dict[str, Any], level: int) -> dict[str, Any]:
    out = dict(body or {})
    provider = relax(original, level)
    if provider:
        out["provider"] = provider
    else:
        out.pop("provider", None)
    return out


def relax_step(role: str, slug: str, ttl: float, original: dict[str, Any], level: int,
               exc: BaseException) -> int | None:
    """The relax level to retry at after ``exc``, or None to let it propagate: not a routing
    refusal, or nothing left to give up. Logs, emits ``provider_fallback`` and remembers the
    level for ``slug``. Shared by the middleware and the direct-call helpers below."""
    step = routing_rejection(exc)
    if step is None:
        return None
    detail = error_detail(exc)
    nxt = next_relax_level(original, level)
    if nxt is None:
        log.error("MODEL ROUTING role=%s model=%s: no endpoint at %r and nothing left to "
                  "relax (level %d/%d) — %s", role, slug or "?", step, level,
                  MAX_RELAX_LEVEL, detail)
        return None
    dropped = sorted(set(relax(original, level)) - set(relax(original, nxt)))
    remember_relaxed(slug, nxt, ttl)
    log.warning("MODEL ROUTING role=%s model=%s: no endpoint passed %r — retrying without "
                "%s (relax level %d/%d, remembered %.0fs): %s", role, slug or "?",
                step, ", ".join(dropped) or "the routing object", nxt, MAX_RELAX_LEVEL,
                ttl, detail)
    emit({"type": "status", "state": "provider_fallback", "role": role, "model": slug,
          "detail": detail, "step": step, "level": nxt, "dropped": dropped})
    return nxt


def _direct_start(model, slug: str):
    """``(body, original, level, kwargs)`` for a direct call: the model's body, its routing
    object, the level it is already known to need, and the call kwargs that apply it."""
    body = routing_body(model)
    original = dict((body or {}).get("provider") or {})
    level = relaxed_level(slug) if original else 0
    kwargs = {"extra_body": _relaxed_body(body, original, level)} if level else {}
    return body, original, level, kwargs


def invoke_with_routing_fallback(model, input, *, role: str, slug: str = "", ttl: float = 300.0):
    """``model.invoke(input)`` with the same routing ladder the middleware applies — for the
    model calls that happen OUTSIDE an agent's model step (the triage router, the compaction
    summarizer): OpenRouter refusing the routing object is retried with less of it, per-call
    (``extra_body`` kwargs win over the model's field; the model is never mutated). A model
    without a routing object is called as-is."""
    slug = slug or model_slug(model)
    body, original, level, kwargs = _direct_start(model, slug)
    while True:
        try:
            return model.invoke(input, **kwargs)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            level = relax_step(role, slug, ttl, original, level, exc)
            if level is None:
                raise
            kwargs = {"extra_body": _relaxed_body(body, original, level)}


async def ainvoke_with_routing_fallback(model, input, *, role: str, slug: str = "",
                                        ttl: float = 300.0):
    """Async twin of ``invoke_with_routing_fallback``."""
    slug = slug or model_slug(model)
    body, original, level, kwargs = _direct_start(model, slug)
    while True:
        try:
            return await model.ainvoke(input, **kwargs)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            level = relax_step(role, slug, ttl, original, level, exc)
            if level is None:
                raise
            kwargs = {"extra_body": _relaxed_body(body, original, level)}


class ProviderRoutingFallbackMiddleware(AgentMiddleware):
    """A call OpenRouter refuses over OUR routing object is retried, at once, with less of it.

    The price cap and ignore list are hard. When they leave no endpoint — OpenRouter's
    account-side filters run first and the public feed knows nothing of them (a "Filter by
    Tier" step took 6 endpoints to 2, the cap took those to 0) — the call 404s instead of
    falling back. Here that 404 climbs ``provider_routing.relax``: the cap first, then the
    provider lists, then the whole object; each step is one immediate retry, since the
    refusal is deterministic. The level that worked is remembered per model for ``ttl``
    seconds (the routing TTL) so every parallel sub-agent on the model, and the next graph
    build, start there. Every other exception propagates untouched.

    The object travels inside the OpenAI-compatible ``extra_body``: a per-call override in
    ``model_settings`` (LangChain binds it as call kwargs, which win over the model's own
    field) carries the relaxed body, so the model instance itself is never mutated."""

    def __init__(self, role: str, model: str = "", *, ttl: float = 300.0) -> None:
        super().__init__()
        self.role = role
        self.model = model or ""
        self.ttl = max(0.0, float(ttl))

    def wrap_model_call(self, request, handler):
        original, level, request = self._start(request)
        while True:
            try:
                return handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                level = self._next_level(original, level, exc)
                if level is None:
                    raise
                request = self._with_level(request, original, level)

    async def awrap_model_call(self, request, handler):
        original, level, request = self._start(request)
        while True:
            try:
                return await handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                level = self._next_level(original, level, exc)
                if level is None:
                    raise
                request = self._with_level(request, original, level)

    @staticmethod
    def _body_of(request) -> dict[str, Any]:
        """The request-body extras the call will send: a per-call ``extra_body`` in
        ``model_settings`` wins over the model's own field."""
        settings = request.model_settings or {}
        if "extra_body" in settings:
            return dict(settings["extra_body"] or {})
        return routing_body(request.model) or {}

    def _start(self, request):
        """The original routing object, and the request begun at the level this model is
        already known to need (0 = as built)."""
        original = dict(self._body_of(request).get("provider") or {})
        level = relaxed_level(self.model) if original else 0
        if level:
            request = self._with_level(request, original, level)
        return original, level, request

    @classmethod
    def _with_level(cls, request, original: dict[str, Any], level: int):
        body = cls._body_of(request)
        provider = relax(original, level)
        if provider:
            body["provider"] = provider
        else:
            body.pop("provider", None)
        return request.override(model_settings={**(request.model_settings or {}),
                                                "extra_body": body})

    def _next_level(self, original: dict[str, Any], level: int, exc: BaseException) -> int | None:
        return relax_step(self.role, self.model, self.ttl, original, level, exc)


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
