"""A model we name can be slow or pricey for a while — never unreachable because of a
routing preference of ours.

Regression for the run killed by ``OpenAIModelNotFoundError: Error code: 404 - … 'No
endpoints found that satisfy the max price for this request' … 'routing_funnel': [{'step':
'Initial Endpoints', 'endpoint_count': 6}, {'step': 'Filter by Tier Endpoint Rows',
'endpoint_count': 2}], 'failed_routing_step': 'Filter by Max Price'``: the price cap
(``provider_routing.py``, 1.25x the public feed's cheapest healthy endpoint) was anchored on
an endpoint the account's own tier filter had already removed, so the two endpoints left
were both over the cap and OpenRouter refused the call outright. Pins (``model_errors.py``
+ ``provider_routing.py``):

  - routing_rejection() recognises that 404 (body or message form) and names the step;
    every other error is None;
  - relax() / next_relax_level(): the ladder drops the cap, then the provider lists, then
    the whole object, skipping levels that change nothing;
  - ProviderRoutingFallbackMiddleware retries at once with the relaxed object in a per-call
    ``extra_body`` override (the rest of the body kept, the model untouched), emits a
    ``provider_fallback`` status, remembers the level per model so the next role — and
    resolve() at the next graph build — start there, and lets the error stand only when
    nothing is left to relax; other errors propagate untouched;
  - wired on every role, right outside the backoff.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.exceptions import ModelNotFoundError, ModelRateLimitError
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import OpenAIModelNotFoundError
from langgraph.errors import GraphBubbleUp

import deep_research_agent.provider_routing as pr
from conftest import build_config, capture_events_cm
from deep_research_agent.model_errors import (ModelBackoffMiddleware,
                                              ProviderRoutingFallbackMiddleware,
                                              routing_rejection)

SLUG = "deepseek/deepseek-v4-flash-0731"
FULL = {"preferred_min_throughput": {"p50": 50.0},
        "max_price": {"prompt": 0.0625, "completion": 0.2},
        "ignore": ["fireworks", "mancer", "venice"]}
# Everything else build_chat_model puts in the body must survive a relaxed retry.
EXTRAS = {"usage": {"include": True}, "max_tokens": 4096}


@pytest.fixture(autouse=True)
def _fresh_memo():
    pr.clear_cache()
    yield
    pr.clear_cache()


def _routing_404(step: str | None = "Filter by Max Price", *, body: bool = True):
    """OpenRouter's refusal exactly as the OpenAI SDK raises it (message carries the body)."""
    payload = {"error": {"message": "No endpoints found that satisfy the max price for this request",
                         "code": 404, "metadata": {
                             "routing_funnel": [{"step": "Initial Endpoints", "endpoint_count": 6},
                                                {"step": "Filter by Tier Endpoint Rows", "endpoint_count": 2}],
                             "failed_routing_step": step}}}
    if step is None:
        del payload["error"]["metadata"]
    resp = httpx.Response(404, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))
    return OpenAIModelNotFoundError(f"Error code: 404 - {payload}", response=resp,
                                    body=payload if body else None)


def _request(provider: dict | None = FULL) -> ModelRequest:
    body = dict(EXTRAS)
    if provider:
        body["provider"] = provider
    model = ChatOpenAI(model=SLUG, api_key="k", base_url="https://openrouter.ai/api/v1",
                       extra_body=body)
    return ModelRequest(model=model, messages=[], model_settings={})


def _sent(request: ModelRequest) -> dict:
    """The body the call would send: the per-call override, else the model's own."""
    settings = request.model_settings or {}
    return dict(settings["extra_body"] if "extra_body" in settings else request.model.extra_body or {})


def _run(mw: ProviderRoutingFallbackMiddleware, request: ModelRequest, handler):
    return asyncio.run(mw.awrap_model_call(request, handler))


def _refuse_while(pred):
    """A handler that 404s while ``pred(provider)`` holds, else answers; records each body."""
    sent: list[dict] = []

    async def handler(request):
        body = _sent(request)
        sent.append(body)
        if pred(body.get("provider")):
            raise _routing_404()
        return "ok"

    return handler, sent


# --- classification ---------------------------------------------------------------------

def test_routing_rejection_names_the_step_from_body_or_message() -> None:
    assert routing_rejection(_routing_404()) == "Filter by Max Price"
    assert routing_rejection(_routing_404(body=False)) == "Filter by Max Price"  # message only
    assert routing_rejection(_routing_404(step=None)) == "unknown"
    # A core-classified not-found without an HTTP status but with the marker still counts.
    assert routing_rejection(ModelNotFoundError("No endpoints found for x/y")) == "unknown"


def test_other_errors_are_not_routing_rejections() -> None:
    resp = httpx.Response(404, request=httpx.Request("POST", "https://openrouter.ai/x"))
    plain = OpenAIModelNotFoundError("Error code: 404 - {'error': {'message': 'Model not found'}}",
                                     response=resp, body=None)
    assert routing_rejection(plain) is None            # a wrong slug: routing less won't help
    assert routing_rejection(ModelRateLimitError("429 no endpoints found")) is None  # not a 404
    assert routing_rejection(ValueError("No endpoints found")) is None


# --- the ladder ----------------------------------------------------------------------------

def test_relax_ladder_drops_cap_then_lists_then_everything() -> None:
    assert pr.relax(FULL, 0) == FULL and pr.relax(FULL, 0) is not FULL
    assert pr.relax(FULL, 1) == {"preferred_min_throughput": {"p50": 50.0},
                                 "ignore": ["fireworks", "mancer", "venice"]}
    assert pr.relax(FULL, 2) == {"preferred_min_throughput": {"p50": 50.0}}
    assert pr.relax(FULL, pr.MAX_RELAX_LEVEL) == {} and pr.MAX_RELAX_LEVEL == 3
    assert pr.relax({"order": ["a"], "only": ["a"], "sort": "price"}, 2) == {"sort": "price"}
    assert pr.relax(None, 1) == {}


def test_next_relax_level_skips_levels_that_change_nothing() -> None:
    assert pr.next_relax_level(FULL, 0) == 1
    assert pr.next_relax_level(FULL, 1) == 2
    assert pr.next_relax_level(FULL, 2) == 3
    assert pr.next_relax_level(FULL, 3) is None
    soft_only = {"preferred_min_throughput": {"p50": 50.0}}
    assert pr.next_relax_level(soft_only, 0) == 3     # no cap, no lists: straight to bare
    assert pr.next_relax_level({"max_price": {"prompt": 1}}, 0) == 1
    assert pr.next_relax_level({"max_price": {"prompt": 1}}, 1) is None  # level 1 already bare
    assert pr.next_relax_level({}, 0) is None


def test_memo_is_per_slug_bounded_by_ttl_and_only_grows() -> None:
    pr.remember_relaxed(SLUG, 1, ttl=60)
    assert pr.relaxed_level(SLUG) == 1 and pr.relaxed_level("other/model") == 0
    pr.remember_relaxed(SLUG, 2, ttl=60)
    pr.remember_relaxed(SLUG, 1, ttl=60)                 # a lower level never un-relaxes
    assert pr.relaxed_level(SLUG) == 2
    pr.remember_relaxed("x/y", 1, ttl=0)                 # TTL 0: nothing remembered
    assert pr.relaxed_level("x/y") == 0
    pr.clear_cache()
    assert pr.relaxed_level(SLUG) == 0


# --- the middleware -------------------------------------------------------------------------

def test_cap_refused_then_retried_without_it() -> None:
    mw = ProviderRoutingFallbackMiddleware("research-subagent", SLUG, ttl=300)
    handler, sent = _refuse_while(lambda p: p and "max_price" in p)
    with capture_events_cm() as events:
        assert _run(mw, _request(), handler) == "ok"
    assert len(sent) == 2
    assert sent[0]["provider"] == FULL
    assert sent[1]["provider"] == pr.relax(FULL, 1)       # cap gone, ignore list and soft pref kept
    assert {k: v for k, v in sent[1].items() if k != "provider"} == EXTRAS  # rest of the body intact
    fb = [e for e in events if e.get("state") == "provider_fallback"]
    assert len(fb) == 1
    assert (fb[0]["role"], fb[0]["model"], fb[0]["step"], fb[0]["level"], fb[0]["dropped"]) == \
        ("research-subagent", SLUG, "Filter by Max Price", 1, ["max_price"])
    assert "No endpoints found that satisfy the max price" in fb[0]["detail"]
    assert pr.relaxed_level(SLUG) == 1


def test_the_model_instance_is_never_mutated() -> None:
    request = _request()
    handler, _ = _refuse_while(lambda p: p and "max_price" in p)
    _run(ProviderRoutingFallbackMiddleware("orchestrator", SLUG), request, handler)
    assert request.model.extra_body["provider"] == FULL
    assert request.model_settings == {}                    # override() copied; ours untouched


def test_remembered_level_starts_every_role_relaxed() -> None:
    pr.remember_relaxed(SLUG, 1, ttl=300)
    handler, sent = _refuse_while(lambda p: p and "max_price" in p)
    assert _run(ProviderRoutingFallbackMiddleware("extract-subagent", SLUG), _request(), handler) == "ok"
    assert len(sent) == 1 and sent[0]["provider"] == pr.relax(FULL, 1)  # no refused call at all


def test_ladder_climbs_to_bare_then_the_error_stands() -> None:
    mw = ProviderRoutingFallbackMiddleware("coding-subagent", SLUG, ttl=300)
    handler, sent = _refuse_while(lambda p: True)          # nothing satisfies OpenRouter
    with capture_events_cm() as events, pytest.raises(OpenAIModelNotFoundError):
        _run(mw, _request(), handler)
    assert [b.get("provider") for b in sent] == [FULL, pr.relax(FULL, 1), pr.relax(FULL, 2), None]
    assert "provider" not in sent[-1] and sent[-1] == EXTRAS  # bare: OpenRouter's own routing
    assert [e["level"] for e in events if e.get("state") == "provider_fallback"] == [1, 2, 3]
    assert pr.relaxed_level(SLUG) == 3


def test_a_bare_request_has_nothing_to_relax() -> None:
    mw = ProviderRoutingFallbackMiddleware("orchestrator", SLUG)
    handler, sent = _refuse_while(lambda p: True)
    with capture_events_cm() as events, pytest.raises(OpenAIModelNotFoundError):
        _run(mw, _request(provider=None), handler)
    assert len(sent) == 1 and not events and pr.relaxed_level(SLUG) == 0


def test_other_errors_propagate_untouched() -> None:
    mw = ProviderRoutingFallbackMiddleware("orchestrator", SLUG)
    calls = []

    async def throttled(request):
        calls.append(1)
        raise ModelRateLimitError("429 temporarily rate-limited upstream")

    async def bubble(request):
        raise GraphBubbleUp()

    with pytest.raises(ModelRateLimitError):
        _run(mw, _request(), throttled)
    assert len(calls) == 1 and pr.relaxed_level(SLUG) == 0
    with pytest.raises(GraphBubbleUp):
        _run(mw, _request(), bubble)


def test_sync_path_matches_async() -> None:
    mw = ProviderRoutingFallbackMiddleware("orchestrator", SLUG)
    sent = []

    def handler(request):
        sent.append(_sent(request))
        if "max_price" in (sent[-1].get("provider") or {}):
            raise _routing_404()
        return "ok"

    assert mw.wrap_model_call(_request(), handler) == "ok"
    assert [b["provider"] for b in sent] == [FULL, pr.relax(FULL, 1)]


# --- resolve() honours the memo; wiring ----------------------------------------------------

def _ep(provider, tag, prompt, completion, uptime=99.5, status=0):
    return {"provider_name": provider, "tag": tag, "status": status, "uptime_last_30m": uptime,
            "pricing": {"prompt": str(prompt / 1e6), "completion": str(completion / 1e6)}}


_ENV_KEYS = ("DRA_PROVIDER_MIN_THROUGHPUT", "DRA_PROVIDER_MAX_LATENCY", "DRA_PROVIDER_SORT",
             "DRA_PROVIDER_MAX_PRICE_FACTOR", "DRA_PROVIDER_MIN_UPTIME",
             "DRA_PROVIDER_ROUTING_TTL", "OPENAI_BASE_URL")


def test_resolve_starts_a_relaxed_model_at_its_level(monkeypatch) -> None:
    feed = [_ep("OpenInference", "openinference/fp8", 0.05, 0.16),
            _ep("Fireworks", "fireworks", 0.22, 0.66, uptime=94.7, status=-2)]

    async def fake_fetch(slug, ttl, timeout=5.0):
        return feed

    monkeypatch.setattr(pr, "fetch_endpoints", fake_fetch)
    cfg = build_config(_ENV_KEYS, openai_api_key="k")
    full = asyncio.run(pr.resolve(cfg, [SLUG]))[SLUG]
    assert full["max_price"] == {"prompt": 0.0625, "completion": 0.2} and full["ignore"] == ["fireworks"]
    pr.remember_relaxed(SLUG, 1, ttl=300)
    assert asyncio.run(pr.resolve(cfg, [SLUG]))[SLUG] == pr.relax(full, 1)
    pr.remember_relaxed(SLUG, 3, ttl=300)
    assert asyncio.run(pr.resolve(cfg, [SLUG]))[SLUG] == {}


def test_wired_right_outside_the_backoff_on_every_role(monkeypatch) -> None:
    from conftest import make_graph_capture
    from deep_research_agent.config import ResearchConfig

    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    config = {"configurable": {"openai_api_key": "k", "mcp_servers": [],
                               "sandbox_url": "http://sandbox.invalid:8080",
                               "provider_routing_ttl": 77}}
    cfg = ResearchConfig.from_runnable_config(config)
    captured = make_graph_capture(monkeypatch, config)

    def check(mws: list, role: str, model: str) -> None:
        backoff = [m for m in mws if isinstance(m, ModelBackoffMiddleware)]
        assert len(backoff) == 1, role
        fb = mws[mws.index(backoff[0]) - 1]                 # immediately outside the backoff
        assert isinstance(fb, ProviderRoutingFallbackMiddleware), role
        assert (fb.role, fb.model, fb.ttl) == (role, model, 77.0), role
        assert sum(isinstance(m, ProviderRoutingFallbackMiddleware) for m in mws) == 1, role

    check(captured["middleware"], "orchestrator", cfg.research_model)
    specs = {s["name"]: s for s in captured["subagents"]}
    for name, model in {"research-subagent": cfg.subagent_model, "extract-subagent": cfg.utility_model,
                        "coding-subagent": cfg.coding_model}.items():
        check(specs[name]["middleware"], name, model)


# --- the two model calls outside an agent step run the same ladder ------------------------

class _Direct:
    """A chat-model stand-in with an OpenRouter body: 404s while the provider it is asked to
    send still carries `max_price`; records the body of every call."""

    def __init__(self, extra_body=None, model_name=SLUG):
        self.extra_body = extra_body
        self.model_name = model_name
        self.sent: list[dict | None] = []

    def _answer(self, kwargs):
        body = kwargs.get("extra_body", self.extra_body)
        self.sent.append(body)
        if body and "max_price" in (body.get("provider") or {}):
            raise _routing_404()
        return "ok"

    def invoke(self, input, **kwargs):
        return self._answer(kwargs)

    async def ainvoke(self, input, **kwargs):
        return self._answer(kwargs)

    def with_config(self, **_):
        class _Bound:
            bound = self
            def invoke(s, input, **kw): return self.invoke(input, **kw)
            async def ainvoke(s, input, **kw): return await self.ainvoke(input, **kw)
        return _Bound()


def test_direct_calls_relax_the_same_way_and_share_the_memo() -> None:
    from deep_research_agent.model_errors import (ainvoke_with_routing_fallback,
                                                  invoke_with_routing_fallback, model_slug,
                                                  routing_body)

    m = _Direct({**EXTRAS, "provider": FULL})
    with capture_events_cm() as events:
        assert invoke_with_routing_fallback(m, [], role="triage") == "ok"
    assert [b["provider"] for b in m.sent] == [FULL, pr.relax(FULL, 1)]
    assert {k: v for k, v in m.sent[1].items() if k != "provider"} == EXTRAS
    assert [e["state"] for e in events] == ["provider_fallback"] and events[0]["role"] == "triage"
    assert pr.relaxed_level(SLUG) == 1
    # The memo is per model: the next direct call (async, through a binding) starts relaxed.
    bound = _Direct({**EXTRAS, "provider": FULL}).with_config(tags=["nostream"])
    assert routing_body(bound)["provider"] == FULL and model_slug(bound) == SLUG
    assert asyncio.run(ainvoke_with_routing_fallback(bound, [], role="triage")) == "ok"
    assert bound.bound.sent == [{**EXTRAS, "provider": pr.relax(FULL, 1)}]


def test_direct_calls_without_a_routing_object_are_plain_calls() -> None:
    from deep_research_agent.model_errors import invoke_with_routing_fallback

    class _Fake:                      # a test double with no extra_body at all
        calls = 0
        def invoke(self, input):      # no **kwargs: must be called with none
            self.calls += 1
            raise _routing_404()

    f = _Fake()
    with pytest.raises(OpenAIModelNotFoundError):
        invoke_with_routing_fallback(f, [], role="compaction", slug="x/y")
    assert f.calls == 1 and pr.relaxed_level("x/y") == 0


def test_triage_router_survives_a_refused_routing_object() -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from deep_research_agent.triage import TriageRouterMiddleware

    class _Router(_Direct):
        replies = [AIMessage("SIMPLE"), AIMessage("Sofia.", id="a1")]
        def _answer(self, kwargs):
            super()._answer(kwargs)
            return self.replies.pop(0)

    model = _Router({"provider": FULL})
    with capture_events_cm() as events:
        out = TriageRouterMiddleware(model).before_model(
            {"messages": [HumanMessage("what's the capital of bulgaria")]}, None)
    assert out is not None and out["jump_to"] == "end"          # routed, not fallen through
    # router: refused once, relaxed; answerer: starts relaxed from the memo — three calls total
    assert [b["provider"] for b in model.sent] == [FULL, pr.relax(FULL, 1), pr.relax(FULL, 1)]
    assert [e["role"] for e in events if e.get("state") == "provider_fallback"] == ["triage"]


def test_compaction_summarizer_survives_a_refused_routing_object() -> None:
    from langchain_core.messages import AIMessage
    from deep_research_agent.compaction import ContextCompactionMiddleware
    from test_compaction import _turn

    class _Summarizer(_Direct):
        def _answer(self, kwargs):
            super()._answer(kwargs)
            return AIMessage("HANDOFF SUMMARY")

    model = _Summarizer({"provider": FULL})
    mw = ContextCompactionMiddleware(model, trigger_tokens=5_000, keep_recent=4)
    update = mw.before_model({"messages": _turn(10)}, None)
    assert update is not None and "messages" in update            # compacted, not skipped
    assert [b["provider"] for b in model.sent] == [FULL, pr.relax(FULL, 1)]
