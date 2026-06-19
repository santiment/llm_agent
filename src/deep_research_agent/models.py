"""Model construction — deliberately provider-agnostic.

Every model goes through an OpenAI-compatible endpoint (``base_url``). With the
default OpenRouter base URL you can name ANY model — ``openai/gpt-4o``,
``anthropic/claude-sonnet-4-6``, ``google/gemini-2.5-pro``, a local vLLM slug —
without locking to one vendor's SDK. Point ``base_url`` at your own gateway and
nothing else changes.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_openai import ChatOpenAI

from .config import ResearchConfig

log = logging.getLogger("deep_research_agent.models")

# The cache_control breakpoint marker (OpenRouter's standard `{"type": "ephemeral"}`). OpenRouter
# forwards it to the provider; providers that cache automatically ignore it — so we attach it only
# for models that cache ONLY with explicit breakpoints (build_chat_model gates via
# cfg.cache_breakpoint_models — data, not a hardcoded vendor).
_CACHE_CONTROL = {"type": "ephemeral"}


def _mark_cache(msg: dict[str, Any]) -> bool:
    """Attach a `cache_control` breakpoint to one chat-completions message dict; return whether
    one was placed. A plain string `content` becomes a single text block (the shape the marker
    rides on); a block-list content gets the marker on its last text block. A contentless
    message — e.g. an assistant turn that is only `tool_calls` — has nothing to anchor to, so it
    is skipped."""
    content = msg.get("content")
    if isinstance(content, str):
        if not content.strip():
            return False
        msg["content"] = [{"type": "text", "text": content, "cache_control": dict(_CACHE_CONTROL)}]
        return True
    if isinstance(content, list):
        for part in reversed(content):
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                part["cache_control"] = dict(_CACHE_CONTROL)
                return True
    return False


def add_cache_control(messages: list[dict[str, Any]]) -> None:
    """Place up to two `cache_control` breakpoints on an OpenAI-format message list, in place:

      1. the SYSTEM message — caches the big static prefix. These providers order tool
         definitions BEFORE the system prompt, so a breakpoint here covers the tool schemas too.
         This prefix is byte-identical on every model call of a run → a cache READ after the
         first write.
      2. a ROLLING breakpoint on the LAST message carrying text — so the conversation prefix that
         grew on the previous step is a cache READ on the next, both around the orchestrator's
         tool loop and across turns on the same thread (where the whole transcript is re-sent).

    These providers cache the LONGEST matching prefix and ignore a breakpoint under the minimum
    cacheable size, so a tiny early rolling breakpoint simply does nothing — never an error.
    Two breakpoints stays well within the usual limit of four."""
    if not messages:
        return
    for m in messages:  # breakpoint 1: the static system prefix (+ tool defs)
        if isinstance(m, dict) and m.get("role") == "system":
            _mark_cache(m)
            break
    for m in reversed(messages):  # breakpoint 2: rolling, on the newest text-bearing message
        if not isinstance(m, dict) or m.get("role") == "system":
            continue
        if _mark_cache(m):
            break


def _needs_cache_breakpoints(model_id: str, markers: list[str]) -> bool:
    """Whether a model's provider caches the re-sent prefix ONLY with explicit `cache_control`
    breakpoints (vs. automatically). Matched by id substring against the configured markers, so
    this stays data-driven and provider-agnostic — no vendor is hardcoded in code."""
    mid = (model_id or "").lower()
    return any(m in mid for m in markers)


class _CachingChatOpenAI(ChatOpenAI):
    """ChatOpenAI that injects `cache_control` breakpoints into every request.

    Some providers cache the re-sent prefix ONLY when the request carries explicit breakpoints
    (others cache automatically). We override the one chokepoint that sync/async/streaming all
    funnel through (`_get_request_payload`), so caching rides along on the orchestrator's
    tool-loop calls too. If a future langchain version renames that method, this override
    silently stops firing — caching is lost but nothing breaks."""

    def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        msgs = payload.get("messages")  # absent on the Responses-API path; OpenRouter uses chat
        if isinstance(msgs, list):
            add_cache_control(msgs)
        return payload


def build_chat_model(model_id: str, cfg: ResearchConfig) -> ChatOpenAI:
    # Some OpenRouter models (e.g. deepseek-v4-flash) emit off-spec streaming chunks that
    # LangChain merges into DOUBLED metadata (finish_reason "stopstop", doubled model_name)
    # and DROP tool_calls — which stalls the ReAct loop. Force streaming off for those.
    streaming = cfg.streaming
    if streaming and any(bad in model_id.lower() for bad in cfg.streaming_denylist):
        log.warning("Streaming force-disabled for %r — off-spec streaming corrupts tool_calls "
                    "(merged/doubled chunks); override via DRA_STREAMING_DENYLIST", model_id)
        streaming = False
    # Prompt caching: some providers cache the re-sent prefix automatically; others do so ONLY
    # when the request carries explicit cache_control breakpoints. For the latter (configured by
    # id substring, never a hardcoded vendor) use the injecting subclass; everyone else gets the
    # plain client unchanged.
    caching = cfg.prompt_caching and _needs_cache_breakpoints(model_id, cfg.cache_breakpoint_models)
    model_cls = _CachingChatOpenAI if caching else ChatOpenAI
    if caching:
        log.info("Prompt caching ON for %r (injecting explicit cache_control breakpoints)",
                 model_id)
    return model_cls(
        model=model_id,
        api_key=cfg.openai_api_key or "missing-key",
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        # Streaming on by default (drives the live "thinking" narration in the UI). Some
        # OpenRouter-proxied models emit off-spec streaming chunks that LangChain merges
        # into doubled metadata (e.g. deepseek-v4-flash's `finish_reason: "stopstop"`) and
        # can drop content; set DRA_STREAMING=false to fetch full responses in one shot.
        streaming=streaming,
        # Do NOT set stream_usage=True here. On this OpenRouter stack it appends a trailing
        # usage-only chunk that some upstream providers emit off-spec; LangChain can
        # mis-merge it and DROP the message's tool_calls, making the agent stop mid-research
        # with an intent-only message (the same class of bug DRA_STREAMING guards against).
        # BudgetMiddleware reads usage_metadata when the provider supplies it anyway, and
        # otherwise estimates tokens from message text — so the ceiling still bites.
    )
