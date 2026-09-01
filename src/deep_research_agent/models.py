"""Model construction — deliberately provider-agnostic.

Every model goes through an OpenAI-compatible endpoint (``base_url``). With the
default OpenRouter base URL you can name ANY model — ``openai/gpt-4o``,
``anthropic/claude-sonnet-5``, ``xiaomi/mimo-v2.5``, a local vLLM slug —
without locking to one vendor's SDK. Point ``base_url`` at your own gateway and
nothing else changes.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from .config import ResearchConfig

log = logging.getLogger("deep_research_agent.models")


def build_chat_model(model_id: str, cfg: ResearchConfig) -> ChatOpenAI:
    # Some OpenRouter models (e.g. deepseek-v4-flash) emit off-spec streaming chunks that
    # LangChain merges into DOUBLED metadata (finish_reason "stopstop", doubled model_name)
    # and DROP tool_calls — which stalls the ReAct loop. Force streaming off for those.
    streaming = cfg.streaming
    if streaming and any(bad in model_id.lower() for bad in cfg.streaming_denylist):
        log.warning("Streaming force-disabled for %r — off-spec streaming corrupts tool_calls "
                    "(merged/doubled chunks); override via DRA_STREAMING_DENYLIST", model_id)
        streaming = False
    # OpenRouter's unified `reasoning` param — an OpenRouter extension, hence
    # extra_body. Ignored by models without reasoning support.
    extra_body: dict = {}
    if cfg.reasoning_effort == "none":
        extra_body["reasoning"] = {"enabled": False}
    elif cfg.reasoning_effort:
        extra_body["reasoning"] = {"effort": cfg.reasoning_effort}
    # Ask OpenRouter to put the ACTUAL charged cost into the response's usage object
    # (surfaces as response_metadata["token_usage"]["cost"]; metering.sum_usage reads it).
    # Non-streamed calls only: streamed usage arrives as a trailing usage-only chunk —
    # the exact off-spec-merge path the stream_usage note below avoids.
    if not streaming and cfg.is_openrouter:
        extra_body["usage"] = {"include": True}
    return ChatOpenAI(
        model=model_id,
        api_key=cfg.openai_api_key or "missing-key",
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        # Always set both explicitly. A proxied provider can stall a single request far
        # past any sane bound; without a timeout that one call pins its research unit —
        # and its concurrency slot — for the rest of the run. max_retries covers the
        # transient 429/5xx the same stack produces. DRA_REQUEST_TIMEOUT / DRA_MAX_RETRIES.
        timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
        extra_body=extra_body or None,
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
