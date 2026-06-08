"""Model construction — deliberately provider-agnostic.

Every model goes through an OpenAI-compatible endpoint (``base_url``). With the
default OpenRouter base URL you can name ANY model — ``openai/gpt-4o``,
``anthropic/claude-sonnet-4-6``, ``google/gemini-2.5-pro``, a local vLLM slug —
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
    return ChatOpenAI(
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
