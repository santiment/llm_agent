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


def provider_preferences(cfg: ResearchConfig) -> dict:
    """OpenRouter's `provider` request object; empty when nothing is set. Thresholds go in
    p50 form — the figure its endpoint feed reports as ``throughput_last_30m`` etc."""
    out: dict = {}
    if cfg.provider_min_throughput > 0:
        out["preferred_min_throughput"] = {"p50": cfg.provider_min_throughput}
    if cfg.provider_max_latency > 0:
        out["preferred_max_latency"] = {"p50": cfg.provider_max_latency}
    if cfg.provider_sort:
        out["sort"] = cfg.provider_sort
    return out


def build_chat_model(model_id: str, cfg: ResearchConfig,
                     provider: dict | None = None) -> ChatOpenAI:
    # Some OpenRouter models (e.g. deepseek-v4-flash) emit off-spec streaming chunks that
    # LangChain merges into DOUBLED metadata (finish_reason "stopstop", doubled model_name)
    # and DROP tool_calls — which stalls the ReAct loop. Force streaming off for those.
    streaming = cfg.streaming
    if streaming and any(bad in model_id.lower() for bad in cfg.streaming_denylist):
        log.warning("Streaming force-disabled for %r — off-spec streaming corrupts tool_calls "
                    "(merged/doubled chunks); override via DRA_STREAMING_DENYLIST", model_id)
        streaming = False
    # No cache pricing → every re-sent prefix is billed in full. Tiers cannot name such a
    # model (tests), so this fires only for a stale deployment or an unlisted slug.
    if cfg.is_openrouter and not cfg.caches_prompts(model_id):
        log.warning("%r has no prompt-cache pricing — re-sent context is billed in full; "
                    "see config.MODEL_CACHING", model_id)

    # OpenRouter's unified `reasoning` param (extra_body), only for models flagged capable:
    # a model that rejects it 400s on every provider with no retry and the run dies, while
    # omitting it merely leaves the provider default.
    extra_body: dict = {}
    if cfg.reasoning_effort and not cfg.supports_reasoning(model_id):
        log.info("reasoning (%s) not sent to %r: not in reasoning_capable — see "
                 "config.MODEL_REASONING / DRA_REASONING_CAPABLE", cfg.reasoning_effort, model_id)
    elif cfg.reasoning_effort == "none":
        extra_body["reasoning"] = {"enabled": False}
    elif cfg.reasoning_effort:
        extra_body["reasoning"] = {"effort": cfg.reasoning_effort}
    # OpenRouter puts the charged cost into usage (response_metadata["token_usage"]["cost"],
    # read by metering.sum_usage). Non-streamed calls only — see the stream_usage note below.
    if not streaming and cfg.is_openrouter:
        extra_body["usage"] = {"include": True}
    # ChatOpenAI sends the output cap as `max_completion_tokens` (OpenAI's current name);
    # OpenRouter documents `max_tokens`, so on that stack send it under both names — same
    # value, whichever the gateway reads wins.
    if cfg.max_output_tokens and cfg.is_openrouter:
        extra_body["max_tokens"] = cfg.max_output_tokens
    # Provider routing: without it the price-weighted default lands the fleet on a model's
    # slowest providers. `provider` is the per-model object from provider_routing.resolve
    # (price cap, ignore list); None = the static soft preferences only.
    if cfg.is_openrouter:
        if provider is None:
            provider = provider_preferences(cfg)
        if provider:
            extra_body["provider"] = provider
    return ChatOpenAI(
        model=model_id,
        api_key=cfg.openai_api_key or "missing-key",
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        # Per-call output cap (DRA_MAX_OUTPUT_TOKENS): the bound on a runaway response —
        # see ResearchConfig.max_output_tokens. None = no cap.
        max_tokens=cfg.max_output_tokens or None,
        # Always set both explicitly. A proxied provider can stall a single request far
        # past any sane bound; without a timeout that one call pins its research unit —
        # and its concurrency slot — for the rest of the run. max_retries covers the
        # sub-second 429/5xx blips the same stack produces (0.5 s doubling to 8 s); a
        # throttle that outlasts them is waited out by ModelBackoffMiddleware within
        # cfg.model_rate_limit_max_wait. DRA_REQUEST_TIMEOUT / DRA_MAX_RETRIES.
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
