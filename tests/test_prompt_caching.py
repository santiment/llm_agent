"""Anthropic prompt caching — cache_control breakpoint injection.

Anthropic models reached via OpenRouter cache ONLY when the request carries explicit
`cache_control` breakpoints; the other providers cache automatically and ignore the field.
These tests pin: (1) the breakpoint placement helper, (2) that `build_chat_model` attaches the
caching client ONLY for Anthropic ids and only when enabled, and (3) that the override actually
injects into the request payload — all offline, no network, no API keys.

Runs with plain Python (``python tests/test_prompt_caching.py``) and is pytest-discoverable.
"""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from deep_research_agent.config import ResearchConfig
from deep_research_agent.models import (
    _CachingChatOpenAI,
    _mark_cache,
    _needs_cache_breakpoints,
    add_cache_control,
    build_chat_model,
)

_CC = {"type": "ephemeral"}


def _cfg(**configurable) -> ResearchConfig:
    saved = os.environ.pop("DRA_PROMPT_CACHING", None)
    try:
        return ResearchConfig.from_runnable_config({"configurable": configurable})
    finally:
        if saved is not None:
            os.environ["DRA_PROMPT_CACHING"] = saved


def _has_cc(content) -> bool:
    """True if a chat-completions message `content` carries a cache_control breakpoint."""
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("cache_control") == _CC for p in content)


def test_mark_cache_string_becomes_text_block() -> None:
    msg = {"role": "system", "content": "You are a research orchestrator."}
    assert _mark_cache(msg) is True
    assert msg["content"] == [
        {"type": "text", "text": "You are a research orchestrator.", "cache_control": _CC}]


def test_mark_cache_skips_contentless_and_empty() -> None:
    # An assistant turn that is only tool_calls (no text) has nothing to anchor to.
    assert _mark_cache({"role": "assistant", "content": "", "tool_calls": [{}]}) is False
    assert _mark_cache({"role": "assistant", "tool_calls": [{}]}) is False


def test_mark_cache_tags_last_text_block_of_a_list() -> None:
    msg = {"role": "user", "content": [
        {"type": "text", "text": "first"}, {"type": "text", "text": "second"}]}
    assert _mark_cache(msg) is True
    assert msg["content"][0].get("cache_control") is None     # only the LAST text block
    assert msg["content"][1]["cache_control"] == _CC


def test_add_cache_control_marks_system_and_rolling_breakpoint() -> None:
    messages = [
        {"role": "system", "content": "BIG STATIC PROMPT"},
        {"role": "user", "content": "Analyze BTC"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},  # skipped
        {"role": "tool", "content": "tool result rows", "tool_call_id": "c1"},
    ]
    add_cache_control(messages)
    assert _has_cc(messages[0]["content"]), "system prefix must be a cache breakpoint"
    # Rolling breakpoint lands on the LAST text-bearing message (the tool result), not the
    # contentless assistant tool-call turn just before it.
    assert _has_cc(messages[3]["content"]), "rolling breakpoint must be the newest text message"
    assert not _has_cc(messages[1]["content"]), "only two breakpoints — not the middle user msg"


def test_add_cache_control_single_message_is_safe() -> None:
    messages = [{"role": "system", "content": "only a system message"}]
    add_cache_control(messages)        # must not raise; system gets the breakpoint
    assert _has_cc(messages[0]["content"])


def test_needs_cache_breakpoints_is_config_driven() -> None:
    markers = ["claude", "anthropic"]  # the default markers
    assert _needs_cache_breakpoints("anthropic/claude-opus-4.8", markers)
    assert _needs_cache_breakpoints("claude-sonnet-4-6", markers)
    assert not _needs_cache_breakpoints("deepseek/deepseek-v4-flash", markers)
    assert not _needs_cache_breakpoints("google/gemini-3.5-flash", markers)
    # Data-driven, no vendor hardcoded: opt any family in, or none.
    assert _needs_cache_breakpoints("some/new-model", ["new-model"])
    assert not _needs_cache_breakpoints("anthropic/claude-opus-4.8", [])


def test_build_chat_model_uses_caching_client_for_marked_models() -> None:
    cfg = _cfg(model_tier="high")  # Claude orchestrator + workers; default markers match
    assert isinstance(build_chat_model("anthropic/claude-opus-4.8", cfg), _CachingChatOpenAI)
    plain = build_chat_model("deepseek/deepseek-v4-flash", cfg)
    assert isinstance(plain, ChatOpenAI) and not isinstance(plain, _CachingChatOpenAI)


def test_cache_breakpoint_models_is_configurable() -> None:
    # Provider-agnostic: point the markers at a different family and IT gets breakpoints,
    # while the previously-default family no longer does. Nothing vendor-specific in code.
    cfg = _cfg(model_tier="high", cache_breakpoint_models=["gemini"])
    assert isinstance(build_chat_model("google/gemini-3.5-flash", cfg), _CachingChatOpenAI)
    assert not isinstance(build_chat_model("anthropic/claude-opus-4.8", cfg), _CachingChatOpenAI)


def test_prompt_caching_flag_disables_injection() -> None:
    cfg = _cfg(model_tier="high", prompt_caching=False)
    assert cfg.prompt_caching is False
    m = build_chat_model("anthropic/claude-opus-4.8", cfg)
    assert not isinstance(m, _CachingChatOpenAI)  # falls back to the plain client


def test_caching_client_injects_into_request_payload() -> None:
    # The real end-to-end check: the override actually rewrites the outgoing payload. No network
    # — _get_request_payload only builds the dict, it does not call the API.
    cfg = _cfg(model_tier="high")
    model = build_chat_model("anthropic/claude-opus-4.8", cfg)
    payload = model._get_request_payload([
        SystemMessage(content="BIG STATIC PROMPT with all the rules"),
        HumanMessage(content="Analyze BTC over 30 days"),
        AIMessage(content="", tool_calls=[
            {"name": "web_search", "args": {"query": "btc"}, "id": "c1"}]),
        ToolMessage(content="[1] result ...", tool_call_id="c1"),
    ])
    msgs = payload["messages"]
    assert _has_cc(msgs[0]["content"]), "system message must carry a breakpoint"
    assert _has_cc(msgs[-1]["content"]), "rolling breakpoint must reach the newest text message"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("OK — prompt caching breakpoint injection verified.")
