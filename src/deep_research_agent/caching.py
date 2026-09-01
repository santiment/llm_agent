"""Prompt-cache breakpoints, injected per model request — crush's caching layout.

A research run makes MANY model calls whose prompts share a long stable prefix:
the system prompt plus everything before the newest messages. Without cache
markers that prefix re-bills at the full input rate on every call. OpenRouter
forwards Anthropic-style ``cache_control`` blocks to providers that support
prompt caching (Anthropic, Gemini, …) and strips them for providers that don't —
so marking the request costs nothing where unsupported and cuts the input bill
where supported.

Placement (same as crush): one breakpoint on the system prompt + one on each of
the last ``_BREAKPOINTS`` cacheable conversation messages, so the whole prefix up
to the newest exchange is a cache hit on the next call.

Two hard rules learned from the langchain-openai source:
  - ``cache_control`` survives ``_convert_message_to_dict`` only for system /
    human / assistant content blocks; on a ToolMessage it is silently DROPPED.
    Breakpoints therefore land on Human/AI messages only.
  - Requests are rewritten via ``request.override(...)`` with COPIES
    (``model_copy``) — the originals live in graph state and must never mutate.

Stateless; one instance is shared by the orchestrator and every sub-agent.
Wired only when the base_url is OpenRouter (agent.py) — a direct OpenAI-compat
endpoint may reject unknown content-block keys. DRA_PROMPT_CACHING=false is the
kill switch.
"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

_BREAKPOINTS = 2  # conversation messages to mark, beyond the system prompt


def _with_cache_control(message):
    """A COPY of ``message`` whose text content is one cache-marked block, or ``None``
    when the message can't carry a marker (non-str or empty content — reasoning models
    already return block lists; rewriting those risks scrambling provider-specific
    shapes for zero gain, the next str-content message gets the marker instead)."""
    content = message.content
    if not isinstance(content, str) or not content.strip():
        return None
    return message.model_copy(update={"content": [
        {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}},
    ]})


def add_cache_breakpoints(system_message, messages):
    """``(system_message, messages)`` with cache breakpoints applied — pure function,
    inputs untouched. System prompt always marked (it is the longest stable prefix);
    then the last ``_BREAKPOINTS`` Human/AI messages with plain-text content. Never a
    ToolMessage (the marker would be silently stripped — see module docstring)."""
    if isinstance(system_message, SystemMessage):
        system_message = _with_cache_control(system_message) or system_message
    out = list(messages or [])
    marked = 0
    for i in range(len(out) - 1, -1, -1):
        if marked >= _BREAKPOINTS:
            break
        if isinstance(out[i], (HumanMessage, AIMessage)):
            copy = _with_cache_control(out[i])
            if copy is not None:
                out[i] = copy
                marked += 1
    return system_message, out


class PromptCacheMiddleware(AgentMiddleware):
    def _rewrite(self, request):
        system_message, messages = add_cache_breakpoints(
            request.system_message, request.messages)
        return request.override(system_message=system_message, messages=messages)

    def wrap_model_call(self, request, handler):
        return handler(self._rewrite(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._rewrite(request))
