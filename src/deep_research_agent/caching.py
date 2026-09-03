"""Prompt-cache breakpoints (Anthropic-style ``cache_control``), injected per model request.

One breakpoint on the system prompt plus one on each of the last ``_BREAKPOINTS`` Human/AI
messages, so the stable prefix is a cache hit on the next call. Never on a ToolMessage:
langchain-openai drops ``cache_control`` from tool-message content. Requests are rewritten
through ``request.override`` with copies; graph-state messages are never mutated. Wired only
for OpenRouter (agent.py), which forwards the markers to caching providers and strips them
elsewhere.
"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

_BREAKPOINTS = 2


def _with_cache_control(message):
    """A copy of ``message`` with its text as one cache-marked block; None when the content
    is not a plain non-empty string (block-list content is left alone)."""
    content = message.content
    if not isinstance(content, str) or not content.strip():
        return None
    return message.model_copy(update={"content": [
        {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}},
    ]})


def add_cache_breakpoints(system_message, messages):
    """``(system_message, messages)`` with breakpoints applied; inputs untouched."""
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
