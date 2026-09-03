"""Pin the prompt-cache breakpoint rewriter (caching.py): the system prompt and the
last two Human/AI plain-text messages get a ``cache_control`` block; ToolMessages are
NEVER marked (langchain-openai silently strips the marker there); originals are never
mutated (they live in graph state).

Runs with plain Python (``python tests/test_prompt_cache.py``) — no pytest needed — and
is also pytest-discoverable.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deep_research_agent.caching import PromptCacheMiddleware, add_cache_breakpoints


def _is_marked(message) -> bool:
    c = message.content
    return (isinstance(c, list) and len(c) == 1 and isinstance(c[0], dict)
            and c[0].get("cache_control") == {"type": "ephemeral"}
            and isinstance(c[0].get("text"), str))


def test_system_prompt_gets_a_breakpoint() -> None:
    sys_out, _ = add_cache_breakpoints(SystemMessage("You are…"), [])
    assert _is_marked(sys_out) and sys_out.content[0]["text"] == "You are…"


def test_last_two_human_ai_marked_toolmessage_skipped() -> None:
    msgs = [HumanMessage("q"), AIMessage("plan"),
            ToolMessage("rows", tool_call_id="1"), AIMessage("more")]
    _, out = add_cache_breakpoints(None, msgs)
    assert _is_marked(out[3]) and _is_marked(out[1])   # the two newest Human/AI
    assert isinstance(out[2].content, str)             # ToolMessage untouched
    assert isinstance(out[0].content, str)             # only 2 breakpoints spent


def test_unmarkable_content_is_skipped_not_marked() -> None:
    msgs = [HumanMessage("q"), AIMessage("real text"),
            AIMessage(""),                                        # empty
            AIMessage([{"type": "reasoning", "reasoning": "…"}])]  # block list
    _, out = add_cache_breakpoints(None, msgs)
    assert not _is_marked(out[2]) and not _is_marked(out[3])
    assert _is_marked(out[1]) and _is_marked(out[0])  # markers fell through to these


def test_originals_are_never_mutated() -> None:
    sys_in = SystemMessage("sys")
    msgs = [HumanMessage("q"), AIMessage("a")]
    add_cache_breakpoints(sys_in, msgs)
    assert sys_in.content == "sys"
    assert msgs[0].content == "q" and msgs[1].content == "a"


class _Req:
    """Duck-typed ModelRequest: just the two fields the middleware rewrites."""

    def __init__(self, system_message, messages):
        self.system_message, self.messages = system_message, messages

    def override(self, **kw):
        return _Req(kw.get("system_message", self.system_message),
                    kw.get("messages", self.messages))


def test_middleware_rewrites_via_override_sync_and_async() -> None:
    mw = PromptCacheMiddleware()
    req = _Req(SystemMessage("sys"), [HumanMessage("q")])

    got = mw.wrap_model_call(req, lambda r: r)
    assert _is_marked(got.system_message) and _is_marked(got.messages[0])

    async def handler(r):
        return r
    got = asyncio.run(mw.awrap_model_call(req, handler))
    assert _is_marked(got.system_message) and _is_marked(got.messages[0])
    assert req.messages[0].content == "q"  # request untouched


if __name__ == "__main__":
    test_system_prompt_gets_a_breakpoint()
    test_last_two_human_ai_marked_toolmessage_skipped()
    test_unmarkable_content_is_skipped_not_marked()
    test_originals_are_never_mutated()
    test_middleware_rewrites_via_override_sync_and_async()
    print("OK — prompt-cache breakpoints verified.")
