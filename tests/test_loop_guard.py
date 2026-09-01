"""Pin the identical-tool-call loop guard (loop_guard.py): fingerprints are
(tool, args, result), only a loop that is STILL RUNNING triggers (keyed on the
most recent call), soft = capped nudge, hard = jump to end.

Runs with plain Python (``python tests/test_loop_guard.py``) — no pytest needed — and
is also pytest-discoverable.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research_agent.loop_guard import (HARD_REPEATS, MAX_LOOP_NUDGES,
                                            SOFT_REPEATS, LoopGuardMiddleware)
from deep_research_agent.turn import LOOP_NUDGE_NAME

_ID = [0]


def _pair(name: str = "fetch_metric", args: dict | None = None,
          result: str = "same rows") -> list:
    """One completed call: the requesting AIMessage + its ToolMessage."""
    _ID[0] += 1
    cid = f"c{_ID[0]}"
    return [
        AIMessage("", tool_calls=[{"name": name, "args": args or {"m": "price"}, "id": cid}]),
        ToolMessage(result, tool_call_id=cid, name=name),
    ]


def _state(*call_lists: list) -> dict:
    msgs: list = [HumanMessage("research X")]
    for calls in call_lists:
        msgs.extend(calls)
    return {"messages": msgs}


def _is_loop_nudge(update) -> bool:
    msgs = (update or {}).get("messages") or []
    return any(getattr(m, "name", None) == LOOP_NUDGE_NAME for m in msgs)


def test_under_soft_threshold_is_quiet() -> None:
    mw = LoopGuardMiddleware()
    state = _state(*[_pair() for _ in range(SOFT_REPEATS - 1)])
    assert mw.before_model(state, None) is None


def test_soft_repeats_nudge_once() -> None:
    mw = LoopGuardMiddleware()
    update = mw.before_model(_state(*[_pair() for _ in range(SOFT_REPEATS)]), None)
    assert _is_loop_nudge(update), update
    assert "jump_to" not in update


def test_varied_args_or_results_never_trigger() -> None:
    mw = LoopGuardMiddleware()
    # Same tool, different args → different fingerprints.
    varied = [_pair(args={"m": f"metric-{i}"}) for i in range(HARD_REPEATS)]
    assert mw.before_model(_state(*varied), None) is None
    # Same call, different RESULTS (e.g. paging) → not a loop either.
    paging = [_pair(result=f"rows page {i}") for i in range(HARD_REPEATS)]
    assert mw.before_model(_state(*paging), None) is None


def test_old_repeats_do_not_retrigger_after_behavior_changes() -> None:
    mw = LoopGuardMiddleware()
    # A past loop, then the model moved on: latest call is unique → stay quiet.
    state = _state(*[_pair() for _ in range(SOFT_REPEATS + 1)],
                   _pair(name="web_search", args={"q": "new"}))
    assert mw.before_model(state, None) is None


def test_nudge_is_capped_then_hard_stop_ends_run() -> None:
    mw = LoopGuardMiddleware()
    repeats = [_pair() for _ in range(SOFT_REPEATS + 1)]
    state = _state(*repeats)
    state["messages"].extend(
        HumanMessage("n", name=LOOP_NUDGE_NAME) for _ in range(MAX_LOOP_NUDGES))
    assert mw.before_model(state, None) is None  # cap reached — no third nudge

    hard = _state(*[_pair() for _ in range(HARD_REPEATS)])
    assert mw.before_model(hard, None) == {"jump_to": "end"}


if __name__ == "__main__":
    test_under_soft_threshold_is_quiet()
    test_soft_repeats_nudge_once()
    test_varied_args_or_results_never_trigger()
    test_old_repeats_do_not_retrigger_after_behavior_changes()
    test_nudge_is_capped_then_hard_stop_ends_run()
    print("OK — loop guard verified.")
