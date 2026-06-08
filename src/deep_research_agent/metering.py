"""Per-run usage ledger — records how much of each budget category a research run spent.

``make_graph`` is a per-run config-factory, so one ``RunMeter`` is created per run and
shared by:
  - the MCP tool wrapper (``events.instrument_tool``) — counts every MCP call and its raw
    result size, across the orchestrator AND all sub-agents (they share the tool objects);
  - ``UsageMeterMiddleware.after_agent`` — reads the meter at run end and adds token /
    model-call counts from the orchestrator's messages, then emits a ``usage`` event and
    logs one ``RESEARCH USAGE`` line.

Scope (honest):
  - tool_calls / errors / rows / bytes are GLOBAL (include sub-agents).
  - input/output/total tokens and model_calls are ORCHESTRATOR-level — sub-agent model
    usage is not in the orchestrator's state. Capturing it needs a model-level wrapper
    (follow-up).
  - LangGraph does not surface the consumed super-step count to middleware, so model_calls
    / messages are the practical proxy for recursion depth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from .events import emit
from .turn import current_turn

log = logging.getLogger("deep_research_agent.usage")


@dataclass
class RunMeter:
    tool_calls: int = 0
    tool_errors: int = 0
    capped_calls: int = 0
    result_bytes: int = 0
    result_rows: int = 0

    def record_tool_result(self, *, ok: bool, result_bytes: int = 0,
                           result_rows: int | None = None, capped: bool = False) -> None:
        """Called once per finished MCP call (after any rate-limit retries), from any agent.
        Sync int adds only — safe under asyncio's single thread even with parallel calls."""
        self.tool_calls += 1
        if not ok:
            self.tool_errors += 1
        self.result_bytes += int(result_bytes or 0)
        if result_rows:
            self.result_rows += int(result_rows)
        if capped:
            self.capped_calls += 1


class UsageMeterMiddleware(AgentMiddleware):
    """Emit a per-run ``usage`` event + ``RESEARCH USAGE`` log at the end of every run."""

    def __init__(self, meter: RunMeter, *, max_tool_calls: int,
                 max_total_tokens: int, recursion_limit: int) -> None:
        super().__init__()
        self.meter = meter
        self.max_tool_calls = max_tool_calls
        self.max_total_tokens = max_total_tokens
        self.recursion_limit = recursion_limit

    def after_agent(self, state: dict, runtime) -> dict[str, Any] | None:
        msgs = current_turn(state.get("messages") or [])
        in_tok = out_tok = tot_tok = model_calls = 0
        for m in msgs:
            if not isinstance(m, AIMessage):
                continue
            model_calls += 1
            um = getattr(m, "usage_metadata", None)
            if isinstance(um, dict):
                in_tok += int(um.get("input_tokens") or 0)
                out_tok += int(um.get("output_tokens") or 0)
                tot_tok += int(um.get("total_tokens") or 0)

        usage = {
            # GLOBAL (orchestrator + sub-agents)
            "tool_calls": self.meter.tool_calls,
            "tool_errors": self.meter.tool_errors,
            "capped_calls": self.meter.capped_calls,
            "result_rows": self.meter.result_rows,
            "result_bytes": self.meter.result_bytes,
            # ORCHESTRATOR-level (sub-agent model usage not included)
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": tot_tok or (in_tok + out_tok),
            "model_calls": model_calls,
            "messages": len(msgs),
            "tool_calls_in_context": sum(1 for m in msgs if isinstance(m, ToolMessage)),
            # configured ceilings, for at-a-glance "how close did we get?"
            "limits": {
                "max_tool_calls": self.max_tool_calls,
                "max_total_tokens": self.max_total_tokens,
                "recursion_limit": self.recursion_limit,
            },
        }
        log.info("RESEARCH USAGE: %s", usage)
        emit({"type": "usage", **usage})
        return None
