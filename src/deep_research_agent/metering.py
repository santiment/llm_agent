"""Per-run usage ledger — records how much of each budget category a research run spent.

``make_graph`` is a per-run config-factory, so one ``RunMeter`` is created per run and
shared by:
  - the instrumented tools — MCP + custom via ``events.instrument_tool``, web search via
    ``build_search_tool`` — counting every call and its raw result size, across the
    orchestrator AND all sub-agents (they share the tool objects);
  - ``UsageMeterMiddleware.after_agent`` — reads the meter at run end and adds token /
    model-call counts from the orchestrator's messages, then emits a ``usage`` event and
    logs one ``RESEARCH USAGE`` line.

Scope (honest):
  - tool_calls / errors / rows / bytes are GLOBAL (include sub-agents).
  - input/output/total tokens and model_calls are per-agent: the orchestrator's from its
    own messages, each sub-agent fleet's via ``SubagentUsageMiddleware`` (attached to the
    sub-agent specs in agent.py — ``after_agent`` runs once per ``task`` call with that
    sub-agent's isolated message state). The ``usage`` event carries both, plus a grand
    total.
  - cost_usd is BEST-EFFORT: OpenRouter returns the actual charged cost only on
    non-streamed calls that asked for it (``usage: {include: true}``, models.py) —
    streamed calls contribute 0. Treat it as a lower bound, not an invoice.
  - LangGraph does not surface the consumed super-step count to middleware, so model_calls
    / messages are the practical proxy for recursion depth.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from .compaction import compacted_counts
from .events import PROTOCOL_VERSION, emit, engine_version
from .turn import current_turn, message_tokens, tool_calls_in

log = logging.getLogger("deep_research_agent.usage")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def fmt_elapsed(seconds: float | None) -> str:
    """Human run time: ``12.3s`` / ``4m 12s`` / ``1h 04m 12s``; ``n/a`` if the clock never ran."""
    if seconds is None:
        return "n/a"
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"


def sum_usage(messages: list) -> dict[str, Any]:
    """Token / model-call / cost totals over the AIMessages of one agent's transcript.

    Per-message totals come from ``turn.message_tokens`` — the same ladder
    BudgetMiddleware enforces with, so the two can never disagree about the same
    messages. ``cost_usd`` is read defensively from OpenRouter's ``token_usage.cost``
    (present only on non-streamed calls that requested it)."""
    in_tok = out_tok = total = model_calls = 0
    cost = 0.0
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        model_calls += 1
        um = getattr(m, "usage_metadata", None)
        if isinstance(um, dict):
            in_tok += int(um.get("input_tokens") or 0)
            out_tok += int(um.get("output_tokens") or 0)
        total += message_tokens(m)
        rm = getattr(m, "response_metadata", None)
        tu = (rm.get("token_usage") or {}) if isinstance(rm, dict) else {}
        try:
            cost += float(tu.get("cost") or 0.0)
        except (TypeError, ValueError):
            pass
    return {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": total,
            "model_calls": model_calls, "cost_usd": cost}


@dataclass
class RunMeter:
    tool_calls: int = 0
    tool_errors: int = 0
    capped_calls: int = 0
    result_bytes: int = 0
    result_rows: int = 0
    # Per-role sub-agent model usage (research-subagent / extract-subagent), accumulated
    # across every `task` invocation of the run by SubagentUsageMiddleware.
    subagent_usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Run clock, started by UsageMeterMiddleware.before_agent (the first hook of a run):
    # monotonic for the duration, wall-clock ISO for the consumer's own arithmetic.
    started_mono: float | None = None
    started_at: str = ""

    def start(self) -> None:
        self.started_mono = time.monotonic()
        self.started_at = utc_now()

    def elapsed_s(self) -> float | None:
        """Seconds since ``start`` (one decimal), or None when the clock never started."""
        if self.started_mono is None:
            return None
        return round(time.monotonic() - self.started_mono, 1)

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

    def record_subagent_usage(self, role: str, usage: dict[str, Any]) -> None:
        """Fold one finished sub-agent run's ``sum_usage`` into the per-role rollup."""
        u = self.subagent_usage.setdefault(role, {"runs": 0})
        for k, v in usage.items():
            u[k] = u.get(k, 0) + (v or 0)
        u["runs"] += 1


class SubagentUsageMiddleware(AgentMiddleware):
    """Attached to a SUB-AGENT spec (never the orchestrator): ``after_agent`` fires once
    per ``task`` invocation with that sub-agent's own isolated message state, so summing
    it here is what makes the fleet's model usage visible at all — the orchestrator's
    state only ever contains the returned findings text.

    It also marks the sub-agent's START and END (``status`` events ``subagent_start`` /
    ``subagent_done`` + a log line each, carrying role and model). Without these, nothing
    in the stream or the log says a cheaper model ran at all until the run's final
    ``usage`` rollup — which never arrives if the run is cancelled mid-way."""

    def __init__(self, meter: RunMeter, role: str, model: str = "") -> None:
        super().__init__()
        self.meter = meter
        self.role = role
        self.model = model or ""

    def before_agent(self, state: dict, runtime) -> dict[str, Any] | None:
        log.info("SUBAGENT START role=%s model=%s", self.role, self.model or "?")
        emit({"type": "status", "state": "subagent_start", "role": self.role,
              "model": self.model})
        return None

    def after_agent(self, state: dict, runtime) -> dict[str, Any] | None:
        usage = sum_usage(state.get("messages") or [])
        # A long sub-agent run can compact its own context; those tokens were still
        # spent — fold them in, same as the orchestrator total below.
        _calls, compacted_tokens = compacted_counts(state)
        usage["total_tokens"] += compacted_tokens
        self.meter.record_subagent_usage(self.role, usage)
        log.info("SUBAGENT DONE role=%s model=%s model_calls=%d tokens=%d cost_usd=%.6f",
                 self.role, self.model or "?", usage["model_calls"], usage["total_tokens"],
                 usage["cost_usd"])
        emit({"type": "status", "state": "subagent_done", "role": self.role,
              "model": self.model, "model_calls": usage["model_calls"],
              "total_tokens": usage["total_tokens"]})
        return None


class UsageMeterMiddleware(AgentMiddleware):
    """Start the run clock and emit the ``run_start`` version handshake at the start; emit
    the per-run ``usage`` event + ``RESEARCH USAGE`` log — both carrying the run time — at
    the end of every run. ``ResearchOutputMiddleware`` reads the same clock for its end
    ``status``, so the run time reaches the consumer in the success AND the no-report case."""

    def __init__(self, meter: RunMeter, *, max_tool_calls: int,
                 max_total_tokens: int, recursion_limit: int) -> None:
        super().__init__()
        self.meter = meter
        self.max_tool_calls = max_tool_calls
        self.max_total_tokens = max_total_tokens
        self.recursion_limit = recursion_limit

    def before_agent(self, state: dict, runtime) -> dict[str, Any] | None:
        # Version handshake, first event of every run: a consumer pins the
        # protocol_version it renders and can flag a mismatch up-front instead of
        # breaking on an unfamiliar shape mid-run.
        self.meter.start()
        emit({"type": "run_start", "protocol_version": PROTOCOL_VERSION,
              "engine_version": engine_version(), "started_at": self.meter.started_at})
        return None

    def after_agent(self, state: dict, runtime) -> dict[str, Any] | None:
        msgs = current_turn(state.get("messages") or [])
        orch = sum_usage(msgs)
        # Messages summarized away by compaction still spent their tokens — add them back
        # so the reported orchestrator total reflects what the run actually consumed.
        compacted_calls, compacted_tokens = compacted_counts(state)
        orch_total = (orch["total_tokens"] or (orch["input_tokens"] + orch["output_tokens"])) \
            + compacted_tokens

        elapsed = self.meter.elapsed_s()
        sub_total_tokens = sum(u["total_tokens"] for u in self.meter.subagent_usage.values())
        cost = orch["cost_usd"] + sum(u["cost_usd"] for u in self.meter.subagent_usage.values())

        usage = {
            # GLOBAL (orchestrator + sub-agents)
            "tool_calls": self.meter.tool_calls,
            "tool_errors": self.meter.tool_errors,
            "capped_calls": self.meter.capped_calls,
            "result_rows": self.meter.result_rows,
            "result_bytes": self.meter.result_bytes,
            # ORCHESTRATOR-level (sub-agent model usage broken out under "subagents")
            "input_tokens": orch["input_tokens"],
            "output_tokens": orch["output_tokens"],
            "total_tokens": orch_total,
            "model_calls": orch["model_calls"],
            "messages": len(msgs),
            "tool_calls_in_context": tool_calls_in(msgs),
            # What compaction summarized out of the transcript (already counted above).
            "compacted_tool_calls": compacted_calls,
            "compacted_tokens": compacted_tokens,
            # Sub-agent fleet model usage, per role, + a whole-run token grand total.
            "subagents": self.meter.subagent_usage,
            "total_tokens_all_agents": orch_total + sub_total_tokens,
            # BEST-EFFORT actual cost (OpenRouter, non-streamed calls only — see module doc).
            "cost_usd": round(cost, 6),
            # Wall-clock run time — the first thing asked when a run ends without a report.
            # Always present; None only if before_agent never ran (offline tests).
            "elapsed_s": elapsed,
            "elapsed": fmt_elapsed(elapsed),
            "started_at": self.meter.started_at,
            "finished_at": utc_now(),
            # configured ceilings, for at-a-glance "how close did we get?"
            "limits": {
                "max_tool_calls": self.max_tool_calls,
                "max_total_tokens": self.max_total_tokens,
                "recursion_limit": self.recursion_limit,
            },
        }
        log.info("RESEARCH USAGE (run time %s): %s", fmt_elapsed(elapsed), usage)
        emit({"type": "usage", **usage})
        return None
