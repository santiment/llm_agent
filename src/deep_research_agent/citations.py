"""Output middleware: harvest sources, persist the final report, signal lifecycle.

After the agent finishes:
  1. Scan every ToolMessage for URLs -> a deduped, ordered ``sources`` list in state
     (structured citations for the host app, independent of the inline [n] markers).
  2. Persist the final report into ``final_report`` so a thread-state read after
     completion finds it. The report is taken from the ``submit_report`` tool call —
     the agent's one explicit deliverable — so there is no "which message is the
     report" guessing and no chance of a duplicate. The ``report`` stream event is
     emitted by the tool itself; we only emit one here as a fallback if the agent
     finished without calling it.

The inline [n] interleaving is the writer model's job; this provides the structured
mirror, not a post-hoc renumber.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from typing_extensions import NotRequired

from .compaction import turn_spend
from .completion import MAX_NUDGES
from .events import domain_of, emit
from .metering import fmt_elapsed
from .report_hygiene import collapse_series, lint_citations, scrub_report
from .turn import (NUDGE_NAME, called, count_nudges, current_turn, did_research_work,
                   is_json_object_dump, looks_delivered, raw_text, text_of,
                   tool_calls_of)

log = logging.getLogger("deep_research_agent.citations")

_URL_RE = re.compile(r"https?://[^\s\)\]\}\"'<>]+")


def _clean_url(u: str) -> str:
    return u.rstrip(".,);]'\"")


def _report_from_submit(messages: list) -> str:
    """The argument of the most-recent submit_report tool call, if any."""
    for m in reversed(messages):
        for name, args in tool_calls_of(m):
            if name == "submit_report":
                rep = args.get("report_markdown")
                if isinstance(rep, str) and rep.strip():
                    return rep
    return ""


class ResearchState(AgentState):
    final_report: NotRequired[str]
    sources: NotRequired[list[dict[str, Any]]]


class ResearchOutputMiddleware(AgentMiddleware):
    state_schema = ResearchState

    def __init__(self, *, max_tool_calls: int, max_total_tokens: int, tool_names=(),
                 meter=None, max_run_seconds: int = 0) -> None:
        super().__init__()
        self.meter = meter  # RunMeter; its clock puts the run time on the end status
        # Ceilings are needed to distinguish "ran out of budget" from "just gave up" when
        # classifying WHY a run ended without a report.
        self.max_tool_calls = max_tool_calls
        self.max_total_tokens = max_total_tokens
        self.max_run_seconds = max_run_seconds
        self.tool_names = tuple(tool_names or ())

    def after_agent(self, state: dict, runtime) -> dict[str, Any] | None:
        # Scope to the CURRENT turn only — a thread accumulates messages across runs,
        # so harvesting/report-finding over all messages would re-surface a prior turn's
        # report and sources on an unrelated follow-up.
        messages = current_turn(state.get("messages") or [])

        seen: dict[str, int] = {}
        sources: list[dict[str, Any]] = []
        for m in messages:
            if not isinstance(m, ToolMessage):
                continue
            text = raw_text(m.content)
            for raw in _URL_RE.findall(text):
                url = _clean_url(raw)
                if url and url not in seen:
                    seen[url] = len(sources) + 1
                    sources.append({"index": len(sources) + 1, "url": url, "domain": domain_of(url)})

        report = _report_from_submit(messages)
        via_tool = bool(report)
        researched = did_research_work(messages)
        salvaged = False
        if not via_tool and researched:
            # No submit_report this turn. Salvage the last AI text as the report ONLY if it
            # actually reads like a report (substantial / has a heading / Sources). A bare
            # "Now let me run X" intent stub is NOT a report — promoting it would show the
            # user a fake report with a success footer (the exact bug). Leave it empty so the
            # classifier below flags the run as an error instead.
            for m in reversed(messages):
                if isinstance(m, AIMessage):
                    txt = text_of(m.content)
                    if not txt.strip():
                        continue
                    # Promote real prose only. A raw JSON blob (e.g. echoed findings
                    # schema) is NOT a report — showing it as one is the garbage we're
                    # guarding against; leave it unsalvaged so the run flags as an error.
                    if looks_delivered(txt) and not is_json_object_dump(txt):
                        report, salvaged = txt, True
                    break

        # Deterministic last-mile hygiene: strip any leaked data-layer machinery so the
        # persisted final_report (and the salvage emit below) match what submit_report already
        # scrubbed on its live emit. Idempotent, so double-scrubbing the submit path is safe.
        report = scrub_report(report, self.tool_names)
        report = collapse_series(report)  # drop raw series the gate could not get rewritten

        # submit_report already emitted the live `report` event; only emit on fallback.
        if report and not via_tool:
            emit({"type": "report", "markdown": report})

        # ---- Authoritative end-of-run determination: EXACTLY why the turn ended ----
        # A research turn MUST end by delivering a report via submit_report. Anything else
        # (gave up mid-research, hit the budget, wrote prose instead of calling the tool) is
        # an error and is logged + emitted as one. NOTE: this only runs on a clean end — its
        # ABSENCE in the logs means the run died via an exception (e.g. GraphRecursionError)
        # before after_agent, which the host streams as a stream error.
        calls, tokens = turn_spend(state, messages)  # includes compacted-away spend
        nudges = count_nudges(messages, NUDGE_NAME)
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        end_state, reason, detail = self._classify(
            via_tool=via_tool, researched=researched, salvaged=salvaged,
            clarified=called(messages, "request_clarification"),
            calls=calls, tokens=tokens, nudges=nudges)

        elapsed = self.meter.elapsed_s() if self.meter is not None else None
        if (end_state == "error" and reason == "ended_without_report" and self.max_run_seconds
                and elapsed is not None and elapsed >= self.max_run_seconds):
            reason = "budget_exhausted"
            detail = (f"Hit the run time ceiling ({fmt_elapsed(elapsed)} of "
                      f"{fmt_elapsed(self.max_run_seconds)}) before delivering a report.")
        if elapsed is not None:
            detail = f"{detail} Run time {fmt_elapsed(elapsed)}."

        cite = lint_citations(report)
        summary = {
            "reason": reason, "detail": detail, "elapsed_s": elapsed, "submit_report": via_tool,
            "salvaged": salvaged,
            "tool_results": calls, "tokens": tokens, "nudges": nudges,
            "report_chars": len(report or ""), "sources": len(sources),
            "last_ai_had_tool_calls": bool(getattr(last_ai, "tool_calls", None)),
            "citations": cite,
            "limits": {"max_tool_calls": self.max_tool_calls,
                       "max_total_tokens": self.max_total_tokens,
                       "max_run_seconds": self.max_run_seconds},
        }
        if end_state == "error":
            log.error("RUN ENDED WITHOUT REPORT after %s: %s", fmt_elapsed(elapsed), summary)
        elif reason == "report_salvaged":
            # Delivered, but through the recovery path — count these per model/tier; a
            # rising rate means the resubmit nudge isn't landing on that model.
            log.warning("RUN END (%s) after %s: %s", reason, fmt_elapsed(elapsed), summary)
        else:
            log.info("RUN END (%s) after %s: %s", reason, fmt_elapsed(elapsed), summary)
        # Non-fatal quality signal: a delivered report whose inline [n] and Sources list don't
        # match (orphan or dangling citations). Warn, don't fail — the report still stands.
        if cite["orphans"] or cite["danglers"]:
            log.warning("CITATION MISMATCH: orphans=%s danglers=%s (inline=%d listed=%d)",
                        cite["orphans"], cite["danglers"], cite["inline"], cite["listed"])
        emit({"type": "status", "state": end_state, "reason": reason, "detail": detail,
              "tool_calls": calls, "tokens": tokens, "report_chars": len(report or ""),
              "elapsed_s": elapsed, "elapsed": fmt_elapsed(elapsed)})

        return {"final_report": report, "sources": sources}

    def _classify(self, *, via_tool: bool, researched: bool, salvaged: bool, clarified: bool,
                  calls: int, tokens: int, nudges: int) -> tuple[str, str, str]:
        """Map the turn's end-state to ``(status_state, reason_code, human_detail)``.

        Order matters: the first matching cause wins, most-specific first. A research turn
        with no report at all is an ``error``; a salvaged plain-text report counts as
        ``done`` (the user got the content) under its own reason so it stays countable."""
        if via_tool:
            return "done", "report_delivered", "submit_report delivered the final report."
        if clarified:
            return "done", "awaiting_clarification", "Paused to ask the user a clarifying question."
        if not researched:
            # No research and no report → a plain conversational reply, which is legitimate.
            return "done", "direct_answer", "Answered conversationally; no research step ran."
        if calls >= self.max_tool_calls or tokens >= self.max_total_tokens:
            return ("error", "budget_exhausted",
                    f"Hit the run budget ({calls}/{self.max_tool_calls} tool calls, "
                    f"{tokens:,}/{self.max_total_tokens:,} tokens) before delivering a report.")
        if nudges >= MAX_NUDGES:
            return ("error", "stalled_after_nudges",
                    f"Model kept stopping mid-research with no tool call; force-completion "
                    f"gave up after {MAX_NUDGES} nudges and it never called submit_report.")
        if salvaged:
            # The content DID reach the user (recovered + scrubbed + emitted as the
            # report) — a recovery, not a failure. Surfaced as "done" with its own
            # reason so the UI stays calm while operators can still count occurrences.
            return ("done", "report_salvaged",
                    "Report recovered from a plain chat message (the model skipped "
                    "submit_report despite the resubmit nudge); delivered normally.")
        return ("error", "ended_without_report",
                "Turn ended after research with no submit_report and no salvageable report.")
