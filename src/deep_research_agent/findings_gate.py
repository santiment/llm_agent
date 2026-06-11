"""Sub-agent findings gate — ``report_gate``'s twin, pointed at the other deliverable.

The `task` tool relays a sub-agent's FINAL message to the orchestrator verbatim, and
that text is the only thing the orchestrator ever sees of the unit's work. With
sub-agents on a cheaper model tier, that handoff must be checkable: a weaker model
economizes first on attribution, and an unsourced finding poisons the report's
citations downstream. The contract (one JSON object: ``summary`` + ``findings[]``,
every finding carrying its ``source``) is stated in ``prompts.FINDINGS_FORMAT``;
this module enforces it deterministically — no model in the loop. It also checks
PROVENANCE: non-empty findings with zero tool calls in the run means the model
answered from memory — bounced, since findings must come from tool results.

An invalid final message is bounced back ONCE with the specific problems (the same
jump-to-model pattern as ``ForceCompletionMiddleware``), then accepted as-is: prose
findings degrade gracefully — the orchestrator can still read them — so this gate
must never fail a run over formatting.

Design note: deepagents' ``SubAgent`` supports native structured output
(``response_format``/ToolStrategy), which would replace this prompt+gate approach
with schema-validated tool calling. Deliberately not used yet: the cheap tier this
feature targets is exactly where forced tool-call output is least reliable through
OpenRouter, and the gate must degrade to prose, not fail. Revisit once tiered models
are proven in production.
"""

from __future__ import annotations

import json
import logging
from itertools import islice
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .events import emit
from .turn import FINDINGS_NUDGE_NAME, count_nudges, text_of

log = logging.getLogger("deep_research_agent.findings_gate")

MAX_FINDINGS_NUDGES = 1

_DECODER = json.JSONDecoder()

# Bound the prose scan: one raw_decode attempt per "{" position, at most this many.
# A real findings message has a handful of objects; past this it's garbage anyway.
_MAX_SCAN_STARTS = 64

# The sub-agent's system prompt (with the full RETURN FORMAT) is still in its context
# at bounce time — point at it rather than restating the schema here, so the contract
# is written in one place and cannot drift.
_BOUNCE = (
    "Your findings were NOT accepted. Problems:\n{problems}\n"
    "Fix ONLY these problems and resend your COMPLETE findings as one JSON object in "
    "the RETURN FORMAT from your instructions. Do not alter any finding, number, or "
    "source you already gathered."
)


def extract_findings(text: str) -> dict | None:
    """Parse the findings object out of a sub-agent's final message.

    Splitting on ``` handles bare JSON (one segment) and any fence style without
    regex. Chatty cheap models also emit preamble objects or prose around the real
    one, so collect every object that parses and prefer the LAST findings-shaped one
    (has "summary" or "findings"). None if nothing parses."""
    text = (text or "").strip()
    if not text:
        return None
    objs: list[dict] = []
    for seg in text.split("```"):
        s = seg.strip()
        if s[:4].lower() == "json":
            s = s[4:].lstrip()
        if not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except ValueError:
            continue
        if isinstance(obj, dict):
            objs.append(obj)
    if not objs:
        # JSON embedded in prose: attempt a decode at each "{" (bounded). raw_decode
        # stops at the end of one complete value, so neighboring objects parse
        # independently instead of concatenating into one unparseable span.
        starts = islice((i for i, ch in enumerate(text) if ch == "{"), _MAX_SCAN_STARTS)
        for i in starts:
            try:
                obj, _ = _DECODER.raw_decode(text, i)
            except ValueError:
                continue
            if isinstance(obj, dict):
                objs.append(obj)
    if not objs:
        return None
    for obj in reversed(objs):  # the real findings object is typically the last
        if "summary" in obj or "findings" in obj:
            return obj
    return objs[-1]


def findings_problems(text: str) -> list[str]:
    """Deterministic contract check of a raw message. Empty list == valid."""
    return _problems(extract_findings(text))


def _problems(obj: dict | None) -> list[str]:
    """Contract check of the parsed object. Empty ``findings`` is deliberately
    allowed (an honest "nothing found" beats pressure to fabricate); what is
    non-negotiable is the object shape and a source on every finding."""
    if obj is None:
        return ["the message is not a single parseable JSON object"]
    problems: list[str] = []
    summary = obj.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        problems.append('"summary" must be a non-empty string')
    findings = obj.get("findings")
    if not isinstance(findings, list):
        problems.append('"findings" must be a list (empty is allowed if the unit yielded nothing)')
    else:
        for i, f in enumerate(findings):
            if not isinstance(f, dict):
                problems.append(f"findings[{i}] must be an object")
                continue
            if not str(f.get("finding") or "").strip():
                problems.append(f'findings[{i}] is missing "finding"')
            if not str(f.get("source") or "").strip():
                problems.append(
                    f'findings[{i}] is missing "source" — every finding must be attributed')
    gaps = obj.get("gaps")
    if gaps is not None and not isinstance(gaps, list):
        problems.append('"gaps", when present, must be a list')
    return problems


def _unit_label(messages: list) -> str:
    """A short label for the sub-agent's assigned unit — the orchestrator's task
    description, i.e. the first real (non-nudge) human message in the sub-agent's state."""
    for m in messages:
        if isinstance(m, HumanMessage) and getattr(m, "name", None) != FINDINGS_NUDGE_NAME:
            return text_of(m.content).strip()[:140]
    return ""


def _emit_findings_event(messages: list, obj: dict) -> None:
    """Emit the validated findings as a typed ``subagent_findings`` event for the UI to
    render as a folded table. Rides the same ``custom`` stream as the sub-agent's
    mcp_call rows; a host that ignores the type loses nothing."""
    findings = obj.get("findings")
    emit({
        "type": "subagent_findings",
        "unit": _unit_label(messages),
        "summary": str(obj.get("summary") or ""),
        "findings": findings if isinstance(findings, list) else [],
        "gaps": obj.get("gaps") if isinstance(obj.get("gaps"), list) else [],
    })


class SubagentFindingsMiddleware(AgentMiddleware):
    """Attached to the research sub-agent (``subagent_spec["middleware"]``), NOT the
    orchestrator. Stateless across invocations on purpose — one instance serves every
    parallel `task` call, so the nudge cap is counted from the sub-agent's own message
    state, never from instance attributes."""

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: dict, runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage):
            return None
        if getattr(last, "tool_calls", None):
            return None  # still working — the loop continues on its own
        content = text_of(last.content)
        if not content.strip():
            return None  # empty terminations are another middleware's failure mode

        obj = extract_findings(content)
        problems = _problems(obj)
        # Provenance: a weak model returning plausible findings WITHOUT having called
        # a single tool is fabricating from memory. (Empty findings with no tools is a
        # legitimate honest "nothing".)
        claims_findings = obj is not None and isinstance(obj.get("findings"), list) \
            and obj["findings"]
        if claims_findings and not any(isinstance(m, ToolMessage) for m in messages):
            problems.append(
                "findings were returned without a single tool call this run — gather data "
                "with the tools first; findings must come from tool results, not memory")
        if not problems:
            # Accepted, valid findings — surface them as a structured event so the UI can
            # render a folded findings table instead of the raw JSON that streams as
            # thinking. Best-effort (no-op offline); never blocks the handoff.
            _emit_findings_event(messages, obj)
            return None

        if count_nudges(messages, FINDINGS_NUDGE_NAME) >= MAX_FINDINGS_NUDGES:
            log.warning(
                "FINDINGS GATE: accepting non-conforming sub-agent output after %d nudge(s) "
                "(prose degrades gracefully): %s", MAX_FINDINGS_NUDGES, problems)
            return None
        log.warning("FINDINGS GATE: bouncing sub-agent output (nudge 1/%d): %s",
                    MAX_FINDINGS_NUDGES, problems)
        emit({"type": "status", "state": "revising", "reason": "subagent_findings",
              "detail": "; ".join(problems)})
        bullets = "\n".join(f"- {p}" for p in problems)
        return {
            "jump_to": "model",
            "messages": [HumanMessage(content=_BOUNCE.format(problems=bullets),
                                      name=FINDINGS_NUDGE_NAME)],
        }
