"""Event-protocol contract — the shapes any frontend renders, pinned as code.

``EVENT_SCHEMAS`` registers every event type and its required keys; ``emit`` warns
(never raises) on a violation. ``run_start`` opens every run with the version
handshake (protocol_version + engine_version) so a consumer detects mismatch up-front.
These tests pin the registry, the validation behavior, and that every ``emit`` call
site in the package uses a registered type.

Runs with plain Python (``python tests/test_events_protocol.py``) — no pytest needed —
and is also pytest-discoverable. No network, no API keys.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import deep_research_agent.events as events
from deep_research_agent.events import (EVENT_SCHEMAS, PROTOCOL_VERSION, STATUS_STATES,
                                        emit, engine_version)
from deep_research_agent.metering import RunMeter, UsageMeterMiddleware
from conftest import capture_events_cm


class _CaptureWarnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def __enter__(self) -> "_CaptureWarnings":
        logging.getLogger("deep_research_agent.events").addHandler(self)
        return self

    def __exit__(self, *exc) -> None:
        logging.getLogger("deep_research_agent.events").removeHandler(self)


def test_registry_pins_the_full_event_vocabulary():
    # Golden set: removing/renaming a type here is a BREAKING protocol change —
    # bump PROTOCOL_VERSION and update the consumers before touching this list.
    assert set(EVENT_SCHEMAS) == {
        "run_start", "search_query", "search_results", "source",
        "mcp_call", "mcp_result", "tool_call", "tool_result",
        "skill", "report", "status", "clarification", "usage", "subagent_findings",
    }


def test_every_emit_call_site_uses_a_registered_type():
    # Scan the package source for `"type": "<literal>"` (and the f-string kinds in
    # instrument_tool) — an emit site with an unregistered type is protocol drift.
    found: set[str] = set()
    for p in Path(events.__file__).parent.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        found.update(re.findall(r'"type":\s*"([a-z_]+)"', text))
        for suffix in re.findall(r'"type":\s*f"\{kind\}_(call|result)"', text):
            found.update({f"mcp_{suffix}", f"tool_{suffix}"})
    # Message CONTENT blocks (caching.py's cache_control markers), not stream events.
    found -= {"text", "ephemeral"}
    assert found, "no emit sites found — scan regex broke?"
    unregistered = found - set(EVENT_SCHEMAS)
    assert not unregistered, f"emit sites with unregistered types: {unregistered}"


def test_status_states_pinned():
    # Golden set, same contract as EVENT_SCHEMAS: a consumer switches on these, so
    # removing/renaming one is a BREAKING change; adding one must land here too
    # (emit warns on any state not in STATUS_STATES).
    assert STATUS_STATES == {
        "mcp_ready", "mcp_error", "budget_soft", "budget_halt", "revising",
        "compacting", "compacted", "loop_detected", "loop_halt", "done", "error",
        "subagent_start", "subagent_done",
    }


def test_unregistered_status_state_warns():
    with _CaptureWarnings() as warns, capture_events_cm() as captured:
        emit({"type": "status", "state": "made_up_state"})
    assert len(captured) == 1  # still emitted — observability never breaks a run
    assert any("unregistered status state" in m for m in warns.messages)


def test_valid_event_passes_without_warning():
    with _CaptureWarnings() as warns, capture_events_cm() as captured:
        emit({"type": "report", "markdown": "# Hi"})
    assert captured == [{"type": "report", "markdown": "# Hi"}]
    assert warns.messages == []


def test_missing_required_key_warns_but_still_emits():
    with _CaptureWarnings() as warns, capture_events_cm() as captured:
        emit({"type": "report"})  # missing "markdown"
    assert len(captured) == 1  # observability must never break a run
    assert any("missing required keys" in m and "markdown" in m for m in warns.messages)


def test_unregistered_type_warns_but_still_emits():
    with _CaptureWarnings() as warns, capture_events_cm() as captured:
        emit({"type": "brand_new_thing"})
    assert len(captured) == 1
    assert any("unregistered event type" in m for m in warns.messages)


def test_run_start_handshake():
    mw = UsageMeterMiddleware(RunMeter(), max_tool_calls=1, max_total_tokens=1,
                              recursion_limit=1)
    with _CaptureWarnings() as warns, capture_events_cm() as captured:
        mw.before_agent({}, None)
    assert warns.messages == []
    (ev,) = captured
    assert ev["type"] == "run_start"
    assert ev["protocol_version"] == PROTOCOL_VERSION
    assert isinstance(ev["engine_version"], str) and ev["engine_version"]
    assert ev["started_at"].endswith("Z")        # UTC anchor for run time if the run dies early


def test_engine_version_never_raises():
    assert isinstance(engine_version(), str)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all event-protocol tests passed")
