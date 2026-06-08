"""Surface skill usage as a custom stream event.

A skill is "used" when the model reads its ``SKILL.md`` (or a supporting file) via the
deepagents ``read_file`` tool. That built-in tool is not instrumented, so on its own it
emits nothing observable. This middleware watches each model step for a ``read_file``
call whose path is under the mounted skills root (``/skills/``) and emits one ``skill``
event the first time each skill is read in the current turn — giving the UI a
"Skill applied: <name>" indicator without changing how skills are loaded or used.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from .events import emit
from .turn import current_turn

# Matches the mount prefix used in agent.build_skills (CompositeBackend route).
_SKILLS_MARKER = "/skills/"


def _tc_name(tc: Any) -> str:
    return (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")) or ""


def _tc_args(tc: Any) -> dict:
    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
    return args if isinstance(args, dict) else {}


def _skill_name_from_path(path: str) -> str | None:
    """``/skills/data-provider/SKILL.md`` -> ``data-provider`` (the skill's directory)."""
    p = (path or "").strip()
    if _SKILLS_MARKER not in p:
        return None
    rest = p.split(_SKILLS_MARKER, 1)[1].strip("/")
    name = rest.split("/", 1)[0] if rest else ""
    return name or None


def _skill_reads(message: Any) -> list[tuple[str, str]]:
    """``(skill_name, path)`` for every ``read_file`` call in an AIMessage hitting a skill."""
    out: list[tuple[str, str]] = []
    if not isinstance(message, AIMessage):
        return out
    for tc in getattr(message, "tool_calls", None) or []:
        if _tc_name(tc) != "read_file":
            continue
        args = _tc_args(tc)
        path = str(args.get("file_path") or args.get("path") or "")
        name = _skill_name_from_path(path)
        if name:
            out.append((name, path))
    return out


class SkillUsageMiddleware(AgentMiddleware):
    """Emit a ``skill`` event the first time each skill is read in the current turn."""

    def after_model(self, state: dict, runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        reads = _skill_reads(last)
        if not reads:
            return None
        # Dedupe within the turn: don't re-announce a skill already read in an earlier
        # step. `last` is the final message of the turn, so scan everything before it.
        turn = current_turn(messages)
        seen = {name for m in turn[:-1] for name, _ in _skill_reads(m)}
        for name, path in reads:
            if name in seen:
                continue
            seen.add(name)
            emit({"type": "skill", "name": name, "path": path, "state": "loaded"})
        return None
