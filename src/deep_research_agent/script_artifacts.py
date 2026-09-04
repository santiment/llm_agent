"""Scripts reach the USER as an artifact, never as a path in prose.

Two jobs, both on the coding worker (and capture-only on the research sub-agent):

  - CAPTURE. A script the agent wrote with ``write_file`` (then patched with
    ``edit_file``) is replayed from the sub-agent's own messages and emitted ONCE, at
    handoff, as a ``script`` event carrying the finished CODE. That is the app-code
    channel: the frontend renders a collapsed "view script" tab from the event stream.
    Nothing about it is added to any model's context — the orchestrator never learns
    the script exists.

  - SCRUB. The handoff itself is stripped of file machinery before it becomes the
    parent's tool result: the legacy ``SCRIPT: /workspace/x.py`` line, sandbox paths
    left in prose, and bare ``<name>.py`` mentions. A path is a dead end for the
    reader — they cannot open the sandbox, and the run's filesystem is gone when it
    ends — and once the orchestrator has read one it repeats it into the report.

The prompt (``CODING_PROMPT``) already forbids all of this; this module is the
deterministic backstop for when a cheap coder ignores it. Prompt first, because
tokens STREAM: a path the model never writes is a path no UI can show mid-run.

The ONE path allowed through is a ``RESULT FILE:`` line — the labeled hand-off of an
output file too large to print, which the caller must pass to the extract worker.
It is a structured field on its own line, so a UI can hide it; prose paths cannot be
told apart from content and are therefore never allowed.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from .events import emit, new_id
from .turn import current_turn, text_of, tool_calls_of

log = logging.getLogger("deep_research_agent.script_artifacts")

# Code the agent authored, by extension -> the language tag the UI highlights with.
# Only source files are captured: a script is worth showing, an offloaded JSON blob is
# not (it is data the report already summarizes).
_LANGUAGES = {".py": "python", ".sh": "bash", ".sql": "sql", ".js": "javascript",
              ".r": "r", ".jl": "julia"}
# One script's code, capped. A 40k-char script is pathological; the tab still renders
# the head and says so, and the cap keeps a runaway write off the event stream.
MAX_CODE_CHARS = 40_000

# ---- handoff scrubbing -----------------------------------------------------------------
# The legacy handoff field, in any shape a model writes it ("SCRIPT:", "- **Script**:").
_SCRIPT_LINE = re.compile(r"(?im)^[ \t]*[-*>]?[ \t]*\**\s*scripts?\**\s*:.*(?:\n|$)")
# The labeled exception — an output file the CALLER must read. Kept verbatim.
_RESULT_FILE_LINE = re.compile(r"(?i)^[ \t]*[-*>]?[ \t]*\**\s*result file\**\s*:")
# A sandbox path, same shape report_hygiene scrubs from the report.
_SANDBOX_PATH = re.compile(
    r"(?<![\w/])(?:/workspace|/skills|/large_tool_results)(?:/[\w\-]+(?:\.[\w\-]+)*)*")
# A bare source-file name with no directory ("mvrv_corr.py") — the other half of the
# leak: dropping the directory does not make a file the reader can open.
_BARE_SCRIPT = re.compile(r"(?<![\w/.])[\w\-]+\.(?:py|sh|sql|js|ipynb)\b")
_CODE_EXT = re.compile(r"\.(?:py|sh|sql|js|ipynb)$", re.IGNORECASE)


def _replacement(path: str) -> str:
    return "the script" if _CODE_EXT.search(path) else "the data file"


def scrub_handoff(text: str) -> str:
    """Drop ``SCRIPT:`` lines and neutralize file paths/names in a coding handoff.

    Line-oriented so a ``RESULT FILE:`` line passes through untouched — it is the one
    path the caller legitimately needs. Pure; safe on text that has no leak at all
    (returns it unchanged), which is the normal case once the prompt is obeyed."""
    if not text:
        return text
    out = _SCRIPT_LINE.sub("", text)
    lines = []
    for line in out.split("\n"):
        if not _RESULT_FILE_LINE.match(line):
            line = _SANDBOX_PATH.sub(lambda m: _replacement(m.group(0)), line)
            line = _BARE_SCRIPT.sub("the script", line)
        lines.append(line)
    return "\n".join(lines).strip()


# ---- script replay ---------------------------------------------------------------------
def _path_of(args: dict) -> str:
    return str(args.get("file_path") or args.get("path") or "")


def _apply_edit(content: str, args: dict) -> str:
    """``edit_file`` semantics: replace ``old_string`` with ``new_string`` (all
    occurrences when ``replace_all``). A miss leaves the content alone — the replay is
    a best-effort mirror of the sandbox, never the source of truth for it."""
    old, new = str(args.get("old_string") or ""), str(args.get("new_string") or "")
    if not old or old not in content:
        return content
    return content.replace(old, new) if args.get("replace_all") else content.replace(old, new, 1)


def replay_scripts(messages: list) -> dict[str, str]:
    """``{path: final content}`` for every source file written in ``messages``.

    Replays ``write_file`` then each later ``edit_file`` in order, so the captured code
    is what the last `execute` actually ran — not the first draft that failed. Reading
    the sandbox back instead would cost a tool call per script and can't run at all
    once the session is torn down."""
    scripts: dict[str, str] = {}
    for m in messages:
        for tool, args in tool_calls_of(m):
            path = _path_of(args)
            if not path or not _LANGUAGES.get(_ext(path)):
                continue
            if tool == "write_file":
                scripts[path] = str(args.get("content") or "")
            elif tool == "edit_file" and path in scripts:
                scripts[path] = _apply_edit(scripts[path], args)
    return scripts


def _ext(path: str) -> str:
    dot = path.rfind(".")
    return path[dot:].lower() if dot >= 0 else ""


def _last_stop(turn: list) -> int:
    """Index of the previous handoff in ``turn`` (an AIMessage with text and no tool
    calls), or -1. A gate can bounce a sub-agent's handoff back for a revision, so the
    model stops more than once per turn; scripts already emitted at the earlier stop
    must not be re-emitted as a second tab."""
    for i in range(len(turn) - 2, -1, -1):
        m = turn[i]
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) \
                and text_of(m.content).strip():
            return i
    return -1


def _emit_scripts(turn: list, agent: str) -> None:
    """One ``script`` event per script the turn produced, skipping any already emitted
    at an earlier stop UNCHANGED (an edit after a bounce re-emits: the tab must show the
    code that ran). ``name`` is the BASENAME only — a tab title, not a path to act on."""
    stop = _last_stop(turn)
    already = replay_scripts(turn[:stop + 1]) if stop >= 0 else {}
    for path, code in replay_scripts(turn).items():
        if already.get(path) == code:
            continue
        body = code[:MAX_CODE_CHARS]
        emit({"type": "script", "id": new_id(), "agent": agent,
              "name": path.rsplit("/", 1)[-1],
              "language": _LANGUAGES.get(_ext(path), "text"),
              "code": body, "truncated": len(body) < len(code)})


class ScriptArtifactsMiddleware(AgentMiddleware):
    """Attach to a sub-agent that writes code (``coding_spec``, and capture-only on the
    research sub-agent). Stateless across invocations — one instance serves every
    parallel `task` call, so everything is derived from the sub-agent's own messages.

    ``scrub`` is on for the coding worker, whose whole final message is free prose that
    the parent reads verbatim. It is OFF for the research sub-agent: that handoff is
    findings JSON guarded by its own gate, and rewriting it here could break the parse.
    """

    def __init__(self, agent: str, scrub: bool = False) -> None:
        super().__init__()
        self.agent = agent
        self.scrub = scrub

    def after_model(self, state: dict, runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or getattr(last, "tool_calls", None):
            return None  # still working — capture the FINAL version of each script
        turn = current_turn(messages)
        _emit_scripts(turn, self.agent)
        if not self.scrub:
            return None
        content = text_of(last.content)
        cleaned = scrub_handoff(content)
        if cleaned == content.strip():
            return None
        log.info("SCRIPT ARTIFACTS: scrubbed file machinery from the %s handoff", self.agent)
        # Same id => this REPLACES the message in state, so the path never reaches the
        # parent's tool result (deepagents forwards the last non-empty AIMessage text).
        return {"messages": [last.model_copy(update={"content": cleaned})]}
