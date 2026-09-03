"""Minimal consumer — any app talks to the agent like this. No package import
needed; it's just HTTP/SSE against the LangGraph server.

    python examples/client.py "What are the recent trends...?"   # inline
    python examples/client.py @prompt.txt                        # read a file (long prompts)
    python examples/client.py -                                  # read STDIN
    cat prompt.txt | python examples/client.py                   # piped STDIN

Handles every event type in the protocol (``EVENT_SCHEMAS`` in
``deep_research_agent/events.py``): the ``run_start`` handshake, searches, sources, tool
calls, skills, clarifications, sub-agent findings, usage and the final report — plus the
assistant thinking tokens that stream on the ``messages`` channel. Mirrors what your
frontend renders.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from langgraph_sdk import get_client

_DEFAULT_Q = "Give me a deep research report on BDCs."

# run.sh advertises DRA_HOST/DRA_PORT; honour them here too, or a non-default server
# silently gets no traffic while this client talks to a dead 127.0.0.1:2024. Bare PORT
# is a compatibility fallback — it's a name every other tool also uses.
BASE_URL = os.environ.get(
    "DRA_URL",
    f"http://{os.environ.get('DRA_HOST', '127.0.0.1')}"
    f":{os.environ.get('DRA_PORT') or os.environ.get('PORT') or '2024'}")

# The protocol version this client understands. The agent opens every run with a
# `run_start` event carrying its own; a mismatch means the shape of later events may have
# changed under us, so say so at the top rather than mis-render halfway through.
EXPECTED_PROTOCOL = 1


def fmt_elapsed(seconds: float) -> str:
    s = max(0.0, seconds)
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"


def read_question(argv: list[str]) -> str:
    """Resolve the prompt from an arg, an ``@file``, ``-``/pipe (STDIN), or default.

    Long prompts are painful to paste into a single-line box (or quote on the
    shell) — keep them in a file and pass ``@prompt.txt``, or pipe them in.
    """
    if len(argv) > 1 and argv[1] not in ("", "-"):
        arg = argv[1]
        if arg.startswith("@"):
            return Path(arg[1:]).expanduser().read_text(encoding="utf-8").strip()
        return arg
    # No usable arg: read STDIN if it's piped or explicitly requested with "-".
    if (len(argv) > 1 and argv[1] == "-") or not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    return _DEFAULT_Q


async def stream_turn(client, thread_id: str, content: str) -> list[str]:
    """Stream one run on ``thread_id``. Returns the clarifying questions the agent is
    waiting on (``[]`` if it finished without asking)."""
    pending: list[str] = []
    t0 = time.monotonic()
    outcome = "ended"
    try:
        async for chunk in _stream(client, thread_id, content):
            if chunk.event == "error":
                # The host streams an exception (e.g. recursion limit) as a stream error: the
                # agent's own end events never arrive, so the wall-clock line below is all
                # that says how long it ran.
                outcome = "failed"
                print(f"\n✗ run error: {chunk.data}")
            elif chunk.event == "custom":
                d = chunk.data
                if d.get("type") == "status" and d.get("state") in ("done", "error"):
                    outcome = d["state"]
                    # `detail` already ends with "Run time …" — the agent's own clock.
                    print(f"\n[{d['state']}] {d.get('reason')}: {d.get('detail')}")
                else:
                    render_event(d, pending)
            elif chunk.event == "messages":
                # streamed assistant thinking tokens
                for m in chunk.data:
                    text = m.get("content") if isinstance(m, dict) else None
                    if isinstance(text, str):
                        print(text, end="", flush=True)
    finally:
        # ALWAYS shown — report, error event, exception or Ctrl-C: the client's own clock,
        # so it never depends on the server's end events arriving.
        print(f"\n[run {outcome} · {fmt_elapsed(time.monotonic() - t0)} wall-clock]")
    return pending


def _stream(client, thread_id: str, content: str):
    return client.runs.stream(
        thread_id,
        "deep_research_agent",
        input={"messages": [{"role": "user", "content": content}]},
        # Per-run overrides; omit to use the server's .env defaults. Models are chosen by
        # TIER NAME only — per-model keys are ignored with a warning (see README).
        config={"configurable": {
            "model_tier": "mid",   # extra-low | low | mid | high
            # "mcp_servers": [{"name": "data-provider", "url": "http://127.0.0.1:8765", "tools": []}],
        }},
        stream_mode=["messages", "updates", "custom"],
        stream_subgraphs=True,  # so subagent events surface too
    )


def render_event(d: dict, pending: list[str]) -> None:
    """One protocol event → one or more printed lines (``pending`` collects clarifying
    questions). The end-state ``status`` is handled by the caller."""
    t = d.get("type")
    if t == "run_start":
        got = d.get("protocol_version")
        note = "" if got == EXPECTED_PROTOCOL else f"  ⚠ expected v{EXPECTED_PROTOCOL}"
        print(f"[protocol v{got} · engine {d.get('engine_version')}]{note}")
    elif t == "clarification":
        pending[:] = [q for q in d.get("questions", []) if q]
        print("\n❓ Clarifying questions:")
        for i, q in enumerate(pending, 1):
            print(f"  {i}. {q}")
    elif t == "search_query":
        print(f"  🔎 {d['query']}")
    elif t == "search_results":
        for r in d.get("results", []):
            print(f"     • {r['domain']:<22} {r['title'][:60]}")
    elif t == "source":
        print(f"     ↳ source: {d['title'][:60]} — {d['url']}")
    # mcp_* and tool_* carry the same shape; the split is only which layer emitted.
    elif t in ("mcp_call", "tool_call"):
        print(f"  🛠  {d['tool']}({d.get('args')})")
    elif t in ("mcp_result", "tool_result"):
        mark = "✓" if d.get("ok") else f"✗ {d.get('error_class', 'error')}"
        print(f"     {mark} {d['tool']}")
    elif t == "skill":
        print(f"  📄 skill {d['name']} [{d['state']}]")
    elif t == "subagent_findings":
        print(f"\n  ── {d['unit']}: {d['summary']}")
        for f in d.get("findings", []):
            print(f"     • {f}")
        for g in d.get("gaps", []):
            print(f"     ? gap: {g}")
    elif t == "report":
        print("\n===== FINAL REPORT =====\n")
        print(d["markdown"])
    elif t == "usage":
        print(f"\n[usage] {d['tool_calls']} tool calls · "
              f"{d['total_tokens']} tokens · run time {d.get('elapsed', 'n/a')} · "
              f"limits {d['limits']}")
    elif t == "status":
        print(f"  [status] {d}")
    else:
        # Additions are compatible by design (only removals/renames bump the
        # version), so an unknown type is informational — never fatal.
        print(f"  [unknown event: {t}]")


def collect_answers(questions: list[str]) -> str:
    """Prompt for each answer and pair it back with its question. Sending the bare
    answers ("the first") loses meaning — the agent can't tell which option "the first"
    refers to — so the reply restates every question alongside its answer."""
    lines = ["Answers to your clarifying questions:"]
    for i, q in enumerate(questions, 1):
        ans = input(f"  {i}. {q}\n     > ").strip()
        lines.append(f"{i}. Q: {q}\n   A: {ans or '(no preference)'}")
    return "\n".join(lines)


async def main(question: str) -> None:
    client = get_client(url=BASE_URL)
    try:
        thread = await client.threads.create()
    except Exception as exc:  # httpx raises several distinct connect errors; all mean the same
        print(f"cannot reach the agent at {BASE_URL}: {exc}\n"
              "start it with ./run.sh (or ./run-stack.sh, which also starts the sandbox)",
              file=sys.stderr)
        raise SystemExit(1) from None

    content = question
    while True:
        # Re-running on the SAME thread_id appends to the persisted message history,
        # so the agent resumes with the full prior context (questions + answers).
        pending = await stream_turn(client, thread["thread_id"], content)
        if not pending:
            break
        content = collect_answers(pending)


if __name__ == "__main__":
    asyncio.run(main(read_question(sys.argv)))
