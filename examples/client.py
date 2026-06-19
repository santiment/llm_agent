"""Minimal consumer — any app talks to the agent like this. No package import
needed; it's just HTTP/SSE against the LangGraph server.

    python examples/client.py "What are the recent trends...?"   # inline
    python examples/client.py @prompt.txt                        # read a file (long prompts)
    python examples/client.py -                                  # read STDIN
    cat prompt.txt | python examples/client.py                   # piped STDIN

Shows the live event protocol: phase / search / mcp / sources, assistant
thinking tokens, and the final report. Mirrors what your frontend renders.
"""

import asyncio
import sys
from pathlib import Path

from langgraph_sdk import get_client

_DEFAULT_Q = "Give me a deep research report on BDCs."


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
    async for chunk in client.runs.stream(
        thread_id,
        "deep_research_agent",
        input={"messages": [{"role": "user", "content": content}]},
        # per-run overrides; omit to use the server's .env defaults
        config={"configurable": {
            "research_model": "openai/gpt-4o",
            "final_report_model": "anthropic/claude-sonnet-4-6",
            # "mcp_servers": [{"name": "data-provider", "url": "http://127.0.0.1:8765", "tools": []}],
        }},
        stream_mode=["messages", "updates", "custom"],
        stream_subgraphs=True,  # so subagent events surface too
    ):
        if chunk.event == "custom":
            d = chunk.data
            t = d.get("type")
            if t == "clarification":
                pending = [q for q in d.get("questions", []) if q]
                print("\n❓ Clarifying questions:")
                for i, q in enumerate(pending, 1):
                    print(f"  {i}. {q}")
            elif t == "phase":
                print(f"\n### {d.get('title')} [{d.get('status')}]")
            elif t == "search_query":
                print(f"  🔎 {d['query']}")
            elif t == "search_results":
                for r in d.get("results", []):
                    print(f"     • {r['domain']:<22} {r['title'][:60]}")
            elif t in ("mcp_call",):
                print(f"  🛠  {d['tool']}({d.get('args')})")
            elif t == "report":
                print("\n===== FINAL REPORT =====\n")
                print(d["markdown"])
            elif t == "status":
                print(f"  [status] {d}")
        elif chunk.event == "messages":
            # streamed assistant thinking tokens
            for m in chunk.data:
                content = m.get("content") if isinstance(m, dict) else None
                if isinstance(content, str):
                    print(content, end="", flush=True)
    return pending


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
    client = get_client(url="http://127.0.0.1:2024")
    thread = await client.threads.create()

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
