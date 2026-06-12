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


async def main(question: str) -> None:
    client = get_client(url="http://127.0.0.1:2024")
    thread = await client.threads.create()

    async for chunk in client.runs.stream(
        thread["thread_id"],
        "deep_research_agent",
        input={"messages": [{"role": "user", "content": question}]},
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
            if t == "phase":
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


if __name__ == "__main__":
    asyncio.run(main(read_question(sys.argv)))
