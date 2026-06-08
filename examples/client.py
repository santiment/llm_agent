"""Minimal consumer — any app talks to the agent like this. No package import
needed; it's just HTTP/SSE against the LangGraph server.

    python examples/client.py "What are the recent trends across the tracked entities, and where can I find supporting data?"

Shows the live event protocol: phase / search / mcp / sources, assistant
thinking tokens, and the final report. Mirrors what your frontend renders.
"""

import asyncio
import sys

from langgraph_sdk import get_client


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
    q = sys.argv[1] if len(sys.argv) > 1 else "Give me a deep research report on BDCs."
    asyncio.run(main(q))
