"""Web search tool (Tavily) that emits rich events for the live UI.

Emits ``search_query`` then ``search_results`` (title/url/domain/snippet per
hit — the favicon grid in the screenshots) and returns numbered sources so the
model can cite them inline as [n]. Swap Tavily for any search backend by
keeping this event + return-format contract.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from ..config import ResearchConfig
from ..events import domain_of, emit, new_id, source_events


def _normalize(raw: Any) -> list[dict[str, Any]]:
    """langchain-tavily returns {'results': [...]}; older shapes vary."""
    if isinstance(raw, dict):
        results = raw.get("results") or []
    elif isinstance(raw, list):
        results = raw
    else:
        return []
    out = []
    for r in results:
        if not isinstance(r, dict):
            continue
        out.append({
            "title": r.get("title") or r.get("url") or "",
            "url": r.get("url") or "",
            "content": r.get("content") or r.get("snippet") or "",
        })
    return out


def build_search_tool(cfg: ResearchConfig) -> StructuredTool | None:
    if not cfg.tavily_api_key:
        return None

    from langchain_tavily import TavilySearch

    backend = TavilySearch(max_results=cfg.search_max_results, api_key=cfg.tavily_api_key)

    async def web_search(query: str) -> str:
        """Search the web for current information. Returns numbered sources [n] — \
        cite the ones you use inline as [n]."""
        qid = new_id()
        emit({"type": "search_query", "id": qid, "query": query, "source": "web"})
        try:
            raw = await backend.ainvoke({"query": query})
        except Exception as exc:
            emit({"type": "search_results", "id": qid, "query": query,
                  "ok": False, "error": str(exc), "results": []})
            return f"Search failed: {exc}"

        results = _normalize(raw)
        emit({
            "type": "search_results", "id": qid, "query": query, "ok": True,
            "count": len(results),
            "results": [{
                "title": r["title"], "url": r["url"],
                "domain": domain_of(r["url"]), "snippet": r["content"][:280],
            } for r in results],
        })
        source_events(results)

        if not results:
            return "No results found."
        return "\n\n".join(
            f"[{i + 1}] {r['title']} — {r['url']}\n{r['content'][:600]}"
            for i, r in enumerate(results)
        )

    return StructuredTool.from_function(
        coroutine=web_search,
        name="web_search",
        description=("Search the web for current public information. Returns a numbered "
                     "list of sources [n] with snippets; cite the ones you use inline as [n]."),
    )
