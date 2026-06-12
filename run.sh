#!/usr/bin/env bash
#
# One-command dev bring-up for the deep-research-agent (LangGraph) service.
#   ./run.sh                 sync deps, then start the LangGraph dev server on :2024
#   ./run.sh ask "<question>" stream one research run against a RUNNING server
#                            (long prompt? use a file: ./run.sh ask @prompt.txt)
#   ./run.sh smoke           ask a canned question against a RUNNING server
#   ./run.sh test            run the offline pytest suite (no API keys / network)
#   ./run.sh --sync          force `uv sync --extra dev`, then start the server
#
# The agent speaks the LangGraph HTTP/SSE API. Start the server (default `up`),
# then point `ask`/`smoke` (or any LangGraph SDK client) at it. Config comes from
# ./.env (OPENAI_API_KEY, TAVILY_API_KEY, DRA_* models, optional MCP / sandbox).
set -euo pipefail
cd "$(dirname "$0")"

# Load .env so this script and the server see the same config (the server also loads it).
if [ -f .env ]; then set -a; . ./.env || true; set +a; fi

HOST="${DRA_HOST:-127.0.0.1}"
PORT="${PORT:-2024}"
BASE="http://${HOST}:${PORT}"

die() { echo "error: $*" >&2; exit 1; }
need_uv() { command -v uv >/dev/null || die "uv not found (https://docs.astral.sh/uv/)"; }

case "${1:-up}" in
  ask)
    need_uv
    shift
    [ "$#" -ge 1 ] || die 'usage: ./run.sh ask "<question>" | ./run.sh ask @prompt.txt'
    echo "▶ streaming run against ${BASE} (server must be up — ./run.sh)…"
    exec uv run python examples/client.py "$*"
    ;;
  smoke)
    need_uv
    echo "▶ smoke run against ${BASE} (server must be up — ./run.sh)…"
    exec uv run python examples/client.py "What are the recent trends across the tracked entities, and where can I find supporting data?"
    ;;
  test)
    need_uv
    uv sync --extra dev
    exec uv run pytest tests/ -q
    ;;
  up|--sync)
    need_uv
    if [ "${1:-up}" = "--sync" ] || [ ! -d .venv ]; then
      echo "▶ uv sync --extra dev (deepagents + LangGraph + CLI into ./.venv)…"
      uv sync --extra dev
    else
      echo "▶ ./.venv present (./run.sh --sync to re-sync)"
    fi
    [ -n "${OPENAI_API_KEY:-}" ] || echo "warning: OPENAI_API_KEY unset in .env — runs will fail"
    [ -n "${TAVILY_API_KEY:-}" ] || echo "warning: TAVILY_API_KEY unset in .env — web search disabled"
    echo "▶ starting LangGraph dev server on ${BASE} (docs: ${BASE}/docs)…"
    exec uv run langgraph dev --host "$HOST" --port "$PORT"
    ;;
  *)
    die 'usage: ./run.sh [ask "<question>"|smoke|test|--sync]'
    ;;
esac
