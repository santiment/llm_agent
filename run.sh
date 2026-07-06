#!/usr/bin/env bash
#
# One-command dev bring-up for the deep-research-agent (LangGraph) service.
#   ./run.sh                 sync deps, then start the LangGraph dev server on :2024
#   ./run.sh ask "<question>" stream one research run against a RUNNING server
#                            (long prompt? use a file: ./run.sh ask @prompt.txt)
#   ./run.sh smoke           ask a canned question against a RUNNING server
#   ./run.sh doctor          check config/deps/reachability without starting anything
#   ./run.sh test            run the offline pytest suite (no API keys / network)
#   ./run.sh --sync          force `uv sync --extra dev`, then start the server
#
# The agent speaks the LangGraph HTTP/SSE API. Start the server (default `up`),
# then point `ask`/`smoke` (or any LangGraph SDK client) at it. Config comes from
# ./.env (OPENAI_API_KEY, TAVILY_API_KEY, DRA_* models, optional MCP / sandbox).
#
# To start this agent AND the llm-sandbox service together, use ./run-stack.sh.
set -euo pipefail
cd "$(dirname "$0")"

# Load .env so this script and the server see the same config (the server also loads it).
if [ -f .env ]; then set -a; . ./.env || true; set +a; fi

HOST="${DRA_HOST:-127.0.0.1}"
PORT="${PORT:-2024}"
BASE="http://${HOST}:${PORT}"

die() { echo "error: $*" >&2; exit 1; }
need_uv() { command -v uv >/dev/null || die "uv not found (https://docs.astral.sh/uv/)"; }

# Any HTTP response means the server is accepting connections — that's all `ask` needs, and
# it doesn't couple us to whichever health path this LangGraph version happens to ship.
server_up() { [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "${BASE}/ok" 2>/dev/null)" != "000" ]; }

# `ask` against a dead server used to surface as a raw httpx traceback from the SDK. Say what
# is actually wrong instead.
need_server() {
  server_up || die "no server at ${BASE} — start one first (./run.sh) or use ./run-stack.sh ask ..."
}

case "${1:-up}" in
  ask)
    need_uv; need_server
    shift
    [ "$#" -ge 1 ] || die 'usage: ./run.sh ask "<question>" | ./run.sh ask @prompt.txt'
    echo "▶ streaming run against ${BASE}…"
    exec uv run python examples/client.py "$*"
    ;;
  smoke)
    need_uv; need_server
    echo "▶ smoke run against ${BASE}…"
    exec uv run python examples/client.py "What are the recent trends across the tracked entities, and where can I find supporting data?"
    ;;
  doctor)
    echo "── deep-research-agent doctor ───────────────────────────────────"
    command -v uv >/dev/null && echo "✓ uv           $(uv --version)" || echo "✗ uv missing — https://docs.astral.sh/uv/"
    [ -d .venv ] && echo "✓ ./.venv      present" || echo "  ./.venv absent — './run.sh --sync' creates it"
    [ -f .env ]  && echo "✓ .env         present" || echo "✗ .env missing — cp .env.example .env"
    [ -n "${OPENAI_API_KEY:-}" ] && echo "✓ OPENAI_API_KEY set" || echo "✗ OPENAI_API_KEY unset — runs will fail"
    [ -n "${TAVILY_API_KEY:-}" ] && echo "✓ TAVILY_API_KEY set" || echo "  TAVILY_API_KEY unset — web search disabled"
    echo
    echo "  server        ${BASE} $(server_up && echo '(up)' || echo '(not running — ./run.sh)')"
    # The execute tool is opt-in: unset URL means the agent silently runs code in-process
    # instead, which is a very different security story than "sandboxed".
    if [ -z "${LLM_SANDBOX_URL:-}" ]; then
      echo "  sandbox       LLM_SANDBOX_URL unset — code execution DISABLED (in-memory fallback)"
    elif curl -fsS --max-time 3 "${LLM_SANDBOX_URL}/healthz" >/dev/null 2>&1; then
      echo "✓ sandbox      ${LLM_SANDBOX_URL} reachable — execute tool ON"
      echo "                run '\$LLM_SANDBOX_REPO/run.sh doctor' to see if it actually isolates"
    else
      echo "✗ sandbox      ${LLM_SANDBOX_URL} unreachable — every execute call will fail"
      echo "                start it, or use ./run-stack.sh which starts both"
    fi
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
    die 'usage: ./run.sh [ask "<question>"|smoke|doctor|test|--sync]'
    ;;
esac
