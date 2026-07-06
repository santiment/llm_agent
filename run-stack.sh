#!/usr/bin/env bash
#
# Bring up the FULL local stack: the llm-sandbox service + this agent, wired together.
#
#   ./run-stack.sh                start sandbox (background) + LangGraph server (foreground)
#   ./run-stack.sh ask "<q>"      start the whole stack, stream one research run, tear down
#   ./run-stack.sh smoke          same with a canned question
#   ./run-stack.sh doctor         check both halves without starting anything
#
# How a run becomes a container:
#   agent run -> HttpSandboxBackend creates a sandbox SESSION on first execute/file op
#             -> llm-sandbox starts ONE container for that session (persistent /workspace)
#             -> every execute/read/write in the run goes to that same container
#             -> SandboxCleanupMiddleware DELETEs the session when the run ends
# Same shape as prod; on k8s a session is a gVisor pod instead of a container.
#
# ISOLATION: the sandbox repo's run.sh decides that, and refuses to start under a runtime the
# daemon does not have. If it warns that sessions run under `runc`, they are NOT isolated —
# fine for developing, never for code you would not run on your laptop anyway.
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] && { set -a; . ./.env || true; set +a; }

SANDBOX_REPO="${LLM_SANDBOX_REPO:-../llm_sandbox}"
SANDBOX_URL="${LLM_SANDBOX_URL:-http://127.0.0.1:8900}"
AGENT_HOST="${DRA_HOST:-127.0.0.1}"
AGENT_PORT="${PORT:-2024}"
AGENT_URL="http://${AGENT_HOST}:${AGENT_PORT}"
LOGDIR="$(mktemp -d "${TMPDIR:-/tmp}/llm-stack.XXXXXX")"

SANDBOX_PID=""
AGENT_PID=""

die() { echo "error: $*" >&2; exit 1; }
say() { echo "▶ $*"; }

cleanup() {
  local code=$?
  [ -n "$AGENT_PID" ] && kill "$AGENT_PID" 2>/dev/null || true
  if [ -n "$SANDBOX_PID" ]; then
    say "stopping sandbox service…"
    kill "$SANDBOX_PID" 2>/dev/null || true
    wait "$SANDBOX_PID" 2>/dev/null || true
    # `uv run` is a supervisor: killing it usually takes uvicorn with it, but not always.
    # A survivor would hold :8900 and silently serve the NEXT run, so match it exactly.
    pkill -f 'uvicorn llm_sandbox.app:app' 2>/dev/null || true
    # Sessions are auto-reaped by their own `sleep <timeout>`, but a killed service leaves
    # them running until then — drop them now so the next run starts clean.
    "$SANDBOX_REPO/run.sh" clean >/dev/null 2>&1 || true
  fi
  [ "$code" -ne 0 ] && echo "logs kept in $LOGDIR" >&2
  exit "$code"
}
trap cleanup EXIT INT TERM

resolve_sandbox() {
  [ -d "$SANDBOX_REPO" ] || die "sandbox repo not found at '$SANDBOX_REPO' (set LLM_SANDBOX_REPO=/path/to/llm_sandbox)"
  [ -x "$SANDBOX_REPO/run.sh" ] || die "'$SANDBOX_REPO/run.sh' missing or not executable"
  SANDBOX_REPO="$(cd "$SANDBOX_REPO" && pwd)"
}

# A token mismatch shows up as a 401 on the agent's FIRST tool call, deep inside a run that
# has already burned model spend. Catch it before anything starts.
# $1=soft -> report and keep going (doctor); otherwise fatal.
check_token() {
  local theirs
  theirs="$(grep -sE '^LLM_SANDBOX_TOKEN=' "$SANDBOX_REPO/.env" | tail -1 | cut -d= -f2- | tr -d '"'"'"' ')"
  if [ "${LLM_SANDBOX_TOKEN:-}" = "$theirs" ]; then
    echo "✓ sandbox token matches"
    return 0
  fi
  echo "✗ token mismatch — the agent would get 401 on its first sandbox call:" >&2
  echo "    agent   LLM_SANDBOX_TOKEN=${LLM_SANDBOX_TOKEN:-<unset>}   (./.env)" >&2
  echo "    sandbox LLM_SANDBOX_TOKEN=${theirs:-<unset>}   ($SANDBOX_REPO/.env)" >&2
  [ "${1:-}" = "soft" ] && return 0
  die "make them equal (both empty also works — that disables auth, dev only)"
}

sandbox_up() { curl -fsS "$SANDBOX_URL/healthz" >/dev/null 2>&1; }

start_sandbox() {
  if sandbox_up; then
    say "sandbox already running at $SANDBOX_URL — reusing it"
    return
  fi
  say "starting sandbox service ($SANDBOX_REPO)…"
  # No subshell: run.sh cd's to its own dir and `exec`s the server, so $! is the server
  # itself. Wrapping it in `( cd … && … )` would leave the server orphaned on kill.
  "$SANDBOX_REPO/run.sh" up >"$LOGDIR/sandbox.log" 2>&1 &
  SANDBOX_PID=$!
  for _ in $(seq 1 120); do          # first run builds the runtime image; be patient
    sandbox_up && break
    kill -0 "$SANDBOX_PID" 2>/dev/null || { sed 's/^/  | /' "$LOGDIR/sandbox.log" >&2; die "sandbox service exited during startup"; }
    sleep 1
  done
  sandbox_up || { sed 's/^/  | /' "$LOGDIR/sandbox.log" >&2; die "sandbox did not become healthy at $SANDBOX_URL"; }
  # Replay the sandbox's isolation verdict here — redirected to a log file it would otherwise
  # be invisible, and "am I actually sandboxed?" is the one thing nobody should have to dig for.
  grep -B2 -A5 'WARNING — NO ISOLATION' "$LOGDIR/sandbox.log" || true
  say "sandbox healthy at $SANDBOX_URL (log: $LOGDIR/sandbox.log)"
}

start_agent_bg() {
  say "starting LangGraph server on $AGENT_URL…"
  ./run.sh up >"$LOGDIR/agent.log" 2>&1 &
  AGENT_PID=$!
  for _ in $(seq 1 120); do
    # No -f on purpose: any HTTP response means the server is accepting connections, which is
    # all `ask` needs. Don't couple bring-up to whichever health path this LangGraph ships.
    curl -s -o /dev/null "$AGENT_URL/ok" 2>/dev/null && break
    kill -0 "$AGENT_PID" 2>/dev/null || { sed 's/^/  | /' "$LOGDIR/agent.log" >&2; die "LangGraph server exited during startup"; }
    sleep 1
  done
  say "agent healthy at $AGENT_URL (log: $LOGDIR/agent.log)"
}

case "${1:-up}" in

  doctor)
    # Each repo owns its own checks; this only adds what neither can see alone — that the
    # two are pointed at each other with a matching token.
    resolve_sandbox
    ./run.sh doctor
    echo
    echo "── sandbox side (repo: $SANDBOX_REPO) ───────────────────────────"
    "$SANDBOX_REPO/run.sh" doctor
    echo
    echo "── wiring ───────────────────────────────────────────────────────"
    check_token soft
    ;;

  up)
    resolve_sandbox; check_token; start_sandbox
    say "starting LangGraph server (Ctrl-C stops the whole stack)…"
    ./run.sh up
    ;;

  ask|smoke)
    resolve_sandbox; check_token; start_sandbox; start_agent_bg
    echo
    ./run.sh "$@"
    ;;

  *)
    die 'usage: ./run-stack.sh [up|ask "<question>"|smoke|doctor]'
    ;;
esac
