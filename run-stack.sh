#!/usr/bin/env bash
#
# Bring up the FULL local stack: the llm-sandbox service + this agent, wired together.
#
#   ./run-stack.sh                start sandbox (background) + LangGraph server (foreground)
#   ./run-stack.sh split          same stack, but a live 2-column log view in tmux:
#                                 agent (left) | sandbox (right) — see ./run-stack-split.sh
#   ./run-stack.sh ask "<q>"      start the whole stack, stream one research run, tear down
#   ./run-stack.sh smoke          same with a canned question
#   ./run-stack.sh doctor         check both halves without starting anything
#   ./run-stack.sh help           print this text
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

# The sandbox repo lives NEXT TO the main checkout. Resolve the default from git's
# common dir so it also works from a linked worktree (…/.worktrees/<name>), where
# this script's own ../ would point inside .worktrees. Outside a git repo the
# fallback keeps the old script-relative behavior.
MAIN_CHECKOUT="$(dirname "$(git rev-parse --git-common-dir 2>/dev/null || echo .git)")"
SANDBOX_REPO="${LLM_SANDBOX_REPO:-$MAIN_CHECKOUT/../llm_sandbox}"
SANDBOX_URL="${LLM_SANDBOX_URL:-http://127.0.0.1:8900}"
AGENT_HOST="${DRA_HOST:-127.0.0.1}"
AGENT_PORT="${DRA_PORT:-${PORT:-2024}}"
AGENT_URL="http://${AGENT_HOST}:${AGENT_PORT}"
LOGDIR="$(mktemp -d "${TMPDIR:-/tmp}/llm-stack.XXXXXX")"

SANDBOX_PID=""
AGENT_PID=""
TMUX_SOCKET=""

die() { echo "error: $*" >&2; exit 1; }
say() { echo "▶ $*"; }

cleanup() {
  local code=$?
  # The split view runs on its own tmux server (private socket) so it can't collide with —
  # or take down — the user's real tmux. Kill that server, never the default one.
  [ -n "$TMUX_SOCKET" ] && tmux -L "$TMUX_SOCKET" kill-server 2>/dev/null || true
  [ -n "$AGENT_PID" ] && kill "$AGENT_PID" 2>/dev/null || true
  if [ -n "$SANDBOX_PID" ]; then
    say "stopping sandbox service…"
    # `uv run` is a supervisor: killing it usually takes uvicorn with it, but not always.
    # A survivor would hold :8900 and silently serve the NEXT run. Record its children
    # BEFORE the kill (they get reparented after) and kill those exact PIDs — a
    # machine-wide `pkill -f uvicorn...` here could take down a SECOND stack's sandbox.
    kids="$(pgrep -P "$SANDBOX_PID" 2>/dev/null || true)"
    kill "$SANDBOX_PID" 2>/dev/null || true
    wait "$SANDBOX_PID" 2>/dev/null || true
    for pid in $kids; do kill "$pid" 2>/dev/null || true; done
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
  # ${…} braces matter: some bash builds parse a multibyte char glued to a bare $VAR into
  # the variable name itself, and set -u then aborts on the "unbound" mangled name.
  say "starting LangGraph server on ${AGENT_URL}…"
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

# Live 2-column log view: agent (left) | sandbox (right), each pane scrolling on its own.
# Side-by-side independently-scrolling regions are exactly what a terminal can't do with
# plain ANSI (scroll regions are full-width rows), so this needs a multiplexer.
view_split() {
  command -v tmux >/dev/null || die "split view needs tmux (brew install tmux)"
  # start_sandbox early-returns when it finds a sandbox to reuse — then nobody writes
  # sandbox.log and tail -F would sit on a missing file forever. Say so in the pane instead.
  [ -f "$LOGDIR/sandbox.log" ] || \
    echo "(sandbox was already running — its logs are in the shell that started it)" >"$LOGDIR/sandbox.log"

  TMUX_SOCKET="llm-stack-$$"
  # TMUX= so this also works from inside an existing tmux session (nested-attach guard keys
  # off that variable); -f /dev/null so the user's tmux.conf can't restyle or rebind the view.
  TMUX='' tmux -L "$TMUX_SOCKET" -f /dev/null new-session -d -s stack \
    "exec tail -n +1 -F '$LOGDIR/agent.log'"
  tmux -L "$TMUX_SOCKET" split-window -h -t stack:0 "exec tail -n +1 -F '$LOGDIR/sandbox.log'"
  tmux -L "$TMUX_SOCKET" set-option -g mouse on
  tmux -L "$TMUX_SOCKET" set-option -g pane-border-status top
  tmux -L "$TMUX_SOCKET" select-pane -t stack:0.0 -T "agent — $AGENT_URL"
  tmux -L "$TMUX_SOCKET" select-pane -t stack:0.1 -T "sandbox — $SANDBOX_URL"
  tmux -L "$TMUX_SOCKET" set-option -g status-left-length 80
  tmux -L "$TMUX_SOCKET" set-option -g status-left " Ctrl-b d quits view AND stops the stack "
  tmux -L "$TMUX_SOCKET" set-option -g status-right " logs: $LOGDIR "
  tmux -L "$TMUX_SOCKET" set-option -g status-right-length 120
  say "split view: agent (left) | sandbox (right) — Ctrl-b d (or Ctrl-C in both panes) stops the stack"
  TMUX='' tmux -L "$TMUX_SOCKET" attach -t stack
  # attach returned = user detached or closed both panes; falling off main exits the
  # script, and the EXIT trap tears the whole stack down.
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

  split)
    resolve_sandbox; check_token; start_sandbox; start_agent_bg
    view_split
    ;;

  ask|smoke)
    resolve_sandbox; check_token; start_sandbox; start_agent_bg
    echo
    ./run.sh "$@"
    ;;

  help|-h|--help)
    # The header comment IS the manual — print it rather than maintain a second copy.
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
    ;;

  *)
    die 'usage: ./run-stack.sh [up|split|ask "<question>"|smoke|doctor|help]'
    ;;
esac
