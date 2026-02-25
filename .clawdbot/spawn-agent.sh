#!/usr/bin/env bash
set -euo pipefail

# Usage:
# .clawdbot/spawn-agent.sh <task-id> <agent:codex|claude> "<description>" "<prompt text or @prompt-file>" [model] [effort]

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLAW_DIR="$ROOT_DIR/.clawdbot"
WORKTREES_DIR="$CLAW_DIR/worktrees"
LOG_DIR="$CLAW_DIR/logs"
REG_PY="$CLAW_DIR/task_registry.py"

TASK_ID="${1:-}"
AGENT="${2:-}"
DESC="${3:-}"
PROMPT_IN="${4:-}"
MODEL="${5:-}"
EFFORT="${6:-high}"

if [[ -z "$TASK_ID" || -z "$AGENT" || -z "$DESC" || -z "$PROMPT_IN" ]]; then
  echo "Usage: .clawdbot/spawn-agent.sh <task-id> <agent:codex|claude> \"<description>\" \"<prompt text or @prompt-file>\" [model] [effort]"
  exit 1
fi

if [[ "$AGENT" != "codex" && "$AGENT" != "claude" ]]; then
  echo "agent must be codex or claude"
  exit 1
fi

mkdir -p "$WORKTREES_DIR" "$LOG_DIR"

BRANCH="feat/${TASK_ID}"
WORKTREE="$WORKTREES_DIR/$TASK_ID"
SESSION="${AGENT}-${TASK_ID}"
SESSION="${SESSION//\//-}"
SESSION="${SESSION//_/-}"
SESSION="${SESSION:0:50}"
LOG_FILE="$LOG_DIR/${TASK_ID}.log"

if [[ -d "$WORKTREE/.git" ]]; then
  echo "Worktree already exists: $WORKTREE"
else
  git -C "$ROOT_DIR" fetch origin main >/dev/null 2>&1 || true
  git -C "$ROOT_DIR" worktree add "$WORKTREE" -b "$BRANCH" origin/main 2>/dev/null || \
  git -C "$ROOT_DIR" worktree add "$WORKTREE" -b "$BRANCH" HEAD
fi

if [[ "$PROMPT_IN" == @* ]]; then
  PROMPT_FILE="${PROMPT_IN#@}"
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Prompt file not found: $PROMPT_FILE"
    exit 1
  fi
  PROMPT="$(cat "$PROMPT_FILE")"
else
  PROMPT="$PROMPT_IN"
fi

if [[ -z "$MODEL" ]]; then
  if [[ "$AGENT" == "codex" ]]; then
    MODEL="gpt-5.3-codex"
  else
    MODEL="claude-opus-4.5"
  fi
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session exists, reusing: $SESSION"
else
  PROMPT_Q="$(printf '%q' "$PROMPT")"
  if [[ "$AGENT" == "codex" ]]; then
    CMD="codex --model $MODEL -c model_reasoning_effort=$EFFORT --dangerously-bypass-approvals-and-sandbox $PROMPT_Q"
  else
    CMD="claude --model $MODEL --dangerously-skip-permissions -p $PROMPT_Q"
  fi

  tmux new-session -d -s "$SESSION" -c "$WORKTREE" "$CMD"
  tmux pipe-pane -o -t "$SESSION" "cat >> '$LOG_FILE'"
fi

STARTED_AT="$(date +%s000)"

python3 "$REG_PY" add <<JSON
{
  "id": "$TASK_ID",
  "tmuxSession": "$SESSION",
  "agent": "$AGENT",
  "description": "$DESC",
  "repo": "$(basename "$ROOT_DIR")",
  "worktree": "$WORKTREE",
  "branch": "$BRANCH",
  "logFile": "$LOG_FILE",
  "startedAt": $STARTED_AT,
  "status": "running",
  "retries": 0,
  "notifyOnComplete": true,
  "model": "$MODEL",
  "effort": "$EFFORT"
}
JSON

echo "spawned task=$TASK_ID agent=$AGENT session=$SESSION"
echo "worktree: $WORKTREE"
echo "log: $LOG_FILE"
