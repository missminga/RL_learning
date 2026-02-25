#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REG="$ROOT_DIR/.clawdbot/active-tasks.json"
WORKTREES="$ROOT_DIR/.clawdbot/worktrees"

if [[ ! -f "$REG" ]]; then
  echo "No registry found"
  exit 0
fi

mapfile -t done_ids < <(python3 - <<PY
import json
from pathlib import Path
arr=json.loads(Path('$REG').read_text(encoding='utf-8'))
for t in arr:
    if t.get('status') in ('done','closed','cancelled'):
        print(t.get('id',''))
PY
)

for tid in "${done_ids[@]}"; do
  [[ -z "$tid" ]] && continue
  wt="$WORKTREES/$tid"
  if [[ -d "$wt" ]]; then
    echo "Removing worktree: $wt"
    git -C "$ROOT_DIR" worktree remove "$wt" --force || true
  fi
done

echo "cleanup done"
