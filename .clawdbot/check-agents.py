#!/usr/bin/env python3
"""
check-agents.py — Agent task monitor (DB-backed).
Reads task state from SQLite, checks tmux/PR status, updates DB.
Falls back to legacy JSON if DB not available.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_STORE = ROOT / "state" / "state_store.py"
DB_AVAILABLE = STATE_STORE.exists()


def sh(cmd, cwd=ROOT):
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def now_ms():
    return int(time.time() * 1000)


# ── DB interface ──────────────────────────────────────────────────────────────

def db_list_active():
    if not DB_AVAILABLE:
        return _json_load()
    code, out, err = sh(["python3", str(STATE_STORE), "list", "running", "reviewing"])
    if code != 0:
        print(f"[warn] state_store list failed: {err}", file=sys.stderr)
        return _json_load()
    try:
        return json.loads(out)
    except Exception:
        return []


def db_update(task_id, patch):
    if not DB_AVAILABLE:
        return
    patch_json = json.dumps(patch)
    p = subprocess.run(
        ["python3", str(STATE_STORE), "update", task_id],
        input=patch_json, cwd=ROOT, text=True, capture_output=True
    )
    if p.returncode != 0:
        print(f"[warn] state_store update failed: {p.stderr}", file=sys.stderr)


# ── Legacy JSON fallback ──────────────────────────────────────────────────────

LEGACY_REG = ROOT / ".clawdbot" / "active-tasks.json"


def _json_load():
    if not LEGACY_REG.exists():
        return []
    try:
        return json.loads(LEGACY_REG.read_text(encoding="utf-8"))
    except Exception:
        return []


def _json_save(data):
    LEGACY_REG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ── Checks ────────────────────────────────────────────────────────────────────

def tmux_alive(session: str) -> bool:
    code, _, _ = sh(["tmux", "has-session", "-t", session])
    return code == 0


def pr_for_branch(branch: str):
    code, out, _ = sh([
        "gh", "pr", "list", "--head", branch, "--state", "all",
        "--json", "number,state,isDraft,url,statusCheckRollup"
    ])
    if code != 0 or not out:
        return None
    try:
        arr = json.loads(out)
        return arr[0] if arr else None
    except Exception:
        return None


def checks_passed(pr) -> bool:
    rollup = pr.get("statusCheckRollup") or []
    for c in rollup:
        if c.get("status") != "COMPLETED":
            return False
        if c.get("conclusion") not in ("SUCCESS", "NEUTRAL", "SKIPPED"):
            return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tasks = db_list_active()

    running = [t for t in tasks if t.get("status") in ("running", "reviewing")]
    if not running:
        print("No running tasks.")
        return

    n = now_ms()
    legacy_changed = False
    legacy_data = _json_load() if not DB_AVAILABLE else []

    for t in running:
        tid = t.get("id") or t.get("task_id", "")
        session = t.get("tmuxSession") or t.get("tmux_session", "")
        branch = t.get("branch", "")
        status = t.get("status", "running")
        patch = {"last_checked_at": n}
        note = ""

        if session and not tmux_alive(session):
            status = "attention"
            note = "tmux session exited"

        pr = pr_for_branch(branch) if branch else None
        if pr:
            patch["pr_number"] = pr.get("number")
            patch["pr_url"] = pr.get("url", "")
            if checks_passed(pr):
                status = "done"
                patch["completed_at"] = n
                note = f"PR #{pr.get('number')} checks passed"
            else:
                status = "reviewing"
                note = f"PR #{pr.get('number')} waiting for checks"

        patch["status"] = status
        if note:
            patch["note"] = note

        print(f"[{status}] {tid}{(' - ' + note) if note else ''}")

        if DB_AVAILABLE:
            db_update(tid, patch)
        else:
            # Legacy JSON update
            for item in legacy_data:
                if item.get("id") == tid:
                    item.update({
                        "status": status,
                        "lastCheckedAt": n,
                        **({"note": note} if note else {}),
                        **({"completedAt": n} if status == "done" else {}),
                        **({"pr": patch.get("pr_number")} if "pr_number" in patch else {}),
                        **({"prUrl": patch.get("pr_url")} if "pr_url" in patch else {}),
                    })
                    legacy_changed = True

    if not DB_AVAILABLE and legacy_changed:
        _json_save(legacy_data)


if __name__ == "__main__":
    main()
