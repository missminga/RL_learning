#!/usr/bin/env python3
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / ".clawdbot" / "active-tasks.json"


def sh(cmd, cwd=ROOT):
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def load():
    if not REG.exists():
        return []
    try:
        return json.loads(REG.read_text(encoding="utf-8"))
    except Exception:
        return []


def save(data):
    REG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tmux_alive(session: str) -> bool:
    code, _, _ = sh(["tmux", "has-session", "-t", session])
    return code == 0


def pr_for_branch(branch: str):
    code, out, _ = sh(["gh", "pr", "list", "--head", branch, "--state", "all", "--json", "number,state,isDraft,url,statusCheckRollup"])
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


def main():
    data = load()
    now = int(time.time() * 1000)
    changed = False

    running = [t for t in data if t.get("status") in ("running", "reviewing")]
    if not running:
        print("No running tasks.")
        return

    for t in running:
        tid = t.get("id", "")
        session = t.get("tmuxSession", "")
        branch = t.get("branch", "")
        status = t.get("status", "running")
        note = ""

        if session and not tmux_alive(session):
            status = "attention"
            note = "tmux session exited"

        pr = pr_for_branch(branch) if branch else None
        if pr:
            t["pr"] = pr.get("number")
            t["prUrl"] = pr.get("url", "")
            if checks_passed(pr):
                status = "done"
                t["completedAt"] = now
                note = f"PR #{pr.get('number')} checks passed"
            else:
                status = "reviewing"
                note = f"PR #{pr.get('number')} waiting for checks"

        t["status"] = status
        t["lastCheckedAt"] = now
        if note:
            t["note"] = note
        changed = True
        print(f"[{status}] {tid} {('- ' + note) if note else ''}")

    if changed:
        save(data)


if __name__ == "__main__":
    main()
