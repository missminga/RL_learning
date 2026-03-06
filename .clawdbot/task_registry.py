#!/usr/bin/env python3
"""
task_registry.py — Thin shim over state_store.py (DB-backed).
Keeps the original CLI interface: add / update / list / get / touch
Falls back to JSON if DB not available.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_STORE = ROOT / "state" / "state_store.py"
DB_AVAILABLE = STATE_STORE.exists()

# ── Legacy JSON fallback ──────────────────────────────────────────────────────
LEGACY = Path(__file__).resolve().parent / "active-tasks.json"


def now_ms():
    return int(time.time() * 1000)


def _json_load():
    if not LEGACY.exists():
        return []
    try:
        return json.loads(LEGACY.read_text(encoding="utf-8"))
    except Exception:
        return []


def _json_save(data):
    LEGACY.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _db(args, stdin=None):
    r = subprocess.run(
        ["python3", str(STATE_STORE)] + args,
        input=stdin, cwd=ROOT, text=True, capture_output=True
    )
    if r.returncode != 0:
        raise SystemExit(f"state_store error: {r.stderr.strip()}")
    return r.stdout.strip()


# ── CLI ───────────────────────────────────────────────────────────────────────

cmd = sys.argv[1] if len(sys.argv) > 1 else ""

if cmd == "add":
    raw = sys.stdin.read().strip()
    item = json.loads(raw)
    if "id" not in item and "task_id" not in item:
        raise SystemExit("missing id")
    if DB_AVAILABLE:
        out = _db(["add"], stdin=raw)
        print(out)
    else:
        data = _json_load()
        tid = item.get("id") or item.get("task_id")
        data = [x for x in data if x.get("id") != tid]
        data.append(item)
        _json_save(data)
        print(json.dumps({"ok": True, "action": "add", "id": tid}))

elif cmd == "update":
    tid = sys.argv[2]
    patch = json.loads(sys.stdin.read().strip() or "{}")
    if DB_AVAILABLE:
        out = _db(["update", tid], stdin=json.dumps(patch))
        print(out)
    else:
        data = _json_load()
        found = False
        for x in data:
            if x.get("id") == tid:
                x.update(patch)
                found = True
                break
        if not found:
            raise SystemExit(f"task not found: {tid}")
        _json_save(data)
        print(json.dumps({"ok": True, "action": "update", "id": tid}))

elif cmd == "list":
    if DB_AVAILABLE:
        filters = sys.argv[2:] if len(sys.argv) > 2 else []
        out = _db(["list"] + filters)
        print(out)
    else:
        print(json.dumps(_json_load(), ensure_ascii=False))

elif cmd == "get":
    tid = sys.argv[2]
    if DB_AVAILABLE:
        out = _db(["get", tid])
        print(out)
    else:
        data = _json_load()
        for x in data:
            if x.get("id") == tid:
                print(json.dumps(x, ensure_ascii=False))
                break
        else:
            raise SystemExit(f"task not found: {tid}")

elif cmd == "touch":
    tid = sys.argv[2]
    patch = json.dumps({"lastCheckedAt": now_ms()})
    if DB_AVAILABLE:
        out = _db(["update", tid], stdin=patch)
        print(out)
    else:
        data = _json_load()
        for x in data:
            if x.get("id") == tid:
                x["lastCheckedAt"] = now_ms()
                _json_save(data)
                print(json.dumps({"ok": True, "action": "touch", "id": tid}))
                break
        else:
            raise SystemExit(f"task not found: {tid}")

else:
    raise SystemExit("usage: task_registry.py [add|update|list|get|touch]")
