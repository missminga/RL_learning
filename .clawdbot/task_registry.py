#!/usr/bin/env python3
import json, sys, time
from pathlib import Path

REG = Path(__file__).resolve().parent / "active-tasks.json"


def load():
    if not REG.exists():
        return []
    try:
        return json.loads(REG.read_text(encoding="utf-8"))
    except Exception:
        return []


def save(data):
    REG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now_ms():
    return int(time.time() * 1000)


cmd = sys.argv[1] if len(sys.argv) > 1 else ""
data = load()

if cmd == "add":
    raw = sys.stdin.read().strip()
    item = json.loads(raw)
    if "id" not in item:
        raise SystemExit("missing id")
    data = [x for x in data if x.get("id") != item["id"]]
    data.append(item)
    save(data)
    print(json.dumps({"ok": True, "action": "add", "id": item["id"]}))
elif cmd == "update":
    tid = sys.argv[2]
    patch = json.loads(sys.stdin.read().strip() or "{}")
    found = False
    for x in data:
        if x.get("id") == tid:
            x.update(patch)
            found = True
            break
    if not found:
        raise SystemExit(f"task not found: {tid}")
    save(data)
    print(json.dumps({"ok": True, "action": "update", "id": tid}))
elif cmd == "list":
    print(json.dumps(data, ensure_ascii=False))
elif cmd == "get":
    tid = sys.argv[2]
    for x in data:
        if x.get("id") == tid:
            print(json.dumps(x, ensure_ascii=False))
            break
    else:
        raise SystemExit(f"task not found: {tid}")
elif cmd == "touch":
    tid = sys.argv[2]
    for x in data:
        if x.get("id") == tid:
            x["lastCheckedAt"] = now_ms()
            save(data)
            print(json.dumps({"ok": True, "action": "touch", "id": tid}))
            break
    else:
        raise SystemExit(f"task not found: {tid}")
else:
    raise SystemExit("usage: task_registry.py [add|update|list|get|touch]")
