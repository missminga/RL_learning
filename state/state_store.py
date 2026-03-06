#!/usr/bin/env python3
"""
state_store.py — SQLite-backed state layer for RL_learning agent orchestration.
Provides a drop-in replacement for the JSON-backed .clawdbot task registry,
plus agent status tracking and a short-term key-value memory store.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).resolve().parent / "agent_state.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Initialize schema (idempotent)."""
    with _connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS agents (
            agent_id        TEXT PRIMARY KEY,
            role            TEXT,
            capability_tags TEXT DEFAULT '[]',
            status          TEXT DEFAULT 'idle',
            last_seen_at    INTEGER,
            meta            TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS tasks (
            task_id             TEXT PRIMARY KEY,
            owner_agent         TEXT,
            status              TEXT DEFAULT 'queued',
            priority            INTEGER DEFAULT 5,
            branch              TEXT,
            tmux_session        TEXT,
            pr_number           INTEGER,
            pr_url              TEXT,
            note                TEXT,
            notify_on_complete  INTEGER DEFAULT 0,
            notified_at         INTEGER,
            last_checked_at     INTEGER,
            completed_at        INTEGER,
            created_at          INTEGER,
            updated_at          INTEGER,
            extra               TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS memory (
            entity      TEXT NOT NULL,
            key         TEXT NOT NULL,
            value_json  TEXT,
            expires_at  INTEGER,
            updated_at  INTEGER,
            PRIMARY KEY (entity, key)
        );

        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT,
            entity_id   TEXT,
            event       TEXT,
            payload     TEXT DEFAULT '{}',
            created_at  INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        CREATE INDEX IF NOT EXISTS idx_events_entity ON events(entity_type, entity_id);
        """)


def now_ms() -> int:
    return int(time.time() * 1000)


# ─── Tasks ───────────────────────────────────────────────────────────────────

def task_add(task: dict) -> None:
    t = now_ms()
    with _connect() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO tasks
            (task_id, owner_agent, status, priority, branch, tmux_session,
             pr_number, pr_url, note, notify_on_complete, notified_at,
             last_checked_at, completed_at, created_at, updated_at, extra)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            task.get("id") or task.get("task_id"),
            task.get("ownerAgent") or task.get("owner_agent"),
            task.get("status", "queued"),
            task.get("priority", 5),
            task.get("branch"),
            task.get("tmuxSession") or task.get("tmux_session"),
            task.get("pr") or task.get("pr_number"),
            task.get("prUrl") or task.get("pr_url"),
            task.get("note"),
            1 if task.get("notifyOnComplete") or task.get("notify_on_complete") else 0,
            task.get("notifiedAt") or task.get("notified_at"),
            task.get("lastCheckedAt") or task.get("last_checked_at"),
            task.get("completedAt") or task.get("completed_at"),
            task.get("createdAt") or task.get("created_at") or t,
            t,
            json.dumps({k: v for k, v in task.items() if k not in {
                "id", "task_id", "ownerAgent", "owner_agent", "status",
                "priority", "branch", "tmuxSession", "tmux_session",
                "pr", "pr_number", "prUrl", "pr_url", "note",
                "notifyOnComplete", "notify_on_complete", "notifiedAt",
                "notified_at", "lastCheckedAt", "last_checked_at",
                "completedAt", "completed_at", "createdAt", "created_at",
            }})
        ))
        _log_event(conn, "task", task.get("id") or task.get("task_id"), "add", task)


def task_update(task_id: str, patch: dict) -> None:
    t = now_ms()
    allowed = {
        "status", "priority", "branch", "tmux_session", "pr_number", "pr_url",
        "note", "notify_on_complete", "notified_at", "last_checked_at",
        "completed_at", "owner_agent", "extra",
    }
    # Accept camelCase aliases
    remap = {
        "tmuxSession": "tmux_session", "prUrl": "pr_url", "prNumber": "pr_number",
        "notifyOnComplete": "notify_on_complete", "notifiedAt": "notified_at",
        "lastCheckedAt": "last_checked_at", "completedAt": "completed_at",
        "ownerAgent": "owner_agent", "pr": "pr_number",
    }
    patch = {remap.get(k, k): v for k, v in patch.items()}
    cols = {k: v for k, v in patch.items() if k in allowed}
    if not cols:
        return
    cols["updated_at"] = t
    set_clause = ", ".join(f"{k} = ?" for k in cols)
    values = list(cols.values()) + [task_id]
    with _connect() as conn:
        conn.execute(f"UPDATE tasks SET {set_clause} WHERE task_id = ?", values)
        _log_event(conn, "task", task_id, "update", patch)


def task_list(status_filter: Optional[list] = None) -> list[dict]:
    with _connect() as conn:
        if status_filter:
            placeholders = ",".join("?" * len(status_filter))
            rows = conn.execute(
                f"SELECT * FROM tasks WHERE status IN ({placeholders}) ORDER BY priority DESC, created_at",
                status_filter,
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tasks ORDER BY priority DESC, created_at").fetchall()
    return [_task_row_to_dict(r) for r in rows]


def task_get(task_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    return _task_row_to_dict(row) if row else None


def _task_row_to_dict(row) -> dict:
    d = dict(row)
    extra = json.loads(d.pop("extra", "{}") or "{}")
    d.update(extra)
    d["notifyOnComplete"] = bool(d.pop("notify_on_complete", 0))
    d["notifiedAt"] = d.pop("notified_at", None)
    d["lastCheckedAt"] = d.pop("last_checked_at", None)
    d["completedAt"] = d.pop("completed_at", None)
    d["createdAt"] = d.pop("created_at", None)
    d["updatedAt"] = d.pop("updated_at", None)
    d["ownerAgent"] = d.pop("owner_agent", None)
    d["tmuxSession"] = d.pop("tmux_session", None)
    d["prUrl"] = d.pop("pr_url", None)
    d["prNumber"] = d.pop("pr_number", None)
    d["id"] = d.pop("task_id")
    return d


# ─── Agents ──────────────────────────────────────────────────────────────────

def agent_upsert(agent_id: str, role: str = "", status: str = "idle",
                 capability_tags: list = None, meta: dict = None) -> None:
    with _connect() as conn:
        conn.execute("""
            INSERT INTO agents (agent_id, role, capability_tags, status, last_seen_at, meta)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(agent_id) DO UPDATE SET
                role = excluded.role,
                status = excluded.status,
                capability_tags = excluded.capability_tags,
                last_seen_at = excluded.last_seen_at,
                meta = excluded.meta
        """, (
            agent_id, role,
            json.dumps(capability_tags or []),
            status, now_ms(),
            json.dumps(meta or {}),
        ))


def agent_list() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM agents ORDER BY agent_id").fetchall()
    return [{**dict(r), "capability_tags": json.loads(r["capability_tags"]),
             "meta": json.loads(r["meta"])} for r in rows]


# ─── Memory ──────────────────────────────────────────────────────────────────

def mem_set(entity: str, key: str, value: Any, ttl_ms: Optional[int] = None) -> None:
    expires = now_ms() + ttl_ms if ttl_ms else None
    with _connect() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO memory (entity, key, value_json, expires_at, updated_at)
            VALUES (?,?,?,?,?)
        """, (entity, key, json.dumps(value), expires, now_ms()))


def mem_get(entity: str, key: str) -> Any:
    with _connect() as conn:
        row = conn.execute(
            "SELECT value_json, expires_at FROM memory WHERE entity=? AND key=?",
            (entity, key)
        ).fetchone()
    if row is None:
        return None
    if row["expires_at"] and now_ms() > row["expires_at"]:
        return None
    return json.loads(row["value_json"])


def mem_del(entity: str, key: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM memory WHERE entity=? AND key=?", (entity, key))


# ─── Events ──────────────────────────────────────────────────────────────────

def _log_event(conn, entity_type: str, entity_id: str, event: str, payload: dict) -> None:
    conn.execute("""
        INSERT INTO events (entity_type, entity_id, event, payload, created_at)
        VALUES (?,?,?,?,?)
    """, (entity_type, entity_id, event, json.dumps(payload, default=str), now_ms()))


def events_recent(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ─── Migration helper ─────────────────────────────────────────────────────────

def migrate_from_json(json_path: Path) -> int:
    """Import existing active-tasks.json into the DB. Returns count of imported tasks."""
    if not json_path.exists():
        return 0
    try:
        tasks = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    count = 0
    for t in tasks:
        if not t.get("id") and not t.get("task_id"):
            continue
        task_add(t)
        count += 1
    return count


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    init_db()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""

    if cmd == "init":
        print("DB initialized at", DB_PATH)

    elif cmd == "migrate":
        json_path = Path(sys.argv[2]) if len(sys.argv) > 2 else \
            Path(__file__).resolve().parent.parent / ".clawdbot" / "active-tasks.json"
        n = migrate_from_json(json_path)
        print(json.dumps({"ok": True, "imported": n}))

    elif cmd == "add":
        raw = sys.stdin.read().strip()
        item = json.loads(raw)
        if not item.get("id") and not item.get("task_id"):
            raise SystemExit("missing id / task_id")
        task_add(item)
        print(json.dumps({"ok": True, "action": "add", "id": item.get("id") or item.get("task_id")}))

    elif cmd == "update":
        tid = sys.argv[2]
        patch = json.loads(sys.stdin.read().strip() or "{}")
        task_update(tid, patch)
        print(json.dumps({"ok": True, "action": "update", "id": tid}))

    elif cmd == "list":
        status_filter = sys.argv[2:] if len(sys.argv) > 2 else None
        print(json.dumps(task_list(status_filter), ensure_ascii=False))

    elif cmd == "get":
        tid = sys.argv[2]
        t = task_get(tid)
        if t:
            print(json.dumps(t, ensure_ascii=False))
        else:
            raise SystemExit(f"task not found: {tid}")

    elif cmd == "mem-set":
        # mem-set <entity> <key> <value_json>
        mem_set(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]))
        print(json.dumps({"ok": True}))

    elif cmd == "mem-get":
        val = mem_get(sys.argv[2], sys.argv[3])
        print(json.dumps(val, ensure_ascii=False))

    elif cmd == "events":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        print(json.dumps(events_recent(limit), ensure_ascii=False))

    elif cmd == "agents":
        print(json.dumps(agent_list(), ensure_ascii=False))

    else:
        raise SystemExit("Usage: state_store.py [init|migrate|add|update|list|get|mem-set|mem-get|events|agents]")
