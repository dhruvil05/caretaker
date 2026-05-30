import sqlite3
import json
import os
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path(__file__).parent.parent / "caretaker.db"
MIGRATION_PATH = Path(__file__).parent / "migrations" / "v001_initial.sql"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def run_migrations():
    sql = MIGRATION_PATH.read_text()
    with get_connection() as conn:
        conn.executescript(sql)

    # Phase 2: add new columns if they don't exist yet (safe migration)
    phase2_columns = [
        "ALTER TABLE memories ADD COLUMN importance_score REAL DEFAULT 0.5",
        "ALTER TABLE memories ADD COLUMN last_accessed_at TEXT",
    ]
    with get_connection() as conn:
        for alter in phase2_columns:
            try:
                conn.execute(alter)
            except Exception:
                pass  # Column already exists — safe to ignore


# ── Phase 1 functions (unchanged) ─────────────────────────────────────────────

def save_memory(memory: dict) -> bool:
    sql = """
        INSERT INTO memories (
            id, source_agent, keywords, short, full,
            type, subtype, fact_type, status, superseded_by,
            importance, decay_score, temperature, retrieval_count,
            created_at, updated_at, last_used
        ) VALUES (
            :id, :source_agent, :keywords, :short, :full,
            :type, :subtype, :fact_type, :status, :superseded_by,
            :importance, :decay_score, :temperature, :retrieval_count,
            :created_at, :updated_at, :last_used
        )
    """
    try:
        with get_connection() as conn:
            conn.execute(sql, memory)
        return True
    except Exception as e:
        print(f"[DB] save_memory error: {e}")
        return False


def get_memories_by_type(mem_type: str = None, status: str = "ACTIVE") -> list:
    if mem_type:
        sql = "SELECT * FROM memories WHERE status = ? AND type = ? ORDER BY importance DESC, created_at DESC"
        params = (status, mem_type)
    else:
        sql = "SELECT * FROM memories WHERE status = ? ORDER BY importance DESC, created_at DESC"
        params = (status,)

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_recent_memories(limit: int = 3) -> list:
    sql = """
        SELECT * FROM memories
        WHERE status = 'ACTIVE'
        ORDER BY created_at DESC
        LIMIT ?
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_memory_by_id(memory_id: str) -> dict | None:
    sql = "SELECT * FROM memories WHERE id = ?"
    with get_connection() as conn:
        row = conn.execute(sql, (memory_id,)).fetchone()
    return dict(row) if row else None


def update_memory_status(memory_id: str, status: str, superseded_by: str = None):
    now = datetime.now(timezone.utc).isoformat()
    sql = "UPDATE memories SET status = ?, superseded_by = ?, updated_at = ? WHERE id = ?"
    with get_connection() as conn:
        conn.execute(sql, (status, superseded_by, now, memory_id))


def increment_retrieval_count(memory_id: str):
    now = datetime.now(timezone.utc).isoformat()
    sql = "UPDATE memories SET retrieval_count = retrieval_count + 1, last_used = ? WHERE id = ?"
    with get_connection() as conn:
        conn.execute(sql, (now, memory_id))


def get_all_active_memories() -> list:
    sql = "SELECT * FROM memories WHERE status = 'ACTIVE' ORDER BY importance DESC"
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


# ── Phase 2 additions ──────────────────────────────────────────────────────────

def update_compression(memory_id: str, short: str, keywords: list):
    """
    Phase 2: Update memory with compressed SHORT text and keywords.
    Called by compression_queue worker after Haiku/local compression completes.
    Also sets status = ACTIVE and records last_accessed_at.
    """
    now = datetime.now(timezone.utc).isoformat()
    sql = """
        UPDATE memories
        SET short = ?, keywords = ?, status = 'ACTIVE',
            last_accessed_at = ?, updated_at = ?
        WHERE id = ?
    """
    with get_connection() as conn:
        conn.execute(sql, (short, json.dumps(keywords), now, now, memory_id))


def update_status(memory_id: str, status: str):
    """
    Phase 2: Simple status update (ACTIVE / OUTDATED / PENDING_COMPRESSION).
    Used by conflict_checker and compression_queue on final retry failure.
    """
    now = datetime.now(timezone.utc).isoformat()
    sql = "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?"
    with get_connection() as conn:
        conn.execute(sql, (status, now, memory_id))


def update_temperature(memory_id: str, temperature: str):
    """
    Phase 2: Update temperature tier after decay or reheat.
    Called by maintenance.py (batch decay) and retrieval_engine (reheat).
    """
    now = datetime.now(timezone.utc).isoformat()
    sql = "UPDATE memories SET temperature = ?, updated_at = ? WHERE id = ?"
    with get_connection() as conn:
        conn.execute(sql, (temperature, now, memory_id))


def touch_last_accessed(memory_id: str):
    """
    Phase 2: Update last_accessed_at timestamp when memory is retrieved.
    Used by temperature reheat logic in retrieval_engine.
    """
    now = datetime.now(timezone.utc).isoformat()
    sql = "UPDATE memories SET last_accessed_at = ? WHERE id = ?"
    with get_connection() as conn:
        conn.execute(sql, (now, memory_id))


def get_active_by_type(mem_type: str) -> list:
    """
    Phase 2: Fetch all ACTIVE memories of a specific type.
    Used by conflict_checker to find existing memories before insert.
    """
    sql = "SELECT * FROM memories WHERE status = 'ACTIVE' AND type = ? ORDER BY created_at DESC"
    with get_connection() as conn:
        rows = conn.execute(sql, (mem_type,)).fetchall()
    return [dict(r) for r in rows]


def get_by_ids(memory_ids: list) -> list:
    """
    Phase 2: Fetch multiple memories by ID list.
    Used by semantic_searcher to hydrate ChromaDB hits with full SQLite records.
    """
    if not memory_ids:
        return []
    placeholders = ",".join("?" * len(memory_ids))
    sql = f"SELECT * FROM memories WHERE id IN ({placeholders})"
    with get_connection() as conn:
        rows = conn.execute(sql, memory_ids).fetchall()
    return [dict(r) for r in rows]


def get_all_for_decay() -> list:
    """
    Phase 2: Fetch all ACTIVE memories with temperature + last_accessed_at.
    Used by maintenance.py for nightly batch_decay run.
    """
    sql = """
        SELECT id, temperature, last_accessed_at, last_used
        FROM memories
        WHERE status = 'ACTIVE'
    """
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


# ── Phase 3 additions ──────────────────────────────────────────────────────────

def get_all_memories(status: str = None) -> list:
    """
    Phase 3: Fetch ALL memories with optional status filter.
    Used by CLI list, export, and cloud_sync push_all.
    If status is None — returns every memory regardless of status.
    """
    if status:
        sql = "SELECT * FROM memories WHERE status = ? ORDER BY temperature DESC, importance DESC, created_at DESC"
        params = (status,)
    else:
        sql = "SELECT * FROM memories ORDER BY status ASC, temperature DESC, importance DESC, created_at DESC"
        params = ()

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def update_memory_fields(memory_id: str, fields: dict) -> bool:
    """
    Phase 3: Update arbitrary fields on a memory record.
    Used by CLI edit command and cloud restore.

    fields: dict of column_name → new_value
    Always sets updated_at to now.
    """
    if not fields:
        return False

    now = datetime.now(timezone.utc).isoformat()
    fields["updated_at"] = now

    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values     = list(fields.values()) + [memory_id]

    sql = f"UPDATE memories SET {set_clause} WHERE id = ?"
    try:
        with get_connection() as conn:
            conn.execute(sql, values)
        return True
    except Exception as e:
        print(f"[DB] update_memory_fields error: {e}")
        return False


def archive_memory(memory_id: str) -> bool:
    """
    Phase 3: Soft-delete a memory by setting status = ARCHIVED.
    Never hard-deletes. Used by CLI delete command.
    """
    now = datetime.now(timezone.utc).isoformat()
    sql = "UPDATE memories SET status = 'ARCHIVED', updated_at = ? WHERE id = ?"
    try:
        with get_connection() as conn:
            conn.execute(sql, (now, memory_id))
        return True
    except Exception as e:
        print(f"[DB] archive_memory error: {e}")
        return False


def restore_memory(memory_id: str) -> bool:
    """
    Phase 3: Restore an ARCHIVED or OUTDATED memory back to ACTIVE.
    Clears superseded_by link and resets temperature to WARM.
    Used by CLI restore command.
    """
    now = datetime.now(timezone.utc).isoformat()
    sql = """
        UPDATE memories
        SET status = 'ACTIVE', superseded_by = NULL,
            temperature = 'WARM', updated_at = ?
        WHERE id = ?
    """
    try:
        with get_connection() as conn:
            conn.execute(sql, (now, memory_id))
        return True
    except Exception as e:
        print(f"[DB] restore_memory error: {e}")
        return False


def get_stats() -> dict:
    """
    Phase 3: Return memory health statistics.
    Used by CLI stats command and nightly maintenance report.

    Returns:
        {
            "total": int,
            "by_status": { "ACTIVE": n, "OUTDATED": n, "ARCHIVED": n, ... },
            "by_type":   { "PROJECT": n, ... },
            "by_temperature": { "HOT": n, "WARM": n, ... },
            "by_agent":  { "claude": n, "chatgpt": n, ... },
        }
    """
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

        status_rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM memories GROUP BY status"
        ).fetchall()

        type_rows = conn.execute(
            "SELECT type, COUNT(*) as cnt FROM memories WHERE status='ACTIVE' GROUP BY type"
        ).fetchall()

        temp_rows = conn.execute(
            "SELECT temperature, COUNT(*) as cnt FROM memories WHERE status='ACTIVE' GROUP BY temperature"
        ).fetchall()

        agent_rows = conn.execute(
            "SELECT source_agent, COUNT(*) as cnt FROM memories GROUP BY source_agent"
        ).fetchall()

    return {
        "total"          : total,
        "by_status"      : {r["status"]: r["cnt"] for r in status_rows},
        "by_type"        : {r["type"]: r["cnt"] for r in type_rows},
        "by_temperature" : {r["temperature"]: r["cnt"] for r in temp_rows},
        "by_agent"       : {r["source_agent"]: r["cnt"] for r in agent_rows},
    }


def search_memories_by_keyword(query: str, limit: int = 10) -> list:
    """
    Phase 3: Basic keyword search fallback for CLI search command.
    Used when ChromaDB is unavailable or for quick CLI lookups.
    Searches in full + short + keywords fields (LIKE).
    """
    pattern = f"%{query}%"
    sql = """
        SELECT * FROM memories
        WHERE status = 'ACTIVE'
          AND (full LIKE ? OR short LIKE ? OR keywords LIKE ?)
        ORDER BY importance DESC, created_at DESC
        LIMIT ?
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (pattern, pattern, pattern, limit)).fetchall()
    return [dict(r) for r in rows]


def upsert_memory(memory: dict) -> bool:
    """
    Phase 3: Insert or update a memory record (used by cloud pull restore).
    If id already exists — updates all fields.
    If id not found — inserts as new.
    """
    existing = get_memory_by_id(memory["id"])
    if existing:
        fields = {k: v for k, v in memory.items() if k != "id"}
        return update_memory_fields(memory["id"], fields)
    else:
        return save_memory(memory)


def get_memories_by_agent(agent_id: str) -> list:
    """
    Phase 3: Fetch all ACTIVE memories captured by a specific agent.
    Used by multi-agent stats and CLI list --agent filter.
    """
    sql = """
        SELECT * FROM memories
        WHERE status = 'ACTIVE' AND source_agent = ?
        ORDER BY importance DESC, created_at DESC
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (agent_id,)).fetchall()
    return [dict(r) for r in rows]


def get_duplicate_candidates() -> list:
    """
    Phase 3: Fetch ACTIVE memory pairs that share the same type and similar keywords.
    Used by nightly deduplication task in nightly_maintenance.py.
    Returns list of dicts with id, type, keywords, short for comparison.
    """
    sql = """
        SELECT id, type, keywords, short, full
        FROM memories
        WHERE status = 'ACTIVE' AND keywords IS NOT NULL
        ORDER BY type, created_at DESC
    """
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]