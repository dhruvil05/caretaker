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