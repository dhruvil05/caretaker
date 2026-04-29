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

def update_memory(memory_id: str, fields: dict) -> bool:
    """
    Update any fields on a memory row by ID.
    Phase 2 addition — used by conflict_checker and compression callback.

    Args:
        memory_id: UUID of memory to update.
        fields:    Dict of column → value pairs to update.

    Returns:
        True on success, False on error.
    """
    if not fields:
        return False

    now = datetime.now(timezone.utc).isoformat()
    fields["updated_at"] = now

    set_clause = ", ".join(f"{col} = ?" for col in fields)
    values     = list(fields.values()) + [memory_id]
    sql        = f"UPDATE memories SET {set_clause} WHERE id = ?"

    try:
        with get_connection() as conn:
            conn.execute(sql, values)
        return True
    except Exception as e:
        print(f"[DB] update_memory error: {e}")
        return False