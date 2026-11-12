from caretaker.paths import db_path
import sqlite3
import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# <data_dir>/data/caretaker.db — see caretaker/paths.py
DB_PATH = db_path()
MIGRATION_PATH = Path(__file__).parent / "migrations" / "v001_initial.sql"


def get_connection():
    # The MCP server, dashboard and CLI can all use the database at once:
    # WAL (set during migration) lets readers and a writer work concurrently,
    # and busy_timeout waits for a lock instead of failing immediately.
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def run_migrations():
    """Bring the database schema up to date (versioned; backs up first). See storage/schema.py."""
    from caretaker.storage.schema import migrate
    migrate(DB_PATH)


# ── Phase 1 functions (unchanged) ─────────────────────────────────────────────

_SAVE_DEFAULTS = {"project": "", "sensitivity": "none", "privacy_findings": None, "privacy_reviewed": 0}


def save_memory(memory: dict) -> bool:
    sql = """
        INSERT INTO memories (
            id, source_agent, keywords, short, full,
            type, subtype, fact_type, status, superseded_by,
            importance, decay_score, temperature, retrieval_count,
            created_at, updated_at, last_used,
            project, sensitivity, privacy_findings, privacy_reviewed
        ) VALUES (
            :id, :source_agent, :keywords, :short, :full,
            :type, :subtype, :fact_type, :status, :superseded_by,
            :importance, :decay_score, :temperature, :retrieval_count,
            :created_at, :updated_at, :last_used,
            :project, :sensitivity, :privacy_findings, :privacy_reviewed
        )
    """
    params = {**_SAVE_DEFAULTS, **memory}
    for key in _SAVE_DEFAULTS:
        if params[key] is None and key != "privacy_findings":
            params[key] = _SAVE_DEFAULTS[key]
    try:
        with get_connection() as conn:
            conn.execute(sql, params)
        return True
    except Exception as e:
        logger.error(f"[DB] save_memory error: {e}")
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


def get_recent_memories(limit: int = 3, project: str = None) -> list:
    """Newest active memories. With `project`, that project's plus global ones."""
    sql = "SELECT * FROM memories WHERE status = 'ACTIVE'"
    args = []
    if project:
        sql += " AND project IN (?, '')"
        args.append(project)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    with get_connection() as conn:
        rows = conn.execute(sql, args).fetchall()
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


def get_memory_columns(conn) -> set:
    """Column names of the local memories table (read from the schema)."""
    return {row[1] for row in conn.execute("PRAGMA table_info(memories)")}


def update_memory_fields(memory_id: str, fields: dict, touch_updated_at: bool = True) -> bool:
    """
    Phase 3: Update arbitrary fields on a memory record.
    Used by CLI edit command, import and cloud restore.

    fields: dict of column_name → new_value
    Only real columns of the memories table are written (column names go
    into the SQL text, so they must never come straight from imported or
    downloaded data). Unknown keys are dropped. `id` can't be changed.
    Sets updated_at to now unless touch_updated_at=False.
    """
    if not fields:
        return False

    fields = dict(fields)  # don't mutate the caller's dict
    fields.pop("id", None)
    if touch_updated_at or "updated_at" not in fields:
        fields["updated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        with get_connection() as conn:
            allowed = get_memory_columns(conn)
            unknown = [k for k in fields if k not in allowed]
            if unknown:
                logger.warning(f"[DB] update_memory_fields: ignoring unknown fields {unknown}")
            fields = {k: v for k, v in fields.items() if k in allowed}

            set_clause = ", ".join(f'"{k}" = ?' for k in fields)
            values     = list(fields.values()) + [memory_id]
            conn.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", values)
        return True
    except Exception as e:
        logger.error(f"[DB] update_memory_fields error: {e}")
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
        logger.error(f"[DB] archive_memory error: {e}")
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
        logger.error(f"[DB] restore_memory error: {e}")
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


_WORD_RE = None
_STOPWORDS = {
    "the", "and", "for", "you", "your", "are", "was", "with", "this", "that", "what", "which", "who",
    "how", "why", "when", "where", "have", "has", "had", "can", "could", "should", "would", "will",
    "about", "from", "into", "does", "did", "not", "but", "any", "all", "our", "out", "its", "i'm",
    "my", "me", "is", "am", "an", "a", "to", "of", "in", "on", "it", "do", "be", "we", "us", "at",
}


def _search_terms(text_or_terms, max_terms: int = 12) -> list:
    """Normalise a query (string or keyword list) into distinct lowercase search words."""
    global _WORD_RE
    if _WORD_RE is None:
        import re
        _WORD_RE = re.compile(r"\w[\w'\-]*\w|\w", re.UNICODE)
    raw = text_or_terms if isinstance(text_or_terms, (list, tuple)) else [text_or_terms or ""]
    terms = []
    for chunk in raw:
        for word in _WORD_RE.findall(str(chunk).lower()):
            if len(word) >= 2 and word not in _STOPWORDS and word not in terms:
                terms.append(word)
    return terms[:max_terms]


_FTS_CACHE = {}


def _has_fulltext(conn) -> bool:
    key = str(DB_PATH)
    if key not in _FTS_CACHE:
        from caretaker.storage.schema import has_fulltext
        _FTS_CACHE[key] = has_fulltext(conn)
    return _FTS_CACHE[key]


def search_fulltext(
    query,
    limit: int = 20,
    statuses=("ACTIVE",),
    temperatures=None,
    memory_type: str = None,
    project: str = None,
    exclude_id: str = None,
    keywords_only: bool = False,
    min_shared: int = 1,
) -> list:
    """
    0.5: ranked keyword search over short/full/keywords using SQLite FTS5 (BM25),
    without loading every memory. `query` is a string or a keyword list. Falls
    back to LIKE matching when this SQLite build has no FTS5.
    Each result carries "match_score" (higher = better).
    """
    terms = _search_terms(query)
    if not terms:
        return []
    filters, args = [], []
    if statuses:
        filters.append(f"m.status IN ({','.join('?' * len(statuses))})")
        args += list(statuses)
    if temperatures:
        filters.append(f"m.temperature IN ({','.join('?' * len(temperatures))})")
        args += list(temperatures)
    if memory_type:
        filters.append("m.type = ?")
        args.append(memory_type)
    if project is not None:
        filters.append("m.project IN (?, '')")      # the project's own memories + global ones
        args.append(project)
    if exclude_id:
        filters.append("m.id != ?")
        args.append(exclude_id)
    where_extra = (" AND " + " AND ".join(filters)) if filters else ""

    with get_connection() as conn:
        if _has_fulltext(conn):
            # Quote every term (no FTS syntax injection).
            quoted = ['"' + t.replace('"', '""') + '"' for t in terms]
            if keywords_only:
                # Duplicate/conflict checks compare keyword sets: whole words in the
                # keywords column only, and (optionally) at least `min_shared` of them.
                if min_shared >= 2 and len(quoted) >= 2:
                    from itertools import combinations
                    match = " OR ".join(f"({a} AND {b})" for a, b in combinations(quoted[:8], 2))
                else:
                    match = " OR ".join(quoted)
                match = f"keywords : ({match})"
            else:
                # Recall/search: prefix-match words of 3+ chars ("deploy" finds "deploys").
                match = " OR ".join(q + ("*" if len(t) >= 3 else "") for q, t in zip(quoted, terms))
            rows = conn.execute(
                "SELECT m.*, bm25(memories_fts) AS _rank FROM memories_fts "
                "JOIN memories m ON m.rowid = memories_fts.rowid "
                f"WHERE memories_fts MATCH ?{where_extra} ORDER BY _rank LIMIT ?",
                [match] + args + [limit],
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["match_score"] = round(-d.pop("_rank"), 6)
                results.append(d)
            return results

        # Fallback: LIKE on each term, ranked by how many terms match.
        likes = " + ".join(
            "((m.full LIKE ?) + (COALESCE(m.short,'') LIKE ?) + (COALESCE(m.keywords,'') LIKE ?) > 0)" for _ in terms
        )
        like_args = []
        for t in terms:
            like_args += [f"%{t}%"] * 3
        rows = conn.execute(
            f"SELECT m.*, ({likes}) AS _hits FROM memories m WHERE ({likes}) > 0{where_extra} "
            "ORDER BY _hits DESC, m.importance DESC LIMIT ?",
            like_args + like_args + args + [limit],
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["match_score"] = float(d.pop("_hits"))
            results.append(d)
        return results


def search_memories_by_keyword(query: str, limit: int = 10) -> list:
    """Keyword search over ACTIVE memories (CLI search, MCP search tool). Ranked; see search_fulltext."""
    return search_fulltext(query, limit=limit)


def get_unsummarised(limit: int = 200) -> list:
    """Active memories still waiting for a short summary, most important first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, full, type FROM memories WHERE status = 'ACTIVE' "
            "AND (short IS NULL OR TRIM(short) = '') AND full IS NOT NULL AND TRIM(full) != '' "
            "ORDER BY importance DESC, created_at DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(r) for r in rows]


def get_top_memories(limit: int = 5, temperatures=("PRIORITY_HOT", "HOT", "WARM"), project: str = None) -> list:
    """Most important active memories — general background when nothing specific matches."""
    sql = "SELECT * FROM memories WHERE status = 'ACTIVE'"
    args = []
    if temperatures:
        sql += f" AND temperature IN ({','.join('?' * len(temperatures))})"
        args += list(temperatures)
    if project is not None:
        sql += " AND project IN (?, '')"
        args.append(project)
    sql += " ORDER BY importance DESC, updated_at DESC LIMIT ?"
    args.append(limit)
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def record_recalls(recalls: list) -> None:
    """
    0.5: record that memories were given to an agent — one transaction instead
    of several connections per memory. `recalls` is a list of
    (memory_id, new_temperature_or_None).
    """
    if not recalls:
        return
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        for memory_id, new_temp in recalls:
            conn.execute(
                "UPDATE memories SET retrieval_count = retrieval_count + 1, last_used = ?, "
                "last_accessed_at = ? WHERE id = ?",
                (now, now, memory_id),
            )
            if new_temp:
                conn.execute(
                    "UPDATE memories SET temperature = ?, updated_at = ? WHERE id = ? AND temperature != ?",
                    (new_temp, now, memory_id, new_temp),
                )


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


def get_duplicate_candidates(since: str = None) -> list:
    """
    ACTIVE memories with keywords, for nightly deduplication. With `since`
    (ISO timestamp) only memories updated after it — the incremental check.
    """
    sql = """
        SELECT id, type, keywords, short, full, project, created_at, updated_at
        FROM memories
        WHERE status = 'ACTIVE' AND keywords IS NOT NULL
    """
    args = []
    if since:
        sql += " AND updated_at >= ?"
        args.append(since)
    sql += " ORDER BY type, created_at DESC"
    with get_connection() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]