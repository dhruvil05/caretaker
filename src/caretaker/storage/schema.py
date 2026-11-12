"""
storage/schema.py
Versioned database migrations.

Each migration has a number and runs once; the applied version is stored in
the `schema_version` table. Before applying pending migrations to a database
that already holds memories, a backup copy is written to
<db folder>/backups/ (the newest MAX_BACKUPS are kept).

Migrations must be idempotent where they touch older databases that were
created before versioning existed (those report version 0 but already have
the v1 tables/columns).
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

INITIAL_SQL = Path(__file__).parent / "migrations" / "v001_initial.sql"
MAX_BACKUPS = 5


def _columns(conn, table: str) -> set:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _add_column(conn, table: str, ddl: str) -> None:
    name = ddl.split()[0]
    if name not in _columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


# ── Migrations ─────────────────────────────────────────────────────────────────

def _m1_initial(conn):
    """Phase 1–3 schema (memories table + indexes + phase 2 columns)."""
    conn.executescript(INITIAL_SQL.read_text(encoding="utf-8"))
    _add_column(conn, "memories", "importance_score REAL DEFAULT 0.5")
    _add_column(conn, "memories", "last_accessed_at TEXT")


def _m2_project_and_privacy(conn):
    """0.5: project scopes + sensitive-data tracking."""
    _add_column(conn, "memories", "project TEXT NOT NULL DEFAULT ''")          # '' = global
    _add_column(conn, "memories", "sensitivity TEXT NOT NULL DEFAULT 'none'")  # none | flagged | redacted
    _add_column(conn, "memories", "privacy_findings TEXT")                     # JSON list of findings
    _add_column(conn, "memories", "privacy_reviewed INTEGER NOT NULL DEFAULT 0")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_project ON memories(project)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sensitivity ON memories(sensitivity)")


def _m3_fulltext(conn):
    """0.5: SQLite FTS5 full-text index over short/full/keywords, kept in sync by triggers."""
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
            "short, full, keywords, content='memories', content_rowid='rowid', "
            "tokenize='unicode61 remove_diacritics 2')"
        )
    except sqlite3.OperationalError as e:
        # SQLite built without FTS5: keyword search falls back to LIKE.
        logger.warning(f"[schema] Full-text search unavailable ({e}); using slower fallback search.")
        return
    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, short, full, keywords)
            VALUES (new.rowid, new.short, new.full, new.keywords);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, short, full, keywords)
            VALUES ('delete', old.rowid, old.short, old.full, old.keywords);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF short, full, keywords ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, short, full, keywords)
            VALUES ('delete', old.rowid, old.short, old.full, old.keywords);
            INSERT INTO memories_fts(rowid, short, full, keywords)
            VALUES (new.rowid, new.short, new.full, new.keywords);
        END;
    """)
    conn.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")


MIGRATIONS = [
    (1, "initial schema", _m1_initial),
    (2, "project scopes and privacy columns", _m2_project_and_privacy),
    (3, "full-text search index", _m3_fulltext),
]
LATEST_VERSION = MIGRATIONS[-1][0]


# ── Runner ─────────────────────────────────────────────────────────────────────

def current_version(conn) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] or 0


def has_fulltext(conn) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'"
    ).fetchone() is not None


def _memory_count(conn) -> int:
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'").fetchone() is None:
        return 0
    return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


def backup_database(db_path, label: str = "backup") -> Path:
    """Copy the database with SQLite's online backup API; keep the newest MAX_BACKUPS."""
    db_path = Path(db_path)
    folder = db_path.parent / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = folder / f"{db_path.stem}-{label}-{stamp}.db"
    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    old = sorted(folder.glob(f"{db_path.stem}-*.db"))
    for stale in old[:-MAX_BACKUPS]:
        try:
            stale.unlink()
        except OSError:
            pass
    return target


def migrate(db_path) -> int:
    """Apply pending migrations. Returns the resulting schema version."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        version = current_version(conn)
        pending = [m for m in MIGRATIONS if m[0] > version]
        if not pending:
            return version
        if _memory_count(conn) > 0:
            path = backup_database(db_path, label=f"before-v{pending[-1][0]}")
            logger.info(f"[schema] Backed up database to {path}")
        for number, description, fn in pending:
            with conn:
                fn(conn)
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (number,))
            logger.info(f"[schema] Applied migration {number}: {description}")
        return LATEST_VERSION
    finally:
        conn.close()
