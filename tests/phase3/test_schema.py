"""
tests/phase3/test_schema.py
Versioned migrations: fresh databases, upgrading pre-versioning databases
(with a backup first), full-text index sync, idempotency.
"""

import sqlite3

from caretaker.storage import schema


OLD_SCHEMA = """
CREATE TABLE memories (
    id TEXT PRIMARY KEY, source_agent TEXT DEFAULT 'claude', keywords TEXT, short TEXT,
    full TEXT NOT NULL, type TEXT NOT NULL, subtype TEXT, fact_type TEXT DEFAULT 'ADDITIVE',
    status TEXT DEFAULT 'ACTIVE', superseded_by TEXT, importance REAL DEFAULT 0.5,
    decay_score REAL DEFAULT 1.0, temperature TEXT DEFAULT 'HOT', retrieval_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_used TEXT
);
INSERT INTO memories (id, full, short, keywords, type, created_at, updated_at)
VALUES ('old-1', 'Building Orbit with FastAPI', 'Builds Orbit', '["orbit","fastapi"]', 'PROJECT', '2026-01-01', '2026-01-01');
"""


def _cols(path):
    with sqlite3.connect(path) as c:
        return {r[1] for r in c.execute("PRAGMA table_info(memories)")}


def _fts_ids(path, query):
    with sqlite3.connect(path) as c:
        return [r[0] for r in c.execute(
            "SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid = f.rowid WHERE memories_fts MATCH ?",
            (query,),
        )]


def test_fresh_database_gets_latest_schema_and_no_backup(tmp_path):
    db = tmp_path / "fresh.db"
    assert schema.migrate(db) == schema.LATEST_VERSION
    assert {"project", "sensitivity", "privacy_findings", "privacy_reviewed", "importance_score"} <= _cols(db)
    assert not (tmp_path / "backups").exists()
    with sqlite3.connect(db) as c:
        assert c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_old_database_is_backed_up_then_upgraded_without_data_loss(tmp_path):
    db = tmp_path / "caretaker.db"
    with sqlite3.connect(db) as c:
        c.executescript(OLD_SCHEMA)

    assert schema.migrate(db) == schema.LATEST_VERSION

    backups = list((tmp_path / "backups").glob("caretaker-before-v*.db"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as c:   # backup holds the pre-migration data
        assert c.execute("SELECT full FROM memories").fetchone()[0] == "Building Orbit with FastAPI"
        assert "project" not in {r[1] for r in c.execute("PRAGMA table_info(memories)")}

    with sqlite3.connect(db) as c:
        row = c.execute("SELECT full, project, sensitivity, privacy_reviewed FROM memories").fetchone()
    assert row == ("Building Orbit with FastAPI", "", "none", 0)
    assert _fts_ids(db, "orbit") == ["old-1"]          # existing rows were indexed


def test_second_run_is_a_no_op(tmp_path):
    db = tmp_path / "caretaker.db"
    with sqlite3.connect(db) as c:
        c.executescript(OLD_SCHEMA)
    schema.migrate(db)
    schema.migrate(db)
    assert len(list((tmp_path / "backups").glob("*.db"))) == 1
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == len(schema.MIGRATIONS)


def test_fulltext_index_follows_insert_update_delete(tmp_path):
    db = tmp_path / "fts.db"
    schema.migrate(db)
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO memories (id, full, type, created_at, updated_at) VALUES ('a', 'likes Postgres', 'PREFERENCE', 'x', 'x')")
    assert _fts_ids(db, "postgres") == ["a"]
    with sqlite3.connect(db) as c:
        c.execute("UPDATE memories SET full = 'likes SQLite' WHERE id = 'a'")
    assert _fts_ids(db, "postgres") == [] and _fts_ids(db, "sqlite") == ["a"]
    with sqlite3.connect(db) as c:
        c.execute("DELETE FROM memories WHERE id = 'a'")
    assert _fts_ids(db, "sqlite") == []


def test_backups_are_rotated(tmp_path):
    db = tmp_path / "caretaker.db"
    with sqlite3.connect(db) as c:
        c.executescript(OLD_SCHEMA)
    for i in range(schema.MAX_BACKUPS + 3):
        schema.backup_database(db, label=f"manual{i:02d}")
    assert len(list((tmp_path / "backups").glob("caretaker-*.db"))) == schema.MAX_BACKUPS
