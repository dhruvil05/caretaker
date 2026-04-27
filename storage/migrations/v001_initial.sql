CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    source_agent    TEXT DEFAULT 'claude',
    keywords        TEXT,
    short           TEXT,
    full            TEXT NOT NULL,
    type            TEXT NOT NULL,
    subtype         TEXT,
    fact_type       TEXT DEFAULT 'ADDITIVE',
    status          TEXT DEFAULT 'ACTIVE',
    superseded_by   TEXT,
    importance      REAL DEFAULT 0.5,
    decay_score     REAL DEFAULT 1.0,
    temperature     TEXT DEFAULT 'HOT',
    retrieval_count INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_used       TEXT
);

CREATE INDEX IF NOT EXISTS idx_status  ON memories(status);
CREATE INDEX IF NOT EXISTS idx_type    ON memories(type);
CREATE INDEX IF NOT EXISTS idx_temp    ON memories(temperature);
CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at);