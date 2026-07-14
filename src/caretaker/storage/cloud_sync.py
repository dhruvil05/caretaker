"""
storage/cloud_sync.py
Phase 3 — Supabase cloud sync for Caretaker.

Responsibilities:
  - Encrypt all memory content before upload (via encrypt.py)
  - Upload new/updated memories to Supabase 'memories' table
  - Download memories from Supabase for restore
  - Full local restore from cloud (disaster recovery)
  - Track last_synced_at per memory to avoid redundant uploads

Supabase table schema expected:
  CREATE TABLE memories (
    id              TEXT PRIMARY KEY,
    source_agent    TEXT,
    type            TEXT,
    status          TEXT,
    temperature     TEXT,
    fact_type       TEXT,
    importance      REAL,
    decay_score     REAL,
    created_at      TEXT,
    updated_at      TEXT,
    last_used       TEXT,
    -- encrypted fields:
    short           TEXT,
    full            TEXT,
    keywords        TEXT,
    subtype         TEXT,
    source_msg      TEXT,
    superseded_by   TEXT,
    retrieval_count INTEGER,
    _encrypted      BOOLEAN DEFAULT TRUE
  );

Setup:
  1. Create a Supabase project at https://supabase.com
  2. Create the 'memories' table (schema above)
  3. Set supabase_url + supabase_key + encrypt_key in config.json
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class CloudSync:
    """
    Supabase-backed cloud sync for Caretaker memories.

    Usage:
        sync = CloudSync(config)
        if sync.is_configured():
            result = sync.push_all()
            result = sync.pull_all()
    """

    def __init__(self, config: dict):
        self.config       = config
        self._client      = None
        self._encryptor   = None
        self._initialized = False

    # ── Init / connectivity ────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """Return True only if supabase_url, supabase_key, and encrypt_key are all set."""
        return bool(
            self.config.get("supabase_url")
            and self.config.get("supabase_key")
            and self.config.get("encrypt_key")
        )

    def initialize(self) -> bool:
        """
        Connect to Supabase and initialize encryptor.
        Returns True on success, False on failure.
        Call this lazily — not in __init__ — so missing deps don't crash startup.
        """
        if self._initialized:
            return True

        if not self.is_configured():
            logger.warning(
                "[CloudSync] Not configured. Set supabase_url, supabase_key, "
                "and encrypt_key in config.json to enable cloud sync."
            )
            return False

        # Initialize encryptor
        try:
            from src.caretaker.storage.encrypt import Encryptor
            self._encryptor = Encryptor(self.config)
        except Exception as e:
            logger.error(f"[CloudSync] Encryptor init failed: {e}")
            return False

        # Initialize Supabase client
        try:
            from supabase import create_client
            self._client = create_client(
                self.config["supabase_url"],
                self.config["supabase_key"],
            )
            self._initialized = True
            logger.info("[CloudSync] Supabase client initialized.")
            return True
        except ImportError:
            logger.error(
                "[CloudSync] 'supabase' package not installed. "
                "Run: uv add supabase"
            )
            return False
        except Exception as e:
            logger.error(f"[CloudSync] Supabase connection failed: {e}")
            return False

    # ── Push (local → cloud) ───────────────────────────────────────────────────

    def push_all(self) -> dict:
        """
        Push all local memories to Supabase.
        Encrypts content fields before upload.
        Uses upsert so re-running is safe (idempotent).

        Returns: { "pushed": int, "failed": int, "skipped": int }
        """
        if not self.initialize():
            return {"pushed": 0, "failed": 0, "skipped": 0, "error": "not_configured"}

        from src.caretaker.storage.local_db import get_connection
        from src.caretaker.storage.encrypt import encrypt_memory_dict

        try:
            with get_connection() as conn:
                rows = conn.execute("SELECT * FROM memories").fetchall()
            memories = [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[CloudSync] Failed to fetch local memories: {e}")
            return {"pushed": 0, "failed": 0, "skipped": 0, "error": str(e)}

        pushed  = 0
        failed  = 0
        skipped = 0

        for mem in memories:
            try:
                encrypted = encrypt_memory_dict(mem, self._encryptor)
                # Clean up fields Supabase doesn't need or that cause issues
                encrypted = self._prepare_for_upload(encrypted)

                self._client.table("memories").upsert(encrypted).execute()
                pushed += 1

            except Exception as e:
                logger.warning(f"[CloudSync] Push failed for {mem.get('id', '?')[:8]}: {e}")
                failed += 1

        logger.info(f"[CloudSync] Push complete. pushed={pushed} failed={failed} skipped={skipped}")
        return {"pushed": pushed, "failed": failed, "skipped": skipped}

    def push_since(self, since_iso: str) -> dict:
        """
        Push only memories updated since a given ISO timestamp.
        Used by nightly maintenance to push only new/changed memories.

        Returns: { "pushed": int, "failed": int }
        """
        if not self.initialize():
            return {"pushed": 0, "failed": 0, "error": "not_configured"}

        from src.caretaker.storage.local_db import get_connection
        from src.caretaker.storage.encrypt import encrypt_memory_dict

        try:
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE updated_at >= ?",
                    (since_iso,)
                ).fetchall()
            memories = [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[CloudSync] Failed to fetch memories since {since_iso}: {e}")
            return {"pushed": 0, "failed": 0, "error": str(e)}

        pushed = 0
        failed = 0

        for mem in memories:
            try:
                encrypted = encrypt_memory_dict(mem, self._encryptor)
                encrypted = self._prepare_for_upload(encrypted)
                self._client.table("memories").upsert(encrypted).execute()
                pushed += 1
            except Exception as e:
                logger.warning(f"[CloudSync] Push failed for {mem.get('id', '?')[:8]}: {e}")
                failed += 1

        logger.info(f"[CloudSync] Incremental push complete. pushed={pushed} failed={failed}")
        return {"pushed": pushed, "failed": failed}

    # ── Pull (cloud → local) ───────────────────────────────────────────────────

    def pull_all(self) -> dict:
        """
        Pull all memories from Supabase and restore to local SQLite.
        Decrypts content fields after download.
        Skips memories that already exist locally with same updated_at.

        Returns: { "restored": int, "skipped": int, "failed": int }
        """
        if not self.initialize():
            return {"restored": 0, "skipped": 0, "failed": 0, "error": "not_configured"}

        from src.caretaker.storage.local_db import get_connection, save_memory
        from src.caretaker.storage.encrypt import decrypt_memory_dict

        # Fetch from Supabase (paginated — max 1000 per page)
        all_remote = []
        try:
            page   = 0
            limit  = 1000
            while True:
                response = (
                    self._client.table("memories")
                    .select("*")
                    .range(page * limit, (page + 1) * limit - 1)
                    .execute()
                )
                batch = response.data or []
                all_remote.extend(batch)
                if len(batch) < limit:
                    break
                page += 1
        except Exception as e:
            logger.error(f"[CloudSync] Failed to fetch from Supabase: {e}")
            return {"restored": 0, "skipped": 0, "failed": 0, "error": str(e)}

        restored = 0
        skipped  = 0
        failed   = 0

        for remote_mem in all_remote:
            try:
                # Decrypt content fields
                decrypted = decrypt_memory_dict(remote_mem, self._encryptor)
                decrypted = self._prepare_for_local(decrypted)

                # Check if local copy is same or newer
                with get_connection() as conn:
                    local = conn.execute(
                        "SELECT updated_at FROM memories WHERE id = ?",
                        (decrypted["id"],)
                    ).fetchone()

                if local:
                    local_ts  = local["updated_at"] or ""
                    remote_ts = decrypted.get("updated_at", "")
                    if local_ts >= remote_ts:
                        skipped += 1
                        continue
                    # Update existing local record
                    self._update_local(decrypted)
                else:
                    # Insert new record
                    save_memory(decrypted)

                restored += 1

            except Exception as e:
                logger.warning(f"[CloudSync] Restore failed for {remote_mem.get('id', '?')[:8]}: {e}")
                failed += 1

        logger.info(f"[CloudSync] Pull complete. restored={restored} skipped={skipped} failed={failed}")
        return {"restored": restored, "skipped": skipped, "failed": failed}

    def get_remote_count(self) -> Optional[int]:
        """Return count of memories in Supabase. Returns None if not configured."""
        if not self.initialize():
            return None
        try:
            response = (
                self._client.table("memories")
                .select("id", count="exact")
                .execute()
            )
            return response.count
        except Exception as e:
            logger.warning(f"[CloudSync] Could not get remote count: {e}")
            return None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _prepare_for_upload(self, mem: dict) -> dict:
        """
        Clean a memory dict for Supabase upload.
        - Remove Python-only internal fields
        - Ensure all values are JSON-serialisable
        - Convert list fields to JSON strings if still lists
        """
        import json
        cleaned = {}
        allowed_fields = {
            "id", "source_agent", "type", "status", "temperature",
            "fact_type", "importance", "decay_score", "created_at",
            "updated_at", "last_used", "short", "full", "keywords",
            "subtype", "source_msg", "superseded_by", "retrieval_count",
            "_encrypted",
        }
        for k, v in mem.items():
            if k not in allowed_fields:
                continue
            if isinstance(v, list):
                v = json.dumps(v)
            elif v is None:
                pass  # Supabase accepts null
            cleaned[k] = v
        return cleaned

    def _prepare_for_local(self, mem: dict) -> dict:
        """
        Prepare a cloud memory dict for local SQLite insertion.
        Ensures required fields have defaults.
        Removes Supabase-specific fields.
        """
        mem.pop("_encrypted", None)
        defaults = {
            "source_agent"  : "claude",
            "keywords"      : None,
            "short"         : None,
            "subtype"       : None,
            "fact_type"     : "ADDITIVE",
            "status"        : "ACTIVE",
            "superseded_by" : None,
            "importance"    : 0.5,
            "decay_score"   : 1.0,
            "temperature"   : "HOT",
            "retrieval_count": 0,
            "last_used"     : None,
            "source_msg"    : None,
        }
        for k, v in defaults.items():
            if k not in mem or mem[k] is None:
                mem[k] = v
        return mem

    def _update_local(self, mem: dict):
        """Update an existing local memory record with cloud version."""
        from src.caretaker.storage.local_db import get_connection
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        with get_connection() as conn:
            conn.execute(
                """
                UPDATE memories SET
                    source_agent=?, keywords=?, short=?, full=?,
                    type=?, subtype=?, fact_type=?, status=?,
                    superseded_by=?, importance=?, decay_score=?,
                    temperature=?, retrieval_count=?, updated_at=?,
                    last_used=?, source_msg=?
                WHERE id=?
                """,
                (
                    mem.get("source_agent"),
                    mem.get("keywords"),
                    mem.get("short"),
                    mem.get("full"),
                    mem.get("type"),
                    mem.get("subtype"),
                    mem.get("fact_type"),
                    mem.get("status"),
                    mem.get("superseded_by"),
                    mem.get("importance"),
                    mem.get("decay_score"),
                    mem.get("temperature"),
                    mem.get("retrieval_count"),
                    mem.get("updated_at", now),
                    mem.get("last_used"),
                    mem.get("source_msg"),
                    mem["id"],
                )
            )