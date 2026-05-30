"""
tests/phase3/test_maintenance.py
Phase 3 — Maintenance Tests: P3-T11 and P3-T12

Fix: all async tests use asyncio.run() directly — no pytest-asyncio needed.
This avoids "async def functions are not natively supported" error when
asyncio_mode is not active (pyproject not installed as package yet).

P3-T11  Nightly Decay    last_used 30 days ago → temperature cooled
P3-T12  Nightly Archive  decay_score < 0.2 → status = ARCHIVED
"""

import sys
import uuid
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import storage.local_db as db_module


# ── Isolated DB ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    test_db = tmp_path / "caretaker_maintenance_test.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(test_db))
    db_module.run_migrations()
    yield test_db


def _make_memory(**overrides) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    base = {
        "id"            : str(uuid.uuid4()),
        "source_agent"  : "claude",
        "keywords"      : '["test", "memory"]',
        "short"         : "Test memory short.",
        "full"          : "Test memory full content for maintenance tests.",
        "type"          : "PROJECT",
        "subtype"       : "test",
        "fact_type"     : "ADDITIVE",
        "status"        : "ACTIVE",
        "superseded_by" : None,
        "importance"    : 0.6,
        "decay_score"   : 1.0,
        "temperature"   : "HOT",
        "retrieval_count": 0,
        "created_at"    : now,
        "updated_at"    : now,
        "last_used"     : now,
        "source_msg"    : "Test source message.",
    }
    base.update(overrides)
    return base


def _seed(**overrides) -> dict:
    mem = _make_memory(**overrides)
    db_module.save_memory(mem)
    return mem


def _old_date(days_ago: int) -> str:
    """Return ISO timestamp N days in the past."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()


def _run(coro):
    """Run an async coroutine synchronously. Works without pytest-asyncio."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_runner(config=None):
    from scheduler.nightly_maintenance import NightlyMaintenance
    return NightlyMaintenance(
        local_db=db_module,
        vector_db=None,
        config=config or {"archive_score": 0.2, "include_cold_in_search": False},
    )


# ── P3-T11: Nightly Decay ──────────────────────────────────────────────────────

class TestP3T11_NightlyDecay:
    """P3-T11: last_used 30 days ago → decay_score reduced / temperature cooled."""

    def test_decay_reduces_temperature_for_old_memory(self):
        old_date = _old_date(30)
        mem = _seed(temperature="HOT", decay_score=1.0, last_used=old_date)

        # Backfill last_accessed_at column too
        with db_module.get_connection() as conn:
            conn.execute(
                "UPDATE memories SET last_accessed_at = ? WHERE id = ?",
                (old_date, mem["id"])
            )

        changed = _run(_make_runner()._task_batch_decay())

        row = db_module.get_memory_by_id(mem["id"])
        assert row["temperature"] in ("WARM", "COLD"), (
            f"HOT memory untouched after 30 days — got {row['temperature']}"
        )

    def test_decay_does_not_cool_recent_memory(self):
        mem = _seed(
            temperature="HOT",
            decay_score=1.0,
            last_used=datetime.now(timezone.utc).isoformat(),
        )
        _run(_make_runner()._task_batch_decay())

        row = db_module.get_memory_by_id(mem["id"])
        assert row["temperature"] == "HOT"

    def test_decay_returns_changed_count(self):
        _seed(temperature="HOT", last_used=_old_date(30))
        _seed(temperature="HOT", last_used=_old_date(30))
        _seed(temperature="HOT", last_used=datetime.now(timezone.utc).isoformat())

        changed = _run(_make_runner()._task_batch_decay())
        assert isinstance(changed, int)
        assert changed >= 2

    def test_priority_hot_never_decays(self):
        old_date = _old_date(60)
        mem = _seed(
            temperature="PRIORITY_HOT",
            decay_score=1.0,
            last_used=old_date,
        )
        with db_module.get_connection() as conn:
            conn.execute(
                "UPDATE memories SET last_accessed_at = ? WHERE id = ?",
                (old_date, mem["id"])
            )

        _run(_make_runner()._task_batch_decay())

        row = db_module.get_memory_by_id(mem["id"])
        assert row["temperature"] == "PRIORITY_HOT", (
            "PRIORITY_HOT should never decay"
        )


# ── P3-T12: Nightly Archive ────────────────────────────────────────────────────

class TestP3T12_NightlyArchive:
    """P3-T12: decay_score < 0.2 → status = ARCHIVED."""

    def test_low_score_memory_gets_archived(self):
        mem = _seed(decay_score=0.1, temperature="COLD", status="ACTIVE")
        _run(_make_runner()._task_stale_cleanup())

        row = db_module.get_memory_by_id(mem["id"])
        assert row["status"] == "ARCHIVED"

    def test_healthy_memory_stays_active(self):
        mem = _seed(decay_score=0.8, temperature="HOT", status="ACTIVE")
        _run(_make_runner()._task_stale_cleanup())

        row = db_module.get_memory_by_id(mem["id"])
        assert row["status"] == "ACTIVE"

    def test_boundary_exactly_at_threshold_stays_active(self):
        """Score == archive_score (0.2) should NOT be archived (uses strict <)."""
        mem = _seed(decay_score=0.2, status="ACTIVE")
        _run(_make_runner()._task_stale_cleanup())

        row = db_module.get_memory_by_id(mem["id"])
        assert row["status"] == "ACTIVE"

    def test_below_threshold_gets_archived(self):
        """Score 0.19 < 0.2 → archived."""
        mem = _seed(decay_score=0.19, status="ACTIVE")
        _run(_make_runner()._task_stale_cleanup())

        row = db_module.get_memory_by_id(mem["id"])
        assert row["status"] == "ARCHIVED"

    def test_full_run_all_returns_result_dict(self):
        _seed(decay_score=0.1, temperature="COLD")
        result = _run(_make_runner().run_all())

        assert isinstance(result, dict)
        assert "archived"         in result
        assert "decayed"          in result
        assert "elapsed_seconds"  in result
        assert "ran_at"           in result

    def test_full_run_archived_count_correct(self):
        _seed(decay_score=0.05, status="ACTIVE")   # should archive
        _seed(decay_score=0.05, status="ACTIVE")   # should archive
        _seed(decay_score=0.90, status="ACTIVE")   # should NOT archive

        result = _run(_make_runner()._task_stale_cleanup())
        assert result >= 2