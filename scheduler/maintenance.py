"""
scheduler/maintenance.py
Nightly maintenance runner for Caretaker Phase 2.
Runs automatically at configured maintenance_time (default: 02:00).

Tasks:
  1. BATCH DECAY     — cool down memories not accessed recently (HOT→WARM→COLD)
  2. CHROMADB SYNC   — remove OUTDATED/COLD memories from vector index
  3. STALE CLEANUP   — archive memories with decay_score below archive threshold
  4. STATS REPORT    — log memory health summary after maintenance
"""

import os
import asyncio
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class MaintenanceRunner:
    """
    Scheduled nightly maintenance for Phase 2.
    Wired into server.py _startup() as a background asyncio task.
    """

    def __init__(self, local_db, vector_db, config: dict = None):
        self.local_db  = local_db
        self.vector_db = vector_db
        self.config    = config or {}
        self._running  = False
        self._task     = None

        # Read maintenance time from config (default 02:00)
        self.maintenance_time = self.config.get("maintenance_time", "02:00")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        """Start the background maintenance scheduler."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info(f"[Maintenance] Scheduler started. Runs daily at {self.maintenance_time}.")

    async def stop(self):
        """Graceful shutdown."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[Maintenance] Scheduler stopped.")

    # ── Scheduler loop ────────────────────────────────────────────────────────

    async def _scheduler_loop(self):
        """Wait until maintenance_time, run tasks, repeat daily."""
        while self._running:
            seconds_until = self._seconds_until_maintenance()
            logger.info(f"[Maintenance] Next run in {seconds_until/3600:.1f}h at {self.maintenance_time}")
            await asyncio.sleep(seconds_until)

            if not self._running:
                break

            await self.run_all()

    def _seconds_until_maintenance(self) -> float:
        """Calculate seconds until next maintenance_time window."""
        now = datetime.now()
        try:
            h, m = map(int, self.maintenance_time.split(":"))
        except Exception:
            h, m = 2, 0

        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if target <= now:
            # Already passed today — schedule for tomorrow
            from datetime import timedelta
            target += timedelta(days=1)

        return (target - now).total_seconds()

    # ── Main maintenance pipeline ─────────────────────────────────────────────

    async def run_all(self):
        """Run all maintenance tasks in order. Can be called manually too."""
        start_time = time.time()
        logger.info("[Maintenance] ── Starting nightly maintenance ──")

        decayed_count  = await self._task_batch_decay()
        synced_count   = await self._task_chromadb_sync()
        archived_count = await self._task_stale_cleanup()

        elapsed = time.time() - start_time
        await self._task_stats_report(decayed_count, synced_count, archived_count, elapsed)

        logger.info(f"[Maintenance] ── Done in {elapsed:.2f}s ──")

    # ── Task 1: Batch decay ───────────────────────────────────────────────────

    async def _task_batch_decay(self) -> int:
        """
        Apply temperature decay to all active memories.
        HOT → WARM after 7d idle. WARM → COLD after 14d idle.
        PRIORITY_HOT never decays.
        Returns count of memories that changed tier.
        """
        logger.info("[Maintenance] Task 1: Batch temperature decay...")

        try:
            from memory.temperature_engine import apply_decay

            memories = self.local_db.get_all_for_decay()
            changed_count = 0

            for mem in memories:
                old_tier = mem.get("temperature", "WARM")

                # Use last_accessed_at if set, fallback to last_used
                last_accessed = mem.get("last_accessed_at") or mem.get("last_used")
                if last_accessed:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
                        last_accessed_ts = dt.timestamp()
                    except Exception:
                        last_accessed_ts = time.time()
                else:
                    last_accessed_ts = time.time()

                new_tier = apply_decay(old_tier, last_accessed_ts)

                if new_tier != old_tier:
                    self.local_db.update_temperature(mem["id"], new_tier)
                    changed_count += 1
                    logger.debug(
                        f"[Maintenance] Decay: {mem['id'][:8]}... "
                        f"{old_tier} → {new_tier}"
                    )

            logger.info(f"[Maintenance] Batch decay complete. {changed_count}/{len(memories)} memories cooled.")
            return changed_count

        except Exception as e:
            logger.error(f"[Maintenance] Batch decay failed: {e}")
            return 0

    # ── Task 2: ChromaDB sync ─────────────────────────────────────────────────

    async def _task_chromadb_sync(self) -> int:
        """
        Remove OUTDATED memories from ChromaDB vector index.
        Also removes COLD memories if include_cold_in_search is False.
        Returns count of entries removed from ChromaDB.
        """
        logger.info("[Maintenance] Task 2: ChromaDB sync...")

        if not self.vector_db:
            logger.info("[Maintenance] No VectorDB available. Skipping ChromaDB sync.")
            return 0

        try:
            removed_count = 0
            include_cold  = self.config.get("include_cold_in_search", False)

            # Get all OUTDATED memory IDs from SQLite
            outdated = [
                m for m in self.local_db.get_all_active_memories.__module__ and []
                # Fetch via raw query
            ]

            # Direct approach: fetch outdated via local_db
            all_mems = self._get_non_active_memories()

            for mem in all_mems:
                status = mem.get("status", "")
                temp   = mem.get("temperature", "")

                should_remove = (
                    status == "OUTDATED"
                    or (temp == "COLD" and not include_cold)
                )

                if should_remove:
                    try:
                        self.vector_db.delete(mem["id"])
                        removed_count += 1
                        logger.debug(f"[Maintenance] ChromaDB removed: {mem['id'][:8]}... (status={status}, temp={temp})")
                    except Exception as e:
                        logger.warning(f"[Maintenance] ChromaDB delete failed for {mem['id']}: {e}")

            logger.info(f"[Maintenance] ChromaDB sync complete. {removed_count} entries removed.")
            return removed_count

        except Exception as e:
            logger.error(f"[Maintenance] ChromaDB sync failed: {e}")
            return 0

    def _get_non_active_memories(self) -> list:
        """Fetch OUTDATED + COLD memories for ChromaDB cleanup."""
        try:
            import sqlite3
            from pathlib import Path

            # Reuse DB_PATH from local_db module
            from storage.local_db import get_connection
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT id, status, temperature FROM memories WHERE status = 'OUTDATED' OR temperature = 'COLD'"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"[Maintenance] Could not fetch non-active memories: {e}")
            return []

    # ── Task 3: Stale cleanup ─────────────────────────────────────────────────

    async def _task_stale_cleanup(self) -> int:
        """
        Archive memories with decay_score below archive threshold.
        Marks them OUTDATED in SQLite (they are already removed from ChromaDB by task 2).
        Returns count of archived memories.
        """
        logger.info("[Maintenance] Task 3: Stale memory cleanup...")

        try:
            archive_threshold = self.config.get("archive_score", 0.2)

            from storage.local_db import get_connection
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT id, decay_score FROM memories WHERE status = 'ACTIVE' AND decay_score < ?",
                    (archive_threshold,)
                ).fetchall()

            stale = [dict(r) for r in rows]
            archived_count = 0

            for mem in stale:
                self.local_db.update_status(mem["id"], "OUTDATED")
                archived_count += 1
                logger.debug(f"[Maintenance] Archived stale memory {mem['id'][:8]}... (decay={mem['decay_score']:.2f})")

            logger.info(f"[Maintenance] Stale cleanup complete. {archived_count} memories archived.")
            return archived_count

        except Exception as e:
            logger.error(f"[Maintenance] Stale cleanup failed: {e}")
            return 0

    # ── Task 4: Stats report ──────────────────────────────────────────────────

    async def _task_stats_report(
        self,
        decayed: int,
        synced: int,
        archived: int,
        elapsed: float,
    ):
        """Log a memory health summary after all tasks complete."""
        try:
            from storage.local_db import get_connection
            with get_connection() as conn:
                total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                active = conn.execute("SELECT COUNT(*) FROM memories WHERE status='ACTIVE'").fetchone()[0]
                outdated = conn.execute("SELECT COUNT(*) FROM memories WHERE status='OUTDATED'").fetchone()[0]

                temp_counts = {}
                for tier in ("PRIORITY_HOT", "HOT", "WARM", "COLD"):
                    count = conn.execute(
                        "SELECT COUNT(*) FROM memories WHERE status='ACTIVE' AND temperature=?", (tier,)
                    ).fetchone()[0]
                    temp_counts[tier] = count

            vector_count = self.vector_db.count() if self.vector_db else "N/A"

            logger.info(
                f"[Maintenance] ── Health Report ──\n"
                f"  Total memories : {total}\n"
                f"  Active         : {active}\n"
                f"  Outdated       : {outdated}\n"
                f"  ChromaDB index : {vector_count}\n"
                f"  Temperatures   : {temp_counts}\n"
                f"  Decayed        : {decayed}\n"
                f"  ChromaDB removed: {synced}\n"
                f"  Archived       : {archived}\n"
                f"  Elapsed        : {elapsed:.2f}s"
            )

        except Exception as e:
            logger.warning(f"[Maintenance] Stats report failed: {e}")