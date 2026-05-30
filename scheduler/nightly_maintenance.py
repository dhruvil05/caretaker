"""
scheduler/nightly_maintenance.py
Phase 3 — Full nightly maintenance pipeline for Caretaker.

Replaces the Phase 2 maintenance.py with the complete Phase 3 pipeline.
Phase 2 maintenance.py still exists and handles the scheduler loop — this
module contains the actual task implementations that nightly_maintenance
runs. It is also callable manually via CLI: caretaker maintenance.

Full pipeline (runs in order every night):
  1. BATCH DECAY          — cool memories not accessed recently
  2. CHROMADB SYNC        — remove OUTDATED/COLD from vector index
  3. STALE CLEANUP        — archive memories with decay_score < threshold
  4. DEDUPLICATION        — merge near-identical ACTIVE memories
  5. IMPORTANCE BOOST     — boost frequently retrieved memories (+0.02/use)
  6. CLOUD SYNC           — encrypt + push updated memories to Supabase
  7. CHROMADB REINDEX     — rebuild vector index for optimal search
  8. STATS REPORT         — write maintenance log with counts

Phase 2 tasks (1-3 + stats) preserved exactly. Phase 3 adds tasks 4-7.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class NightlyMaintenance:
    """
    Full Phase 3 nightly maintenance pipeline.

    Usage (manual via CLI):
        runner = NightlyMaintenance(local_db, vector_db, config)
        result = await runner.run_all()

    Usage (scheduled — wired in scheduler.py):
        runner = NightlyMaintenance(local_db, vector_db, config)
        await runner.start()   # starts APScheduler job
        await runner.stop()    # clean shutdown
    """

    def __init__(self, local_db, vector_db, config: dict = None):
        self.local_db   = local_db
        self.vector_db  = vector_db
        self.config     = config or {}
        self._scheduler = None   # APScheduler instance, set in start()

    # ── Main pipeline ──────────────────────────────────────────────────────────

    async def run_all(self) -> dict:
        """
        Run all 8 maintenance tasks in order.
        Returns a summary dict with counts from every task.
        Can be called manually from CLI without a scheduler running.
        """
        start_time = time.time()
        now_iso    = datetime.now(timezone.utc).isoformat()

        logger.info("[NightlyMaintenance] ── Starting Phase 3 nightly maintenance ──")

        results = {}

        # ── Task 1: Batch decay
        results["decayed"] = await self._task_batch_decay()

        # ── Task 2: ChromaDB sync
        results["chroma_removed"] = await self._task_chromadb_sync()

        # ── Task 3: Stale cleanup
        results["archived"] = await self._task_stale_cleanup()

        # ── Task 4: Deduplication (Phase 3 new)
        results["deduped"] = await self._task_deduplication()

        # ── Task 5: Importance boost (Phase 3 new)
        results["boosted"] = await self._task_importance_boost()

        # ── Task 6: Cloud sync (Phase 3 new)
        results["cloud"] = await self._task_cloud_sync(since_iso=now_iso)

        # ── Task 7: ChromaDB reindex (Phase 3 new)
        results["reindexed"] = await self._task_chromadb_reindex()

        # ── Task 8: Stats report
        elapsed = time.time() - start_time
        await self._task_stats_report(results, elapsed)

        results["elapsed_seconds"] = round(elapsed, 2)
        results["ran_at"]          = now_iso

        logger.info(f"[NightlyMaintenance] ── Done in {elapsed:.2f}s ──")
        return results

    # ── Task 1: Batch decay ────────────────────────────────────────────────────

    async def _task_batch_decay(self) -> int:
        """
        Apply temperature decay to all ACTIVE memories.
        HOT → WARM after 7d idle. WARM → COLD after 14d idle.
        PRIORITY_HOT never decays.
        Returns count of memories that changed tier.
        """
        logger.info("[NightlyMaintenance] Task 1: Batch temperature decay...")

        try:
            from memory.temperature_engine import apply_decay

            memories      = self.local_db.get_all_for_decay()
            changed_count = 0

            for mem in memories:
                old_tier     = mem.get("temperature", "WARM")
                last_accessed = mem.get("last_accessed_at") or mem.get("last_used")

                if last_accessed:
                    try:
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
                        f"[NightlyMaintenance] Decay: {mem['id'][:8]}... "
                        f"{old_tier} → {new_tier}"
                    )

            logger.info(
                f"[NightlyMaintenance] Batch decay done. "
                f"{changed_count}/{len(memories)} memories cooled."
            )
            return changed_count

        except Exception as e:
            logger.error(f"[NightlyMaintenance] Batch decay failed: {e}")
            return 0

    # ── Task 2: ChromaDB sync ──────────────────────────────────────────────────

    async def _task_chromadb_sync(self) -> int:
        """
        Remove OUTDATED + COLD memories from ChromaDB vector index.
        SQLite is always source of truth — ChromaDB just mirrors search-eligible set.
        Returns count of entries removed from ChromaDB.
        """
        logger.info("[NightlyMaintenance] Task 2: ChromaDB sync...")

        if not self.vector_db:
            logger.info("[NightlyMaintenance] No VectorDB. Skipping ChromaDB sync.")
            return 0

        try:
            include_cold  = self.config.get("include_cold_in_search", False)
            removed_count = 0

            from storage.local_db import get_connection
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT id, status, temperature FROM memories "
                    "WHERE status = 'OUTDATED' OR status = 'ARCHIVED' OR temperature = 'COLD'"
                ).fetchall()

            stale = [dict(r) for r in rows]

            for mem in stale:
                status = mem.get("status", "")
                temp   = mem.get("temperature", "")

                should_remove = (
                    status in ("OUTDATED", "ARCHIVED")
                    or (temp == "COLD" and not include_cold)
                )

                if should_remove:
                    try:
                        self.vector_db.delete(mem["id"])
                        removed_count += 1
                        logger.debug(
                            f"[NightlyMaintenance] ChromaDB removed: "
                            f"{mem['id'][:8]}... (status={status}, temp={temp})"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[NightlyMaintenance] ChromaDB delete failed "
                            f"for {mem['id']}: {e}"
                        )

            logger.info(
                f"[NightlyMaintenance] ChromaDB sync done. "
                f"{removed_count} entries removed."
            )
            return removed_count

        except Exception as e:
            logger.error(f"[NightlyMaintenance] ChromaDB sync failed: {e}")
            return 0

    # ── Task 3: Stale cleanup ──────────────────────────────────────────────────

    async def _task_stale_cleanup(self) -> int:
        """
        Archive ACTIVE memories whose decay_score dropped below archive threshold.
        Marks them ARCHIVED in SQLite (ChromaDB already cleaned in task 2).
        Returns count of archived memories.
        """
        logger.info("[NightlyMaintenance] Task 3: Stale memory cleanup...")

        try:
            archive_threshold = self.config.get("archive_score", 0.2)

            from storage.local_db import get_connection
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT id, decay_score FROM memories "
                    "WHERE status = 'ACTIVE' AND decay_score < ?",
                    (archive_threshold,)
                ).fetchall()

            stale          = [dict(r) for r in rows]
            archived_count = 0

            for mem in stale:
                self.local_db.update_status(mem["id"], "ARCHIVED")
                archived_count += 1
                logger.debug(
                    f"[NightlyMaintenance] Archived: {mem['id'][:8]}... "
                    f"(decay={mem['decay_score']:.2f})"
                )

            logger.info(
                f"[NightlyMaintenance] Stale cleanup done. "
                f"{archived_count} memories archived."
            )
            return archived_count

        except Exception as e:
            logger.error(f"[NightlyMaintenance] Stale cleanup failed: {e}")
            return 0

    # ── Task 4: Deduplication (Phase 3) ───────────────────────────────────────

    async def _task_deduplication(self) -> int:
        """
        Find near-identical ACTIVE memories of the same type and merge them.
        Strategy:
          - Group memories by TYPE
          - Within each group, compare keyword overlap
          - If two memories share > 70% keywords → merge SHORT summaries, keep newest
          - Older duplicate marked OUTDATED

        Returns count of memories merged/removed.
        """
        logger.info("[NightlyMaintenance] Task 4: Deduplication...")

        try:
            candidates = self.local_db.get_duplicate_candidates()
            merged_count = 0

            # Group by type
            by_type: dict = {}
            for mem in candidates:
                t = mem.get("type", "UNKNOWN")
                by_type.setdefault(t, []).append(mem)

            for mem_type, group in by_type.items():
                if len(group) < 2:
                    continue

                # Compare each pair within type group
                checked_ids = set()
                for i, mem_a in enumerate(group):
                    if mem_a["id"] in checked_ids:
                        continue
                    for mem_b in group[i + 1:]:
                        if mem_b["id"] in checked_ids:
                            continue

                        overlap = self._keyword_overlap(
                            mem_a.get("keywords"), mem_b.get("keywords")
                        )

                        if overlap >= 0.70:
                            # Merge: keep newest, mark oldest OUTDATED
                            newer, older = self._newer_first(mem_a, mem_b)

                            # Merge SHORT summaries if both have them
                            if newer.get("short") and older.get("short"):
                                merged_short = self._merge_shorts(
                                    newer["short"], older["short"]
                                )
                                self.local_db.update_memory_fields(
                                    newer["id"], {"short": merged_short}
                                )

                            # Mark older as OUTDATED
                            self.local_db.update_memory_fields(
                                older["id"],
                                {
                                    "status"       : "OUTDATED",
                                    "superseded_by": newer["id"],
                                }
                            )

                            # Remove older from ChromaDB if exists
                            if self.vector_db:
                                try:
                                    self.vector_db.delete(older["id"])
                                except Exception:
                                    pass

                            checked_ids.add(older["id"])
                            merged_count += 1
                            logger.debug(
                                f"[NightlyMaintenance] Dedup: merged "
                                f"{older['id'][:8]} → {newer['id'][:8]} "
                                f"(overlap={overlap:.0%}, type={mem_type})"
                            )

            logger.info(
                f"[NightlyMaintenance] Deduplication done. "
                f"{merged_count} duplicates merged."
            )
            return merged_count

        except Exception as e:
            logger.error(f"[NightlyMaintenance] Deduplication failed: {e}")
            return 0

    def _keyword_overlap(self, kw_a, kw_b) -> float:
        """Calculate Jaccard overlap between two keyword lists."""
        if not kw_a or not kw_b:
            return 0.0

        # Keywords stored as JSON string or list
        set_a = self._parse_keywords(kw_a)
        set_b = self._parse_keywords(kw_b)

        if not set_a or not set_b:
            return 0.0

        intersection = set_a & set_b
        union        = set_a | set_b
        return len(intersection) / len(union) if union else 0.0

    def _parse_keywords(self, kw) -> set:
        """Parse keywords from list or JSON string to a lowercase set."""
        if isinstance(kw, list):
            return {k.lower().strip() for k in kw if k}
        if isinstance(kw, str):
            try:
                parsed = json.loads(kw)
                if isinstance(parsed, list):
                    return {k.lower().strip() for k in parsed if k}
            except Exception:
                # Comma-separated fallback
                return {k.lower().strip() for k in kw.split(",") if k.strip()}
        return set()

    def _newer_first(self, a: dict, b: dict):
        """Return (newer, older) based on created_at."""
        ts_a = a.get("created_at", "")
        ts_b = b.get("created_at", "")
        if ts_a >= ts_b:
            return a, b
        return b, a

    def _merge_shorts(self, short_a: str, short_b: str) -> str:
        """
        Merge two SHORT summaries into one.
        Simple strategy: take the longer one (more information).
        Cap at 300 chars to stay within short field budget.
        """
        winner = short_a if len(short_a) >= len(short_b) else short_b
        return winner[:300]

    # ── Task 5: Importance boost ───────────────────────────────────────────────

    async def _task_importance_boost(self) -> int:
        """
        Boost importance score of frequently retrieved memories.
        +0.02 per retrieval_count above threshold (max boost: +0.15).
        Keeps hot memories hot even after time passes.
        Returns count of memories boosted.
        """
        logger.info("[NightlyMaintenance] Task 5: Importance boost...")

        try:
            from storage.local_db import get_connection

            # Fetch frequently accessed active memories
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT id, importance, retrieval_count, temperature "
                    "FROM memories "
                    "WHERE status = 'ACTIVE' AND retrieval_count >= 3"
                ).fetchall()

            candidates   = [dict(r) for r in rows]
            boosted_count = 0

            for mem in candidates:
                old_importance = mem.get("importance", 0.5)
                retrieval_count = mem.get("retrieval_count", 0)

                # Boost: +0.02 per extra retrieval above threshold (3), cap +0.15
                boost = min((retrieval_count - 2) * 0.02, 0.15)
                new_importance = min(old_importance + boost, 1.0)
                new_importance = round(new_importance, 4)

                if new_importance > old_importance:
                    self.local_db.update_memory_fields(
                        mem["id"], {"importance": new_importance}
                    )

                    # Recalculate temperature based on new importance
                    from memory.temperature_engine import assign_temperature
                    new_temp = assign_temperature(new_importance, mem.get("temperature", "WARM"))
                    if new_temp != mem.get("temperature"):
                        self.local_db.update_temperature(mem["id"], new_temp)

                    boosted_count += 1
                    logger.debug(
                        f"[NightlyMaintenance] Boosted: {mem['id'][:8]}... "
                        f"importance {old_importance:.2f} → {new_importance:.2f}"
                    )

            logger.info(
                f"[NightlyMaintenance] Importance boost done. "
                f"{boosted_count} memories boosted."
            )
            return boosted_count

        except Exception as e:
            logger.error(f"[NightlyMaintenance] Importance boost failed: {e}")
            return 0

    # ── Task 6: Cloud sync (Phase 3) ──────────────────────────────────────────

    async def _task_cloud_sync(self, since_iso: str) -> dict:
        """
        Encrypt and push new/updated memories to Supabase.
        Uses incremental push (only memories updated since maintenance started).
        Skips gracefully if cloud sync not configured.
        Returns dict from CloudSync.push_since().
        """
        logger.info("[NightlyMaintenance] Task 6: Cloud sync...")

        try:
            from storage.cloud_sync import CloudSync
            cloud = CloudSync(self.config)

            if not cloud.is_configured():
                logger.info(
                    "[NightlyMaintenance] Cloud sync not configured. "
                    "Set supabase_url, supabase_key, encrypt_key in config.json to enable."
                )
                return {"pushed": 0, "failed": 0, "skipped": "not_configured"}

            # Push only memories modified before this maintenance run started
            # (use a wide window — 24h back — to catch anything missed)
            from datetime import timedelta
            window_start = datetime.now(timezone.utc) - timedelta(hours=24)
            window_iso   = window_start.isoformat()

            result = cloud.push_since(window_iso)
            logger.info(
                f"[NightlyMaintenance] Cloud sync done. "
                f"pushed={result.get('pushed', 0)} failed={result.get('failed', 0)}"
            )
            return result

        except ImportError as e:
            logger.warning(f"[NightlyMaintenance] Cloud sync skipped (missing dep): {e}")
            return {"pushed": 0, "failed": 0, "skipped": str(e)}
        except Exception as e:
            logger.error(f"[NightlyMaintenance] Cloud sync failed: {e}")
            return {"pushed": 0, "failed": 0, "error": str(e)}

    # ── Task 7: ChromaDB reindex (Phase 3) ────────────────────────────────────

    async def _task_chromadb_reindex(self) -> int:
        """
        Rebuild ChromaDB vector index for all ACTIVE HOT+WARM memories.
        Ensures search index is in sync with SQLite source of truth.
        Only reindexes memories whose SHORT summary exists but is missing
        from ChromaDB — avoids re-embedding everything every night.
        Returns count of memories reindexed.
        """
        logger.info("[NightlyMaintenance] Task 7: ChromaDB reindex...")

        if not self.vector_db:
            logger.info("[NightlyMaintenance] No VectorDB. Skipping reindex.")
            return 0

        try:
            from storage.local_db import get_connection

            # Fetch ACTIVE memories with SHORT text that should be in ChromaDB
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT id, short, type, temperature "
                    "FROM memories "
                    "WHERE status = 'ACTIVE' "
                    "  AND short IS NOT NULL "
                    "  AND temperature IN ('PRIORITY_HOT', 'HOT', 'WARM')"
                ).fetchall()

            candidates    = [dict(r) for r in rows]
            reindexed_count = 0

            for mem in candidates:
                # Check if already in ChromaDB
                try:
                    existing = self.vector_db.get(mem["id"])
                    if existing:
                        continue  # Already indexed — skip
                except Exception:
                    pass  # Not found — needs reindex

                # Add to ChromaDB
                try:
                    self.vector_db.add(
                        memory_id=mem["id"],
                        text=mem["short"],
                        metadata={
                            "type"       : mem.get("type", "UNKNOWN"),
                            "temperature": mem.get("temperature", "WARM"),
                        }
                    )
                    reindexed_count += 1
                    logger.debug(
                        f"[NightlyMaintenance] Reindexed: {mem['id'][:8]}..."
                    )
                except Exception as e:
                    logger.warning(
                        f"[NightlyMaintenance] Reindex failed for "
                        f"{mem['id'][:8]}: {e}"
                    )

            logger.info(
                f"[NightlyMaintenance] Reindex done. "
                f"{reindexed_count} memories reindexed."
            )
            return reindexed_count

        except Exception as e:
            logger.error(f"[NightlyMaintenance] Reindex failed: {e}")
            return 0

    # ── Task 8: Stats report ───────────────────────────────────────────────────

    async def _task_stats_report(self, results: dict, elapsed: float):
        """
        Log a full memory health summary after all tasks complete.
        Also writes a maintenance log file for audit trail.
        """
        try:
            from storage.local_db import get_stats
            stats = get_stats()

            vector_count = self.vector_db.count() if self.vector_db else "N/A"
            cloud_result = results.get("cloud", {})

            report_lines = [
                "[NightlyMaintenance] ── Phase 3 Health Report ──",
                f"  Ran at           : {results.get('ran_at', 'unknown')}",
                f"  Total memories   : {stats.get('total', '?')}",
                f"  By status        : {stats.get('by_status', {})}",
                f"  By temperature   : {stats.get('by_temperature', {})}",
                f"  By type          : {stats.get('by_type', {})}",
                f"  By agent         : {stats.get('by_agent', {})}",
                f"  ChromaDB index   : {vector_count}",
                f"  ── Task Results ──",
                f"  Decayed          : {results.get('decayed', 0)}",
                f"  ChromaDB removed : {results.get('chroma_removed', 0)}",
                f"  Archived         : {results.get('archived', 0)}",
                f"  Deduped          : {results.get('deduped', 0)}",
                f"  Importance boost : {results.get('boosted', 0)}",
                f"  Cloud pushed     : {cloud_result.get('pushed', 0)}",
                f"  Cloud failed     : {cloud_result.get('failed', 0)}",
                f"  Reindexed        : {results.get('reindexed', 0)}",
                f"  Elapsed          : {elapsed:.2f}s",
            ]

            report = "\n".join(report_lines)
            logger.info(report)

            # Write to log file
            self._write_log_file(report)

        except Exception as e:
            logger.warning(f"[NightlyMaintenance] Stats report failed: {e}")

    def _write_log_file(self, report: str):
        """Write maintenance report to logs/maintenance.log."""
        try:
            from pathlib import Path
            log_dir = Path(__file__).parent.parent / "logs"
            log_dir.mkdir(exist_ok=True)
            log_path = log_dir / "maintenance.log"

            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n{timestamp}\n{report}\n")
        except Exception as e:
            logger.warning(f"[NightlyMaintenance] Could not write log file: {e}")