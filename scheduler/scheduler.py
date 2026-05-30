"""
scheduler/scheduler.py
Phase 3 — APScheduler setup for Caretaker nightly maintenance.

Wires NightlyMaintenance into APScheduler so it runs automatically
every night at the time configured in config.json (maintenance_time).

This is a standalone scheduler module — separate from the asyncio-based
MaintenanceRunner in maintenance.py (Phase 2). Both can coexist:
  - maintenance.py  : Phase 2 asyncio loop (still used by server.py)
  - scheduler.py    : Phase 3 APScheduler (used by CLI + optional server wiring)

The scheduler can also be triggered manually:
    sched = CaretakerScheduler(config, local_db, vector_db)
    sched.start()
    sched.trigger_now()   # run maintenance immediately
    sched.stop()

APScheduler is optional — if not installed, scheduler degrades gracefully.
Install with: uv add apscheduler
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class CaretakerScheduler:
    """
    APScheduler wrapper for nightly Caretaker maintenance.

    Lifecycle:
        start()        → register APScheduler job, begin scheduling
        trigger_now()  → run maintenance immediately (bypass schedule)
        status()       → return dict with next_run_time, job_id, etc.
        stop()         → shutdown scheduler cleanly
    """

    JOB_ID = "caretaker_nightly_maintenance"

    def __init__(self, config: dict, local_db, vector_db):
        self.config     = config
        self.local_db   = local_db
        self.vector_db  = vector_db
        self._scheduler = None
        self._running   = False

        # Parse maintenance_time from config (default 02:00)
        raw_time = config.get("maintenance_time", "02:00")
        try:
            h, m = map(int, raw_time.split(":"))
            self._hour   = h
            self._minute = m
        except Exception:
            logger.warning(
                f"[Scheduler] Invalid maintenance_time '{raw_time}'. "
                "Defaulting to 02:00."
            )
            self._hour   = 2
            self._minute = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Start APScheduler with a daily cron job.
        Returns True on success, False if APScheduler not installed.
        """
        if self._running:
            logger.info("[Scheduler] Already running.")
            return True

        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.warning(
                "[Scheduler] APScheduler not installed. "
                "Run: uv add apscheduler  — to enable nightly scheduling. "
                "Manual trigger still available via CLI: caretaker maintenance"
            )
            return False

        self._scheduler = AsyncIOScheduler()

        trigger = CronTrigger(
            hour=self._hour,
            minute=self._minute,
            timezone="UTC",
        )

        self._scheduler.add_job(
            func=self._run_maintenance_job,
            trigger=trigger,
            id=self.JOB_ID,
            name="Caretaker Nightly Maintenance",
            replace_existing=True,
            misfire_grace_time=3600,   # Allow up to 1h late start (e.g. if machine was asleep)
        )

        self._scheduler.start()
        self._running = True

        next_run = self._get_next_run_time()
        logger.info(
            f"[Scheduler] Started. Nightly job registered. "
            f"Next run: {next_run} UTC  "
            f"(configured: {self._hour:02d}:{self._minute:02d} UTC)"
        )
        return True

    def stop(self):
        """Shutdown APScheduler cleanly."""
        if self._scheduler and self._running:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception as e:
                logger.warning(f"[Scheduler] Shutdown warning: {e}")
            self._running = False
            logger.info("[Scheduler] Stopped.")

    def is_running(self) -> bool:
        """Return True if scheduler is active."""
        return self._running

    # ── Manual trigger ─────────────────────────────────────────────────────────

    async def trigger_now(self) -> dict:
        """
        Run maintenance immediately regardless of schedule.
        Used by CLI: caretaker maintenance
        Returns maintenance result dict.
        """
        logger.info("[Scheduler] Manual trigger: running maintenance now...")
        return await self._run_maintenance_job()

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Return scheduler status dict.
        Used by CLI: caretaker stats  and  caretaker config
        """
        if not self._running or not self._scheduler:
            return {
                "running"      : False,
                "next_run"     : None,
                "schedule"     : f"{self._hour:02d}:{self._minute:02d} UTC",
                "job_id"       : self.JOB_ID,
                "apscheduler"  : self._is_apscheduler_installed(),
            }

        return {
            "running"   : True,
            "next_run"  : self._get_next_run_time(),
            "schedule"  : f"{self._hour:02d}:{self._minute:02d} UTC",
            "job_id"    : self.JOB_ID,
            "apscheduler": True,
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _run_maintenance_job(self) -> dict:
        """
        Actual maintenance job called by APScheduler or trigger_now().
        Instantiates NightlyMaintenance and runs the full pipeline.
        Catches all exceptions — scheduler must never crash from a failed job.
        """
        try:
            from scheduler.nightly_maintenance import NightlyMaintenance
            runner = NightlyMaintenance(
                local_db=self.local_db,
                vector_db=self.vector_db,
                config=self.config,
            )
            result = await runner.run_all()
            return result

        except Exception as e:
            logger.error(
                f"[Scheduler] Maintenance job failed with unhandled exception: {e}",
                exc_info=True,
            )
            return {"error": str(e), "ran_at": datetime.now(timezone.utc).isoformat()}

    def _get_next_run_time(self) -> Optional[str]:
        """Return ISO string of next scheduled run time."""
        if not self._scheduler:
            return None
        try:
            job = self._scheduler.get_job(self.JOB_ID)
            if job and job.next_run_time:
                return job.next_run_time.isoformat()
        except Exception:
            pass
        return None

    def _is_apscheduler_installed(self) -> bool:
        """Check if APScheduler is available without importing scheduler."""
        try:
            import apscheduler  # noqa
            return True
        except ImportError:
            return False


# ── Standalone runner (for testing / manual use) ───────────────────────────────

async def run_maintenance_now(config: dict) -> dict:
    """
    Convenience function to run maintenance once without starting a scheduler.
    Used by CLI caretaker maintenance command and tests.

    Usage:
        result = asyncio.run(run_maintenance_now(config))
    """
    from storage import local_db as _local_db

    # Try to get vector_db if available
    vector_db = None
    try:
        from storage.vector_db import VectorDB
        from pathlib import Path

        project_root  = Path(__file__).parent.parent
        chromadb_path = config.get("database", {}).get(
            "chromadb_path",
            str(project_root / "data" / "chromadb"),
        )
        vector_db = VectorDB(persist_directory=chromadb_path)
        vector_db.initialize()
    except Exception as e:
        logger.warning(f"[Scheduler] VectorDB not available for maintenance: {e}")

    from scheduler.nightly_maintenance import NightlyMaintenance
    runner = NightlyMaintenance(
        local_db=_local_db,
        vector_db=vector_db,
        config=config,
    )
    return await runner.run_all()