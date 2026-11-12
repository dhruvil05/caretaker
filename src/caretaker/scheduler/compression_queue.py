"""
scheduler/compression_queue.py
Async compression queue — compression NEVER blocks the MCP response.
Flow:
  1. Capture engine writes memory with status=PENDING_COMPRESSION to SQLite
  2. Adds job to this queue
  3. Worker picks job, runs compressor, updates SQLite + ChromaDB
  4. Status → ACTIVE

0.5: the queue no longer needs the vector index (the default install has
none), and on start it picks up memories still waiting for a summary — jobs
live only in memory, so they are lost whenever the AI app closes.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CompressionJob:
    memory_id: str
    full_text: str
    memory_type: str
    retry_count: int = 0
    created_at: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()


class CompressionQueue:
    """
    Async FIFO queue for background compression jobs.
    - Max 3 retries with exponential backoff on failure
    - Never blocks MCP get_context or save_message calls
    - Single worker coroutine processes jobs one at a time
    """

    MAX_RETRIES = 3
    BASE_BACKOFF = 2.0   # seconds
    MAX_QUEUE_SIZE = 500
    BACKLOG_BATCH = 200   # memories picked up per server start

    def __init__(self, compressor, local_db, vector_db=None):
        self.compressor = compressor
        self.local_db = local_db
        self.vector_db = vector_db          # optional: None without the semantic extra
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._pending: set = set()          # memory ids queued or in progress
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        # Event loop that owns the queue and the worker (set in start()).
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self):
        """Start the background worker. Call once on server startup."""
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("[CompressionQueue] Worker started.")

    async def stop(self):
        """Graceful shutdown — waits for queue to drain."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("[CompressionQueue] Worker stopped.")

    def submit(self, memory_id: str, full_text: str, memory_type: str) -> None:
        """
        Thread-safe, synchronous way to add a job. Safe to call from any
        thread — e.g. an MCP tool running in FastMCP's sync-tool threadpool —
        while the worker runs on a different thread's event loop.

        asyncio.Queue is not thread-safe, so when called from a thread other
        than the owning loop's, the put is marshalled onto that loop with
        call_soon_threadsafe (which also wakes the loop immediately).
        """
        job = CompressionJob(
            memory_id=memory_id,
            full_text=full_text,
            memory_type=memory_type,
        )
        loop = self._loop
        if loop is None or loop.is_closed():
            # Worker never started (e.g. tests / sync context) — no other
            # loop is touching the queue, so a direct put is safe.
            self._put_job(job)
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._put_job(job)
        else:
            loop.call_soon_threadsafe(self._put_job, job)

    def enqueue_backlog(self, limit: Optional[int] = None) -> int:
        """Queue memories that are still waiting for a summary (oldest jobs are
        lost when the server stops). Returns how many were queued."""
        try:
            rows = self.local_db.get_unsummarised(limit or self.BACKLOG_BATCH)
        except Exception as e:
            logger.warning(f"[CompressionQueue] Could not read the summary backlog: {e}")
            return 0
        for row in rows:
            self.submit(row["id"], row["full"], row["type"] or "LEARNING")
        if rows:
            logger.info(f"[CompressionQueue] Queued {len(rows)} memories still waiting for a summary.")
        return len(rows)

    async def enqueue(self, memory_id: str, full_text: str, memory_type: str):
        """
        Add a compression job to the queue (async API, kept for callers
        already on the owning loop). Thread-safe via submit().
        """
        self.submit(memory_id, full_text, memory_type)

    def _put_job(self, job: CompressionJob) -> None:
        """Put a job on the queue. Must run on the owning loop's thread."""
        memory_id = job.memory_id
        if job.retry_count == 0 and memory_id in self._pending:
            return   # already queued (e.g. by both a save and the backlog)
        try:
            self._queue.put_nowait(job)
            self._pending.add(memory_id)
            logger.debug(f"[CompressionQueue] Enqueued job for memory_id={memory_id}")
        except asyncio.QueueFull:
            self._pending.discard(memory_id)   # let a later start pick it up
            logger.warning(
                f"[CompressionQueue] Queue full ({self.MAX_QUEUE_SIZE}). "
                f"Skipping compression for memory_id={memory_id}"
            )

    async def _worker_loop(self):
        """Main worker loop — runs forever until stopped."""
        while self._running:
            try:
                job: CompressionJob = await asyncio.wait_for(
                    self._queue.get(), timeout=5.0
                )
                try:
                    await self._process_job(job)
                finally:
                    self._queue.task_done()
            except asyncio.TimeoutError:
                continue  # No jobs, keep waiting
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[CompressionQueue] Unexpected error in worker: {e}")

    async def _process_job(self, job: CompressionJob):
        """
        Run compression for one job.
        On failure: retry up to MAX_RETRIES with exponential backoff.
        """
        try:
            logger.debug(f"[CompressionQueue] Processing memory_id={job.memory_id}, type={job.memory_type}")

            # Run compression in thread pool to avoid blocking event loop.
            # Safe here: we're always called from inside the running worker
            # task, so get_running_loop() is guaranteed to succeed.
            loop = asyncio.get_running_loop()
            # 0.5: memories saved before sensitive-data protection may still
            # hold secrets; never send those to a (possibly remote) summariser.
            from caretaker.privacy import protect
            safe_text = protect(job.full_text).text
            short, keywords = await loop.run_in_executor(
                None,
                self.compressor.compress,
                safe_text,
                job.memory_type,
            )

            if not short:
                raise ValueError("Compression returned empty SHORT")

            # Update SQLite: set short, keywords, status=ACTIVE
            self.local_db.update_compression(
                memory_id=job.memory_id,
                short=short,
                keywords=keywords,
            )

            # BUG FIX: Fetch full SQLite record so ChromaDB metadata stays in sync
            # Previously: upsert used default temperature="HOT" always — causing filter mismatches
            record = self.local_db.get_memory_by_id(job.memory_id)
            temperature      = (record or {}).get("temperature", "HOT")
            memory_type_real = (record or {}).get("type", job.memory_type)
            importance_score = float((record or {}).get("importance") or 0.5)

            # Update ChromaDB index with SHORT embedding + correct metadata
            if self.vector_db is not None:
                self.vector_db.upsert(
                    memory_id=job.memory_id,
                    short=short,
                    keywords=keywords,
                    temperature=temperature,
                    memory_type=memory_type_real,
                    importance_score=importance_score,
                )

            self._pending.discard(job.memory_id)
            logger.info(f"[CompressionQueue] Compressed memory_id={job.memory_id} — short={short[:60]!r}")

        except Exception as e:
            logger.warning(
                f"[CompressionQueue] Compression failed for memory_id={job.memory_id} "
                f"(attempt {job.retry_count + 1}/{self.MAX_RETRIES}): {e}"
            )

            if job.retry_count < self.MAX_RETRIES - 1:
                # Exponential backoff before re-queue
                backoff = self.BASE_BACKOFF ** (job.retry_count + 1)
                logger.info(f"[CompressionQueue] Retrying in {backoff:.1f}s...")
                await asyncio.sleep(backoff)

                # Re-queue this same job so retry_count is kept (a new job
                # would start at 0 again and retry forever).
                job.retry_count += 1
                self._put_job(job)
            else:
                self._pending.discard(job.memory_id)
                # Final failure — mark memory as ACTIVE with raw text only (no SHORT)
                logger.error(
                    f"[CompressionQueue] All retries exhausted for memory_id={job.memory_id}. "
                    f"Memory stored without compression."
                )
                self.local_db.update_status(job.memory_id, "ACTIVE")

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()