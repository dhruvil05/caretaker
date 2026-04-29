"""
scheduler/compression_queue.py
--------------------------------
Async queue for background compression jobs.

Compression NEVER blocks the MCP response path.
Every captured memory is queued here and processed in the background.

Flow:
  Capture Engine → enqueue(memory_id, type, full)
                       ↓  (async, non-blocking)
  Queue Worker   → compress_memory() → update SQLite + ChromaDB

Phase 2 — Compression Tribe / Queue Worker
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------

@dataclass
class CompressionJob:
    """A single compression job sitting in the queue."""
    memory_id:   str
    memory_type: str
    full_text:   str
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attempts:    int = 0

    def to_dict(self) -> dict:
        return {
            "memory_id":   self.memory_id,
            "memory_type": self.memory_type,
            "full_text":   self.full_text[:80] + "..." if len(self.full_text) > 80 else self.full_text,
            "enqueued_at": self.enqueued_at.isoformat(),
            "attempts":    self.attempts,
        }


# ---------------------------------------------------------------------------
# Completion callback type
# Called after each job finishes (success or failure)
# Signature: callback(memory_id, short, keywords, success)
# ---------------------------------------------------------------------------

CompletionCallback = Callable[
    [str, str, list[str], bool],
    Awaitable[None]
]


# ---------------------------------------------------------------------------
# Compression Queue
# ---------------------------------------------------------------------------

class CompressionQueue:
    """
    Async background queue for memory compression jobs.

    Usage:
        queue = CompressionQueue(api_key=..., on_complete=save_to_db)
        await queue.start()
        await queue.enqueue("mem-001", "PROJECT", "raw full text...")
        # ... later ...
        await queue.stop()
    """

    def __init__(
        self,
        api_key:       Optional[str] = None,
        model:         str = "claude-haiku-4-5-20251001",
        concurrency:   int = 2,
        max_retries:   int = 3,
        on_complete:   Optional[CompletionCallback] = None,
    ):
        """
        Args:
            api_key:     Anthropic API key (uses env var if None).
            model:       Haiku model string.
            concurrency: Max parallel compression jobs.
            max_retries: Max retry attempts per job.
            on_complete: Async callback called after each job.
                         Signature: (memory_id, short, keywords, success) -> None
        """
        self._api_key     = api_key
        self._model       = model
        self._concurrency = concurrency
        self._max_retries = max_retries
        self._on_complete = on_complete

        self._queue:    asyncio.Queue[CompressionJob] = asyncio.Queue()
        self._workers:  list[asyncio.Task] = []
        self._running:  bool = False

        # Stats
        self._processed: int = 0
        self._failed:    int = 0
        self._enqueued:  int = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        """Start background worker tasks."""
        if self._running:
            logger.warning("compression_queue: already running")
            return

        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(worker_id=i))
            for i in range(self._concurrency)
        ]
        logger.info(
            "compression_queue: started %d worker(s)", self._concurrency
        )

    async def stop(self, wait: bool = True) -> None:
        """
        Stop the queue gracefully.

        Args:
            wait: If True, wait for all queued jobs to finish before stopping.
        """
        if not self._running:
            return

        self._running = False

        if wait:
            # Drain the queue before stopping
            await self._queue.join()

        # Cancel worker tasks
        for task in self._workers:
            task.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

        logger.info(
            "compression_queue: stopped. processed=%d failed=%d",
            self._processed, self._failed,
        )

    async def enqueue(
        self,
        memory_id:   str,
        memory_type: str,
        full_text:   str,
    ) -> None:
        """
        Add a memory to the compression queue (non-blocking).

        Args:
            memory_id:   UUID of the memory.
            memory_type: Memory TYPE (PROJECT, PREFERENCE, etc.).
            full_text:   Raw FULL memory text.
        """
        job = CompressionJob(
            memory_id=memory_id,
            memory_type=memory_type,
            full_text=full_text,
        )
        await self._queue.put(job)
        self._enqueued += 1

        logger.debug(
            "compression_queue: enqueued job for %s (queue size=%d)",
            memory_id, self._queue.qsize(),
        )

    def enqueue_nowait(
        self,
        memory_id:   str,
        memory_type: str,
        full_text:   str,
    ) -> None:
        """
        Non-async version of enqueue. Safe to call from sync code.
        Raises QueueFull if queue is full (rarely happens).
        """
        job = CompressionJob(
            memory_id=memory_id,
            memory_type=memory_type,
            full_text=full_text,
        )
        self._queue.put_nowait(job)
        self._enqueued += 1

    @property
    def queue_size(self) -> int:
        """Current number of jobs waiting in queue."""
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        """Return current queue statistics."""
        return {
            "running":   self._running,
            "enqueued":  self._enqueued,
            "processed": self._processed,
            "failed":    self._failed,
            "queued":    self._queue.qsize(),
        }

    # -----------------------------------------------------------------------
    # Internal worker
    # -----------------------------------------------------------------------

    async def _worker(self, worker_id: int) -> None:
        """
        Background worker loop. Pulls jobs from queue and compresses them.
        Runs until stop() is called.
        """
        from compression.compressor import compress_memory

        logger.debug("compression_queue: worker-%d started", worker_id)

        while self._running or not self._queue.empty():
            try:
                # Wait for a job (timeout allows checking _running flag)
                try:
                    job: CompressionJob = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                job.attempts += 1

                logger.info(
                    "compression_queue: worker-%d processing %s (attempt %d)",
                    worker_id, job.memory_id, job.attempts,
                )

                # Run compression
                result = await compress_memory(
                    memory_id=job.memory_id,
                    memory_type=job.memory_type,
                    full_text=job.full_text,
                    api_key=self._api_key,
                    model=self._model,
                    max_retries=self._max_retries,
                )

                if result.success:
                    self._processed += 1
                    logger.info(
                        "compression_queue: worker-%d SUCCESS %s",
                        worker_id, job.memory_id,
                    )
                else:
                    self._failed += 1
                    logger.error(
                        "compression_queue: worker-%d FAILED %s: %s",
                        worker_id, job.memory_id, result.error,
                    )

                # Fire completion callback (saves to DB + ChromaDB)
                if self._on_complete:
                    try:
                        await self._on_complete(
                            result.memory_id,
                            result.short,
                            result.keywords,
                            result.success,
                        )
                    except Exception as cb_err:
                        logger.error(
                            "compression_queue: completion callback error: %s", cb_err
                        )

                self._queue.task_done()

            except asyncio.CancelledError:
                logger.debug("compression_queue: worker-%d cancelled", worker_id)
                break
            except Exception as exc:
                logger.error(
                    "compression_queue: worker-%d unexpected error: %s",
                    worker_id, exc,
                )
                # Mark task done to avoid blocking queue.join()
                try:
                    self._queue.task_done()
                except ValueError:
                    pass

        logger.debug("compression_queue: worker-%d stopped", worker_id)


# ---------------------------------------------------------------------------
# Module-level singleton (shared across the app)
# ---------------------------------------------------------------------------

_global_queue: Optional[CompressionQueue] = None


def get_queue() -> CompressionQueue:
    """Return the global compression queue instance."""
    global _global_queue
    if _global_queue is None:
        raise RuntimeError(
            "compression_queue: global queue not initialised. "
            "Call init_queue() first."
        )
    return _global_queue


def init_queue(
    api_key:     Optional[str] = None,
    model:       str = "claude-haiku-4-5-20251001",
    concurrency: int = 2,
    on_complete: Optional[CompletionCallback] = None,
) -> CompressionQueue:
    """
    Initialise the global compression queue.
    Call once at server startup.

    Args:
        api_key:     Anthropic API key.
        model:       Haiku model string.
        concurrency: Max parallel workers.
        on_complete: Async callback after each job completes.

    Returns:
        The initialised CompressionQueue instance.
    """
    global _global_queue
    _global_queue = CompressionQueue(
        api_key=api_key,
        model=model,
        concurrency=concurrency,
        on_complete=on_complete,
    )
    logger.info("compression_queue: global queue initialised")
    return _global_queue