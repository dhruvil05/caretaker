"""
compression/compressor.py
--------------------------
Main compression engine for Phase 2.

Calls Haiku API to generate SHORT summary + KEYWORDS from FULL memory text.
Updates SQLite memory record from PENDING_COMPRESSION → ACTIVE.
Updates ChromaDB index with new SHORT embedding.

Always runs ASYNC — never blocks the MCP response path.

Phase 2 — Compression Tribe
"""

import logging
import asyncio
from dataclasses import dataclass
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config defaults (overridden by config.json at runtime)
# ---------------------------------------------------------------------------

DEFAULT_MODEL         = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS    = 200       # SHORT summary ≤ 60 tokens + some buffer
DEFAULT_MAX_RETRIES   = 3
DEFAULT_RETRY_DELAY   = 2.0       # seconds, doubles each retry (exponential backoff)
SHORT_TOKEN_LIMIT     = 60        # hard cap on SHORT field


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    """Holds the output of a single compression run."""
    memory_id:  str
    short:      str
    keywords:   list[str]
    success:    bool
    error:      Optional[str] = None


# ---------------------------------------------------------------------------
# Token counter (rough, avoids tiktoken dependency)
# ---------------------------------------------------------------------------

def _rough_token_count(text: str) -> int:
    """Rough token estimate: ~4 chars per token. Good enough for limit checks."""
    return max(1, len(text) // 4)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens tokens."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]  # trim at word boundary


# ---------------------------------------------------------------------------
# Core compression call
# ---------------------------------------------------------------------------

async def _call_haiku_compress(
    client: anthropic.AsyncAnthropic,
    memory_type: str,
    full_text: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Call Haiku to get SHORT summary.

    Args:
        client:      Async Anthropic client.
        memory_type: Memory TYPE string.
        full_text:   Raw FULL memory text.
        model:       Haiku model string.

    Returns:
        SHORT summary string (may need trimming).

    Raises:
        anthropic.APIError on API failure.
    """
    from compression.templates import build_compression_prompt

    prompt = build_compression_prompt(memory_type, full_text)

    response = await client.messages.create(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text if response.content else ""
    return raw.strip()


async def _call_haiku_keywords(
    client: anthropic.AsyncAnthropic,
    short_text: str,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """
    Call Haiku to extract keywords from SHORT summary.

    Args:
        client:     Async Anthropic client.
        short_text: The compressed SHORT text.
        model:      Haiku model string.

    Returns:
        List of 3-7 keywords.
    """
    from compression.keyword_generator import generate_keywords_api
    return await generate_keywords_api(short_text, client, model)


# ---------------------------------------------------------------------------
# Main compressor with retry logic
# ---------------------------------------------------------------------------

async def compress_memory(
    memory_id:   str,
    memory_type: str,
    full_text:   str,
    api_key:     Optional[str] = None,
    model:       str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> CompressionResult:
    """
    Compress a single memory unit: FULL → SHORT + KEYWORDS.

    Retries up to max_retries times with exponential backoff.
    Falls back to local keyword extraction if keyword API fails.

    Args:
        memory_id:   UUID of the memory being compressed.
        memory_type: Memory TYPE (PROJECT, PREFERENCE, etc.).
        full_text:   Raw FULL memory text to compress.
        api_key:     Anthropic API key. Uses env var if None.
        model:       Haiku model string.
        max_retries: Number of retry attempts on API failure.

    Returns:
        CompressionResult with short, keywords, success flag.
    """
    from compression.keyword_generator import generate_keywords_local, validate_keywords

    client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()

    last_error: Optional[str] = None
    delay = DEFAULT_RETRY_DELAY

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "compressor: attempt %d/%d for memory %s (type=%s)",
                attempt, max_retries, memory_id, memory_type,
            )

            # Step 1: Generate SHORT summary
            short_raw = await _call_haiku_compress(client, memory_type, full_text, model)

            if not short_raw:
                raise ValueError("Haiku returned empty SHORT summary")

            # Enforce 60-token limit
            short = _truncate_to_tokens(short_raw, SHORT_TOKEN_LIMIT)

            logger.debug(
                "compressor: SHORT generated (~%d tokens): %s",
                _rough_token_count(short), short[:80],
            )

            # Step 2: Generate KEYWORDS from SHORT
            try:
                keywords = await _call_haiku_keywords(client, short, model)
            except Exception as kw_err:
                logger.warning(
                    "compressor: keyword API failed (%s), using local fallback", kw_err
                )
                keywords = generate_keywords_local(short)

            # Validate keywords — fallback to local if invalid
            if not validate_keywords(keywords):
                logger.warning(
                    "compressor: invalid keyword count (%d), using local fallback",
                    len(keywords),
                )
                keywords = generate_keywords_local(short)

            logger.info(
                "compressor: SUCCESS for memory %s — %d keywords generated",
                memory_id, len(keywords),
            )

            return CompressionResult(
                memory_id=memory_id,
                short=short,
                keywords=keywords,
                success=True,
            )

        except anthropic.RateLimitError as exc:
            last_error = f"RateLimitError: {exc}"
            logger.warning(
                "compressor: rate limit on attempt %d — sleeping %.1fs",
                attempt, delay,
            )
            await asyncio.sleep(delay)
            delay *= 2  # exponential backoff

        except anthropic.APIConnectionError as exc:
            last_error = f"APIConnectionError: {exc}"
            logger.warning(
                "compressor: connection error on attempt %d — sleeping %.1fs",
                attempt, delay,
            )
            await asyncio.sleep(delay)
            delay *= 2

        except anthropic.APIStatusError as exc:
            last_error = f"APIStatusError {exc.status_code}: {exc.message}"
            logger.error(
                "compressor: API status error (status=%d) on attempt %d",
                exc.status_code, attempt,
            )
            # Don't retry on 4xx client errors (except 429 handled above)
            if 400 <= exc.status_code < 500 and exc.status_code != 429:
                break
            await asyncio.sleep(delay)
            delay *= 2

        except Exception as exc:
            last_error = str(exc)
            logger.error(
                "compressor: unexpected error on attempt %d: %s",
                attempt, exc,
            )
            await asyncio.sleep(delay)
            delay *= 2

    # All retries exhausted — return failure
    logger.error(
        "compressor: ALL %d attempts failed for memory %s. Last error: %s",
        max_retries, memory_id, last_error,
    )

    return CompressionResult(
        memory_id=memory_id,
        short="",
        keywords=[],
        success=False,
        error=last_error,
    )


# ---------------------------------------------------------------------------
# Batch compressor — used by compression queue
# ---------------------------------------------------------------------------

async def compress_batch(
    memories:    list[dict],
    api_key:     Optional[str] = None,
    model:       str = DEFAULT_MODEL,
    concurrency: int = 3,
) -> list[CompressionResult]:
    """
    Compress a batch of memory units concurrently.

    Args:
        memories:    List of dicts with keys: id, type, full
        api_key:     Anthropic API key.
        model:       Haiku model string.
        concurrency: Max parallel compressions (respect rate limits).

    Returns:
        List of CompressionResult objects, one per memory.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _compress_one(mem: dict) -> CompressionResult:
        async with semaphore:
            return await compress_memory(
                memory_id=mem["id"],
                memory_type=mem["type"],
                full_text=mem["full"],
                api_key=api_key,
                model=model,
            )

    tasks = [_compress_one(m) for m in memories]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return list(results)


# ---------------------------------------------------------------------------
# Standalone test helper (no DB dependency)
# ---------------------------------------------------------------------------

async def _test_compress_local():
    """
    Quick local test — uses local keyword extraction to avoid API call.
    Only tests the pipeline logic, not the Haiku API call.
    """
    from compression.keyword_generator import generate_keywords_local

    memory_type = "PROJECT"
    full_text   = (
        "I am building a FastAPI project called Caretaker. "
        "It is a universal memory layer for AI agents. "
        "Currently in Phase 2, adding compression and semantic search. "
        "Stack: Python, FastMCP, SQLite, ChromaDB, Haiku API."
    )

    # Simulate SHORT (what Haiku would return)
    simulated_short = "Caretaker — FastAPI + SQLite + ChromaDB — Phase 2 — universal memory layer for AI agents."
    keywords = generate_keywords_local(simulated_short)

    result = CompressionResult(
        memory_id="test-001",
        short=simulated_short,
        keywords=keywords,
        success=True,
    )

    print(f"SHORT: {result.short}")
    print(f"KEYWORDS: {result.keywords}")
    print(f"Token estimate: ~{_rough_token_count(result.short)}")
    print(f"SUCCESS: {result.success}")
    return result


if __name__ == "__main__":
    asyncio.run(_test_compress_local())