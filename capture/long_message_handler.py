"""
capture/long_message_handler.py
--------------------------------
Handles long messages (>400 tokens) before they enter the memory pipeline.

Two strategies depending on message content:

  SPLIT  — Message covers 2+ distinct topics.
           Each topic becomes its own memory unit (separate id, type, keywords).
           P2-T12: 600-token 2-topic message → 2 memory units.

  COMPRESS — Message covers 1 focused topic but is too long.
             Single memory unit with SHORT ≤300 tokens.
             P2-T13: 600-token 1-topic message → compressed ≤300 tokens.

Decision flow:
  1. Token count message
  2. If ≤ TOKEN_THRESHOLD → pass through unchanged (not long)
  3. If > TOKEN_THRESHOLD → detect topics
  4. If multi-topic (≥2 distinct topics) → SPLIT
  5. If single-topic → COMPRESS

Topic detection:
  - Uses paragraph/sentence boundary detection
  - Keyword shift analysis between segments
  - Falls back to COMPRESS if detection is uncertain

Phase 2 — Capture
"""

import re
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOKEN_THRESHOLD:      int   = 400    # messages above this are "long"
COMPRESS_MAX_TOKENS:  int   = 280    # target max for compressed single-topic
TOPIC_SHIFT_THRESHOLD: float = 0.25  # min keyword dissimilarity to split topics
MIN_SEGMENT_TOKENS:   int   = 80     # minimum tokens per split segment

# ---------------------------------------------------------------------------
# Token counter (rough, no external dependency)
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Rough token count: word count * 1.3 (BPE estimate)."""
    words = len(text.strip().split())
    return max(1, int(words * 1.3))


def _truncate(text: str, max_tokens: int) -> str:
    """Hard-truncate text to max_tokens at word boundary."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


# ---------------------------------------------------------------------------
# Handler result
# ---------------------------------------------------------------------------

@dataclass
class HandlerResult:
    """
    Output of long message handling.

    segments: List of text segments ready to enter memory pipeline.
             Each segment is one memory unit.
    strategy: 'passthrough' | 'split' | 'compress'
    topic_count: Number of distinct topics detected.
    original_tokens: Token count of original message.
    """
    segments:        list[str]
    strategy:        str
    topic_count:     int
    original_tokens: int
    segment_tokens:  list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.segment_tokens:
            self.segment_tokens = [_count_tokens(s) for s in self.segments]


# ---------------------------------------------------------------------------
# Topic detector
# ---------------------------------------------------------------------------

def _split_into_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraph-level chunks.
    Splits on: double newline, numbered list items, or sentence groups.
    """
    # Split on double newline first
    parts = re.split(r"\n{2,}", text.strip())

    # If no paragraph breaks, split on sentence boundaries (every ~3 sentences)
    if len(parts) == 1:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunk_size = max(2, len(sentences) // 3)
        parts = []
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i + chunk_size]).strip()
            if chunk:
                parts.append(chunk)

    return [p.strip() for p in parts if p.strip()]


def _extract_keywords_simple(text: str, top_n: int = 8) -> set[str]:
    """
    Fast local keyword extraction for topic shift detection.
    No external dependencies.
    """
    stop = {
        "a","an","the","and","or","but","in","on","at","to","for","of","with",
        "by","is","are","was","were","i","my","we","you","it","this","that",
        "be","been","have","has","had","do","does","did","will","would","can",
        "could","should","also","just","very","so","then","if","when","about",
    }
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())
    scored = {}
    for t in tokens:
        if t not in stop:
            scored[t] = scored.get(t, 0) + 1

    return set(sorted(scored, key=lambda x: -scored[x])[:top_n])


def _jaccard_distance(set_a: set, set_b: set) -> float:
    """
    Jaccard distance between two keyword sets.
    0.0 = identical, 1.0 = completely different.
    """
    if not set_a or not set_b:
        return 1.0
    union = set_a | set_b
    inter = set_a & set_b
    return 1.0 - len(inter) / len(union)


def detect_topics(text: str) -> list[str]:
    """
    Detect distinct topics in a long message.

    Returns list of text segments, one per distinct topic.
    If single topic detected → returns list with one item (full text).

    P2-T12: 2-topic message → 2 segments.
    P2-T13: 1-topic message → 1 segment.

    Args:
        text: Input message text.

    Returns:
        List of topic-aligned text segments (1 or more).
    """
    paragraphs = _split_into_paragraphs(text)

    if len(paragraphs) <= 1:
        return [text.strip()]

    # Calculate keyword sets per paragraph
    kw_sets = [_extract_keywords_simple(p) for p in paragraphs]

    # Group paragraphs into topics by keyword similarity
    topics:    list[list[str]] = [[paragraphs[0]]]
    topic_kws: list[set]       = [kw_sets[0]]

    for i in range(1, len(paragraphs)):
        para = paragraphs[i]
        kws  = kw_sets[i]

        # Compare against current topic's keywords
        dist = _jaccard_distance(topic_kws[-1], kws)

        # Skip very short paragraphs (transitional sentences)
        if _count_tokens(para) < 20:
            topics[-1].append(para)
            # Update topic keywords
            topic_kws[-1] = topic_kws[-1] | kws
            continue

        if dist >= TOPIC_SHIFT_THRESHOLD:
            # New topic detected
            topics.append([para])
            topic_kws.append(kws)
        else:
            # Same topic — append to current
            topics[-1].append(para)
            topic_kws[-1] = topic_kws[-1] | kws

    # Merge tiny segments into neighbours
    merged: list[str] = []
    for topic_parts in topics:
        segment = "\n\n".join(topic_parts).strip()
        if _count_tokens(segment) < MIN_SEGMENT_TOKENS and merged:
            # Too small — merge into previous
            merged[-1] = merged[-1] + "\n\n" + segment
        else:
            merged.append(segment)

    return merged if merged else [text.strip()]


# ---------------------------------------------------------------------------
# Compressor (single-topic, reduces to ≤300 tokens)
# ---------------------------------------------------------------------------

async def _compress_segment(
    text:       str,
    api_client=None,
    max_tokens: int = COMPRESS_MAX_TOKENS,
) -> str:
    """
    Compress a single-topic segment to ≤300 tokens.

    Uses Haiku API if client provided, falls back to extractive compression.

    Args:
        text:       Input segment text.
        api_client: Optional Anthropic async client.
        max_tokens: Target max token count.

    Returns:
        Compressed text ≤ max_tokens tokens.
    """
    if _count_tokens(text) <= max_tokens:
        return text.strip()

    # Haiku API compression
    if api_client:
        try:
            prompt = (
                f"Compress the following text to under {max_tokens * 4} characters. "
                f"Keep ALL facts, names, numbers, decisions. Remove filler and repetition. "
                f"Output ONLY the compressed text, nothing else.\n\n"
                f"TEXT:\n{text}\n\nCOMPRESSED:"
            )
            response = await api_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tokens + 50,
                messages=[{"role": "user", "content": prompt}],
            )
            compressed = response.content[0].text.strip() if response.content else ""
            if compressed and _count_tokens(compressed) <= max_tokens:
                logger.debug(
                    "long_message_handler: Haiku compressed %d→%d tokens",
                    _count_tokens(text), _count_tokens(compressed),
                )
                return compressed
        except Exception as exc:
            logger.warning(
                "long_message_handler: Haiku compression failed (%s) — extractive fallback",
                exc,
            )

    # Extractive fallback — keep highest-information sentences
    return _extractive_compress(text, max_tokens)


def _extractive_compress(text: str, max_tokens: int) -> str:
    """
    Simple extractive compression: score sentences by keyword density,
    keep highest-scoring until token budget fills.

    Args:
        text:       Input text.
        max_tokens: Token budget.

    Returns:
        Compressed text with most important sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    if not sentences:
        return _truncate(text, max_tokens)

    # Score sentences: keyword density + position bonus
    scored = []
    global_kws = _extract_keywords_simple(text, top_n=15)

    for i, sent in enumerate(sentences):
        tokens = set(re.findall(r"[a-zA-Z]{3,}", sent.lower()))
        kw_hits = len(tokens & global_kws)
        pos_bonus = 1.2 if i == 0 or i == len(sentences) - 1 else 1.0
        score = (kw_hits / max(1, len(tokens))) * pos_bonus
        scored.append((score, i, sent))

    # Sort by score, keep highest
    scored.sort(key=lambda x: -x[0])

    selected_indices = set()
    token_count = 0

    for score, idx, sent in scored:
        est = _count_tokens(sent)
        if token_count + est <= max_tokens:
            selected_indices.add(idx)
            token_count += est
        if token_count >= max_tokens:
            break

    # Reconstruct in original order
    result = " ".join(
        sent for i, (_, idx, sent) in enumerate(
            sorted(scored, key=lambda x: x[1])
        )
        if idx in selected_indices
    )

    return result.strip() or _truncate(text, max_tokens)


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

async def handle_long_message(
    text:       str,
    api_client=None,
) -> HandlerResult:
    """
    Main entry point for long message handling.

    Decision tree:
      ≤400 tokens → passthrough
      >400 tokens, multi-topic → SPLIT
      >400 tokens, single-topic → COMPRESS

    P2-T12: 600-token 2-topic → 2 memory units.
    P2-T13: 600-token 1-topic → compressed ≤300 tokens.

    Args:
        text:       Raw message text from user or agent.
        api_client: Optional Anthropic async client for Haiku compression.

    Returns:
        HandlerResult with segments and strategy metadata.
    """
    original_tokens = _count_tokens(text)

    # Pass through short messages unchanged
    if original_tokens <= TOKEN_THRESHOLD:
        logger.debug(
            "long_message_handler: passthrough (%d tokens ≤ %d)",
            original_tokens, TOKEN_THRESHOLD,
        )
        return HandlerResult(
            segments=[text.strip()],
            strategy="passthrough",
            topic_count=1,
            original_tokens=original_tokens,
        )

    logger.info(
        "long_message_handler: long message detected (%d tokens > %d)",
        original_tokens, TOKEN_THRESHOLD,
    )

    # Detect topics
    topic_segments = detect_topics(text)
    topic_count    = len(topic_segments)

    logger.info(
        "long_message_handler: detected %d topic(s)", topic_count
    )

    # SPLIT — multiple distinct topics
    if topic_count >= 2:
        logger.info("long_message_handler: SPLIT strategy → %d segments", topic_count)
        return HandlerResult(
            segments=topic_segments,
            strategy="split",
            topic_count=topic_count,
            original_tokens=original_tokens,
        )

    # COMPRESS — single topic, too long
    compressed = await _compress_segment(
        text=topic_segments[0],
        api_client=api_client,
        max_tokens=COMPRESS_MAX_TOKENS,
    )

    logger.info(
        "long_message_handler: COMPRESS strategy → %d tokens (was %d)",
        _count_tokens(compressed), original_tokens,
    )

    return HandlerResult(
        segments=[compressed],
        strategy="compress",
        topic_count=1,
        original_tokens=original_tokens,
    )


# ---------------------------------------------------------------------------
# Sync wrapper (for non-async contexts)
# ---------------------------------------------------------------------------

def handle_long_message_sync(text: str) -> HandlerResult:
    """
    Sync version of handle_long_message — no API compression.
    Uses extractive compression only.

    Args:
        text: Raw message text.

    Returns:
        HandlerResult with segments.
    """
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            handle_long_message(text, api_client=None)
        )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Utility: generate segment IDs
# ---------------------------------------------------------------------------

def make_segment_ids(count: int, base_id: Optional[str] = None) -> list[str]:
    """
    Generate UUIDs for split segments.

    Args:
        count:   Number of IDs to generate.
        base_id: Optional base UUID (ignored, always generates fresh UUIDs).

    Returns:
        List of UUID strings.
    """
    return [str(uuid.uuid4()) for _ in range(count)]