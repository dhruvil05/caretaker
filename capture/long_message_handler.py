"""
capture/long_message_handler.py
Handles messages that exceed the long message threshold (default: 400 tokens).
Two strategies:
  SPLIT   — break into logical chunks, each becomes its own memory
  COMPRESS — treat as single memory, compress to SHORT via compressor
Choice is determined by message structure (paragraphs vs continuous prose).
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# 1 token ≈ 4 characters (conservative estimate)
CHARS_PER_TOKEN = 4
DEFAULT_LONG_THRESHOLD_TOKENS = 400
MAX_CHUNK_TOKENS = 300  # Max tokens per split chunk


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def is_long_message(text: str, threshold_tokens: int = DEFAULT_LONG_THRESHOLD_TOKENS) -> bool:
    """Return True if text exceeds the long message threshold."""
    return estimate_tokens(text) >= threshold_tokens


def handle_long_message(
    text: str,
    memory_type: str,
    threshold_tokens: int = DEFAULT_LONG_THRESHOLD_TOKENS,
) -> Tuple[str, List[str]]:
    """
    Decide how to handle a long message.

    Returns:
        (strategy: str, chunks: List[str])
        strategy = "SPLIT" | "COMPRESS" | "PASSTHROUGH"
        chunks = list of text segments to process independently (SPLIT)
                 OR list with one item (COMPRESS / PASSTHROUGH)
    """
    if not is_long_message(text, threshold_tokens):
        return ("PASSTHROUGH", [text])

    token_count = estimate_tokens(text)
    logger.info(f"[LongMsgHandler] Long message detected: ~{token_count} tokens. Type: {memory_type}")

    # Detect if message has natural paragraph/section breaks
    chunks = _try_paragraph_split(text)

    if len(chunks) >= 2:
        logger.info(f"[LongMsgHandler] Strategy: SPLIT into {len(chunks)} chunks")
        return ("SPLIT", chunks)
    else:
        logger.info("[LongMsgHandler] Strategy: COMPRESS (no natural splits found)")
        return ("COMPRESS", [text])


def _try_paragraph_split(text: str) -> List[str]:
    """
    Try to split text into logical chunks by paragraph breaks.
    Returns chunks only if they are meaningfully sized.
    """
    # Split on double newlines (paragraph breaks) or numbered sections
    raw_chunks = re.split(r'\n\s*\n|\n(?=\d+[\.\)]\s)', text.strip())

    # Filter: keep chunks with enough content
    meaningful = [
        chunk.strip()
        for chunk in raw_chunks
        if len(chunk.strip()) > 50  # At least 50 chars
    ]

    # Merge tiny chunks with previous
    merged = _merge_small_chunks(meaningful, min_tokens=50)

    # Enforce max chunk size — split oversized chunks at sentence boundary
    final = []
    for chunk in merged:
        if estimate_tokens(chunk) > MAX_CHUNK_TOKENS:
            sub_chunks = _split_at_sentences(chunk, max_tokens=MAX_CHUNK_TOKENS)
            final.extend(sub_chunks)
        else:
            final.append(chunk)

    return final


def _merge_small_chunks(chunks: List[str], min_tokens: int = 50) -> List[str]:
    """Merge chunks that are too small with the previous chunk."""
    if not chunks:
        return []

    merged = [chunks[0]]
    for chunk in chunks[1:]:
        if estimate_tokens(chunk) < min_tokens and merged:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)
    return merged


def _split_at_sentences(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> List[str]:
    """Split oversized text at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if estimate_tokens(candidate) > max_tokens and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current.strip())

    return chunks if chunks else [text]


def process_long_message(
    text: str,
    memory_type: str,
    compressor,
    threshold_tokens: int = DEFAULT_LONG_THRESHOLD_TOKENS,
) -> List[dict]:
    """
    Full pipeline for long message handling.
    Returns list of memory-ready dicts, each with:
      - full_text: str
      - short: str
      - keywords: List[str]
      - memory_type: str
      - chunk_index: int (for SPLIT strategy)
    """
    strategy, chunks = handle_long_message(text, memory_type, threshold_tokens)

    results = []

    if strategy == "PASSTHROUGH":
        short, keywords = compressor.compress(text, memory_type)
        results.append({
            "full_text": text,
            "short": short,
            "keywords": keywords,
            "memory_type": memory_type,
            "chunk_index": 0,
        })

    elif strategy == "SPLIT":
        for i, chunk in enumerate(chunks):
            short, keywords = compressor.compress(chunk, memory_type)
            results.append({
                "full_text": chunk,
                "short": short,
                "keywords": keywords,
                "memory_type": memory_type,
                "chunk_index": i,
            })

    elif strategy == "COMPRESS":
        # Compress the whole thing — SHORT will summarize everything
        short, keywords = compressor.compress(text, memory_type)
        results.append({
            "full_text": text,
            "short": short,
            "keywords": keywords,
            "memory_type": memory_type,
            "chunk_index": 0,
        })

    return results