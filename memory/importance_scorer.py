"""
memory/importance_scorer.py
----------------------------
Assigns an importance score (0.0 – 1.0) to a memory unit at capture time.

Score is based on:
  - Memory TYPE          (some types are inherently more important)
  - FACT_TYPE            (REPLACEABLE facts score higher — they define current state)
  - Content signals      (keywords, length, specificity)
  - Explicit markers     (user corrections, decisions score higher)

Score drives:
  - Initial temperature assignment (via temperature_engine.py)
  - Nightly decay baseline
  - Retrieval ranking weight

Phase 2 — Memory Brain
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base importance scores by TYPE
# Higher = more likely to be needed in retrieval
# ---------------------------------------------------------------------------

TYPE_BASE_SCORES: dict[str, float] = {
    "PROJECT":    0.75,   # Core — almost always relevant
    "CORRECTION": 0.80,   # Highest — user explicitly fixed something
    "DECISION":   0.70,   # High — choices shape everything downstream
    "PROBLEM":    0.65,   # High — blockers need to be remembered
    "PERSONAL":   0.60,   # Medium-high — identity info used frequently
    "PREFERENCE": 0.55,   # Medium — style/tool choices
    "LEARNING":   0.50,   # Medium — knowledge accumulation
    "EMOTION":    0.35,   # Lower — context, not factual
}

DEFAULT_BASE_SCORE: float = 0.45   # fallback for unknown types

# ---------------------------------------------------------------------------
# FACT_TYPE modifier
# ---------------------------------------------------------------------------

FACT_TYPE_MODIFIER: dict[str, float] = {
    "REPLACEABLE": +0.10,   # Defines current state — more important
    "ADDITIVE":    +0.00,   # No modifier — additive facts score at base
}

# ---------------------------------------------------------------------------
# Content signal patterns — boost score when present
# ---------------------------------------------------------------------------

HIGH_SIGNAL_PATTERNS: list[tuple[str, float]] = [
    # Explicit importance markers
    (r"\b(important|critical|must|always|never|required|key)\b", +0.08),
    # Version numbers, specific tech (precise = valuable)
    (r"\b(v\d+|version\s*\d+|\d+\.\d+\.\d+)\b",                +0.05),
    # Stack / tool names (specificity signal)
    (r"\b(fastapi|sqlite|chromadb|supabase|anthropic|haiku|claude)\b", +0.05),
    # Decisions / conclusions
    (r"\b(decided|chose|going with|switched to|replaced|migrated)\b", +0.07),
    # Problems / errors
    (r"\b(error|bug|broken|failed|crash|issue|blocked)\b",       +0.06),
    # Corrections
    (r"\b(wrong|incorrect|fix|corrected|actually|no wait)\b",    +0.07),
    # Goals
    (r"\b(goal|target|aim|plan|roadmap|milestone)\b",            +0.05),
]

LOW_SIGNAL_PATTERNS: list[tuple[str, float]] = [
    # Casual / low-information
    (r"\b(maybe|probably|might|could|perhaps|possibly)\b",       -0.05),
    # Filler phrases
    (r"\b(just checking|by the way|anyway|random thought)\b",    -0.06),
    # Very short content (likely low-value)
]

# ---------------------------------------------------------------------------
# Length bonus — longer messages carry more information
# ---------------------------------------------------------------------------

def _length_bonus(text: str) -> float:
    """
    Small bonus for longer, denser messages.
    Cap at +0.08 to avoid over-weighting verbosity.
    """
    word_count = len(text.split())
    if word_count >= 80:
        return 0.08
    elif word_count >= 40:
        return 0.05
    elif word_count >= 15:
        return 0.02
    return 0.0


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_importance(
    memory_type: str,
    fact_type:   str,
    full_text:   str,
    subtype:     Optional[str] = None,
) -> float:
    """
    Calculate importance score for a memory unit at capture time.

    Args:
        memory_type: TYPE string (PROJECT, PREFERENCE, etc.)
        fact_type:   FACT_TYPE string (REPLACEABLE or ADDITIVE)
        full_text:   Raw FULL memory content.
        subtype:     Optional SUBTYPE for fine-grained scoring.

    Returns:
        Float score in range [0.0, 1.0].
    """
    type_upper = memory_type.upper()
    fact_upper = fact_type.upper()
    text_lower = full_text.lower()

    # 1. Base score from type
    score = TYPE_BASE_SCORES.get(type_upper, DEFAULT_BASE_SCORE)

    # 2. FACT_TYPE modifier
    score += FACT_TYPE_MODIFIER.get(fact_upper, 0.0)

    # 3. High-signal pattern boosts
    for pattern, delta in HIGH_SIGNAL_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            score += delta

    # 4. Low-signal pattern penalties
    for pattern, delta in LOW_SIGNAL_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            score += delta  # delta is negative

    # 5. Length bonus
    score += _length_bonus(full_text)

    # 6. Clamp to [0.0, 1.0]
    score = max(0.0, min(1.0, score))

    logger.debug(
        "importance_scorer: type=%s fact=%s score=%.3f",
        type_upper, fact_upper, score,
    )

    return round(score, 3)


# ---------------------------------------------------------------------------
# Batch scorer
# ---------------------------------------------------------------------------

def score_batch(memories: list[dict]) -> list[float]:
    """
    Score a list of memory dicts.

    Each dict must have: type, fact_type, full
    Optional: subtype

    Returns:
        List of float scores, same order as input.
    """
    return [
        score_importance(
            memory_type=m.get("type", "LEARNING"),
            fact_type=m.get("fact_type", "ADDITIVE"),
            full_text=m.get("full", ""),
            subtype=m.get("subtype"),
        )
        for m in memories
    ]


# ---------------------------------------------------------------------------
# Recalculate after retrieval (boost frequently retrieved memories)
# ---------------------------------------------------------------------------

def boost_on_retrieval(current_importance: float, retrieval_count: int) -> float:
    """
    Slightly boost importance score for memories retrieved frequently.
    Used during nightly maintenance.

    +0.02 per retrieval, capped at 1.0.
    Only applies when retrieval_count > 0.

    Args:
        current_importance: Current importance score.
        retrieval_count:    Total times this memory was retrieved.

    Returns:
        Updated importance score.
    """
    if retrieval_count <= 0:
        return current_importance

    boost = min(retrieval_count * 0.02, 0.15)  # max +0.15 from retrieval
    return round(min(1.0, current_importance + boost), 3)