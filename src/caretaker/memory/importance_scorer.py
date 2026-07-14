"""
memory/importance_scorer.py
Scores a memory's importance from 0.0 to 1.0 at capture time.
Score is used by temperature_engine to assign initial temperature tier.
No API call — pure rule-based scoring. Fast.
"""

import re
from typing import Dict

# Base scores per memory type
TYPE_BASE_SCORES: Dict[str, float] = {
    "PRIORITY_HOT": 1.0,
    "PROJECT":       0.85,
    "DECISION":      0.80,
    "PROBLEM":       0.75,
    "CORRECTION":    0.70,
    "PREFERENCE":    0.65,
    "PERSONAL":      0.60,
    "LEARNING":      0.55,
    "EMOTION":       0.40,
}

DEFAULT_BASE_SCORE = 0.50

# Urgency signal words → boost score
URGENCY_SIGNALS = [
    r"\bcritical\b", r"\bblocking\b", r"\burgent\b", r"\bcrash\b",
    r"\bsegfault\b", r"\bfailed\b", r"\bbroken\b", r"\bpriority\b",
    r"\bimportant\b", r"\bmust\b", r"\bnever\b", r"\balways\b",
    r"\bkey\b", r"\bcore\b", r"\bfundamental\b", r"\barchitecture\b",
]

# Decay signals → reduce score
WEAK_SIGNALS = [
    r"\bthanks\b", r"\bok\b", r"\bokay\b", r"\bsure\b", r"\bgot it\b",
    r"\bhi\b", r"\bhello\b", r"\bbye\b", r"\bsee you\b", r"\btest\b",
]


def score_importance(full_text: str, memory_type: str) -> float:
    """
    Score importance of a memory from 0.0 (trivial) to 1.0 (critical).

    Rules:
      1. Start with type base score
      2. +0.05 per urgency signal found (max +0.20)
      3. -0.05 per weak signal found (max -0.15)
      4. +0.05 if text is long (>200 chars) — more content = more likely important
      5. Clamp to [0.05, 1.0]
    """
    base = TYPE_BASE_SCORES.get(memory_type.upper(), DEFAULT_BASE_SCORE)
    text_lower = full_text.lower()

    # Urgency boost
    urgency_count = sum(
        1 for pattern in URGENCY_SIGNALS
        if re.search(pattern, text_lower)
    )
    urgency_boost = min(urgency_count * 0.05, 0.20)

    # Weak signal penalty
    weak_count = sum(
        1 for pattern in WEAK_SIGNALS
        if re.search(pattern, text_lower)
    )
    weak_penalty = min(weak_count * 0.05, 0.15)

    # Length bonus
    length_bonus = 0.05 if len(full_text.strip()) > 200 else 0.0

    score = base + urgency_boost - weak_penalty + length_bonus

    # Clamp
    return round(max(0.05, min(1.0, score)), 4)


def score_batch(memories: list) -> list:
    """
    Score a list of memory dicts in place.
    Each dict must have 'full_text' and 'memory_type' keys.
    Adds 'importance_score' key to each dict.
    Returns the updated list.
    """
    for mem in memories:
        mem["importance_score"] = score_importance(
            full_text=mem.get("full_text", ""),
            memory_type=mem.get("memory_type", "LEARNING"),
        )
    return memories