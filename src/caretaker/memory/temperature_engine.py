"""
memory/temperature_engine.py
Assigns and decays temperature tiers for all memories.
Tiers: PRIORITY_HOT → HOT → WARM → COLD

Rules:
  PRIORITY_HOT  score >= 0.95  OR  type == PRIORITY_HOT
  HOT           score >= 0.65
  WARM          score >= 0.35
  COLD          score < 0.35

Decay:
  - On every retrieval fetch, memories not accessed recently decay one tier
  - Decay is time-based: HOT → WARM after 7d idle, WARM → COLD after 14d idle
  - PRIORITY_HOT never decays automatically (manual only)
  - Retrieval resets decay clock (access = warmth)
"""

import time
import logging
from typing import Literal

logger = logging.getLogger(__name__)

TemperatureTier = Literal["PRIORITY_HOT", "HOT", "WARM", "COLD"]

# Idle seconds before decay kicks in
DECAY_THRESHOLDS: dict = {
    "HOT":  7  * 24 * 3600,   # 7 days
    "WARM": 14 * 24 * 3600,   # 14 days
}


def assign_temperature(importance_score: float, memory_type: str) -> TemperatureTier:
    """
    Assign initial temperature tier based on importance score and memory type.
    Called once at memory capture time.
    """
    mem_type_upper = memory_type.upper()

    if mem_type_upper == "PRIORITY_HOT" or importance_score >= 0.95:
        return "PRIORITY_HOT"
    elif importance_score >= 0.65:
        return "HOT"
    elif importance_score >= 0.35:
        return "WARM"
    else:
        return "COLD"


def apply_decay(current_tier: TemperatureTier, last_accessed_at: float) -> TemperatureTier:
    """
    Check if a memory should decay based on time since last access.
    Returns the new (possibly decayed) tier.

    PRIORITY_HOT → never decays
    HOT          → WARM after 7d idle
    WARM         → COLD after 14d idle
    COLD         → stays COLD
    """
    if current_tier == "PRIORITY_HOT":
        return "PRIORITY_HOT"  # Never decays

    idle_seconds = time.time() - last_accessed_at

    if current_tier == "HOT":
        threshold = DECAY_THRESHOLDS["HOT"]
        if idle_seconds > threshold:
            logger.debug(f"[TempEngine] HOT → WARM (idle {idle_seconds/3600:.1f}h)")
            return "WARM"

    elif current_tier == "WARM":
        threshold = DECAY_THRESHOLDS["WARM"]
        if idle_seconds > threshold:
            logger.debug(f"[TempEngine] WARM → COLD (idle {idle_seconds/3600:.1f}h)")
            return "COLD"

    return current_tier


def reheat(current_tier: TemperatureTier) -> TemperatureTier:
    """
    Reheat a memory one tier up when it is accessed/retrieved.
    COLD → WARM, WARM → HOT (PRIORITY_HOT stays PRIORITY_HOT).
    Called after a memory is returned in a retrieval result.
    """
    tier_up = {
        "COLD":         "WARM",
        "WARM":         "HOT",
        "HOT":          "HOT",
        "PRIORITY_HOT": "PRIORITY_HOT",
    }
    new_tier = tier_up.get(current_tier, current_tier)
    if new_tier != current_tier:
        logger.debug(f"[TempEngine] Reheated: {current_tier} → {new_tier}")
    return new_tier


def batch_decay(memories: list) -> list:
    """
    Apply decay to a list of memory dicts.
    Each dict must have 'temperature' and 'last_accessed_at' keys.
    Returns list of dicts that had their temperature changed (need DB update).
    """
    changed = []
    for mem in memories:
        old_tier = mem.get("temperature", "WARM")
        last_accessed = mem.get("last_accessed_at", time.time())
        new_tier = apply_decay(old_tier, last_accessed)
        if new_tier != old_tier:
            mem["temperature"] = new_tier
            changed.append(mem)
    return changed


def get_search_tiers(include_cold: bool = False) -> list:
    """
    Return temperature tiers to include in semantic search.
    By default: skip COLD (30-50% cost saving).
    """
    tiers = ["PRIORITY_HOT", "HOT", "WARM"]
    if include_cold:
        tiers.append("COLD")
    return tiers