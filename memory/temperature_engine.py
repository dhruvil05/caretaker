"""
memory/temperature_engine.py
-----------------------------
Assigns and updates temperature tier for memory units.

Temperature is derived from TWO scores combined:
  - importance  : content richness (set at capture, boosted by retrieval)
  - decay_score : time-based fading (reduced nightly by decay engine)

Combined score = (importance * 0.6) + (decay_score * 0.4)

Temperature tiers (from config defaults):
  PRIORITY_HOT  → combined > 0.7   (always fetched first)
  HOT           → combined > 0.5   (fetched in all standard retrievals)
  WARM          → 0.2 ≤ combined ≤ 0.5 (fetched only if semantically relevant)
  COLD          → combined < 0.2   (never fetched)

Phase 2 — Memory Brain
"""

import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds (mirror config.json defaults)
# ---------------------------------------------------------------------------

DEFAULT_PRIORITY_HOT_THRESHOLD: float = 0.70
DEFAULT_HOT_THRESHOLD:          float = 0.50
DEFAULT_COLD_THRESHOLD:         float = 0.20

# Weight split between importance and decay
IMPORTANCE_WEIGHT: float = 0.60
DECAY_WEIGHT:      float = 0.40

# Valid temperature values
TEMPERATURE_VALUES = ("PRIORITY_HOT", "HOT", "WARM", "COLD", "ARCHIVED")


# ---------------------------------------------------------------------------
# Temperature result
# ---------------------------------------------------------------------------

@dataclass
class TemperatureResult:
    temperature:    str    # PRIORITY_HOT | HOT | WARM | COLD
    combined_score: float  # weighted score used to decide tier
    importance:     float
    decay_score:    float


# ---------------------------------------------------------------------------
# Core temperature calculator
# ---------------------------------------------------------------------------

def calculate_temperature(
    importance:             float,
    decay_score:            float,
    priority_hot_threshold: float = DEFAULT_PRIORITY_HOT_THRESHOLD,
    hot_threshold:          float = DEFAULT_HOT_THRESHOLD,
    cold_threshold:         float = DEFAULT_COLD_THRESHOLD,
) -> TemperatureResult:
    """
    Calculate temperature tier from importance and decay scores.

    Args:
        importance:             Memory importance score (0.0–1.0).
        decay_score:            Memory decay score (0.0–1.0, starts at 1.0).
        priority_hot_threshold: Combined score above which = PRIORITY_HOT.
        hot_threshold:          Combined score above which = HOT.
        cold_threshold:         Combined score below which = COLD.

    Returns:
        TemperatureResult with tier and combined score.
    """
    # Clamp inputs
    importance  = max(0.0, min(1.0, importance))
    decay_score = max(0.0, min(1.0, decay_score))

    # Weighted combination
    combined = round(
        (importance * IMPORTANCE_WEIGHT) + (decay_score * DECAY_WEIGHT),
        3,
    )

    # Assign tier
    if combined > priority_hot_threshold:
        tier = "PRIORITY_HOT"
    elif combined > hot_threshold:
        tier = "HOT"
    elif combined >= cold_threshold:
        tier = "WARM"
    else:
        tier = "COLD"

    logger.debug(
        "temperature_engine: importance=%.3f decay=%.3f combined=%.3f → %s",
        importance, decay_score, combined, tier,
    )

    return TemperatureResult(
        temperature=tier,
        combined_score=combined,
        importance=importance,
        decay_score=decay_score,
    )


# ---------------------------------------------------------------------------
# Assign temperature on first capture
# ---------------------------------------------------------------------------

def assign_initial_temperature(
    importance: float,
    config:     Optional[dict] = None,
) -> TemperatureResult:
    """
    Assign temperature at capture time.

    For NEW memories, temperature is based on importance score alone
    (not weighted with decay) because decay hasn't started yet.
    This ensures P2-T06 (importance=0.6 → HOT) and
    P2-T07 (importance=0.15 → COLD) pass correctly.

    Args:
        importance: Score from importance_scorer.
        config:     Optional config dict with threshold overrides.

    Returns:
        TemperatureResult.
    """
    thresholds = _extract_thresholds(config)
    p_hot = thresholds.get("priority_hot_threshold", DEFAULT_PRIORITY_HOT_THRESHOLD)
    hot   = thresholds.get("hot_threshold",          DEFAULT_HOT_THRESHOLD)
    cold  = thresholds.get("cold_threshold",          DEFAULT_COLD_THRESHOLD)

    importance = max(0.0, min(1.0, importance))

    if importance > p_hot:
        tier = "PRIORITY_HOT"
    elif importance > hot:
        tier = "HOT"
    elif importance >= cold:
        tier = "WARM"
    else:
        tier = "COLD"

    logger.debug(
        "temperature_engine: initial importance=%.3f → %s", importance, tier
    )

    return TemperatureResult(
        temperature=tier,
        combined_score=round(importance, 3),
        importance=importance,
        decay_score=1.0,
    )


# ---------------------------------------------------------------------------
# Recalculate temperature after decay or retrieval
# ---------------------------------------------------------------------------

def recalculate_temperature(
    importance:  float,
    decay_score: float,
    config:      Optional[dict] = None,
) -> TemperatureResult:
    """
    Recalculate temperature after nightly decay or retrieval boost.
    Used by decay engine and nightly maintenance.

    Args:
        importance:  Current importance score.
        decay_score: Updated decay score.
        config:      Optional config dict with threshold overrides.

    Returns:
        TemperatureResult with updated tier.
    """
    thresholds = _extract_thresholds(config)
    return calculate_temperature(
        importance=importance,
        decay_score=decay_score,
        **thresholds,
    )


# ---------------------------------------------------------------------------
# Batch recalculation — used by nightly maintenance
# ---------------------------------------------------------------------------

def recalculate_batch(
    memories: list[dict],
    config:   Optional[dict] = None,
) -> list[dict]:
    """
    Recalculate temperature for a batch of memory dicts.

    Each dict must have: id, importance, decay_score
    Returns list of dicts with: id, old_temperature, new_temperature, combined_score

    Args:
        memories: List of memory dicts from SQLite.
        config:   Optional config with threshold overrides.

    Returns:
        List of update dicts showing what changed.
    """
    updates = []
    thresholds = _extract_thresholds(config)

    for mem in memories:
        result = calculate_temperature(
            importance=mem.get("importance", 0.5),
            decay_score=mem.get("decay_score", 1.0),
            **thresholds,
        )

        old_temp = mem.get("temperature", "HOT")
        new_temp = result.temperature

        if old_temp != new_temp:
            logger.info(
                "temperature_engine: %s changed %s → %s (score=%.3f)",
                mem.get("id", "?"), old_temp, new_temp, result.combined_score,
            )

        updates.append({
            "id":              mem.get("id"),
            "old_temperature": old_temp,
            "new_temperature": new_temp,
            "combined_score":  result.combined_score,
            "changed":         old_temp != new_temp,
        })

    return updates


# ---------------------------------------------------------------------------
# Utility: check if memory should be fetched
# ---------------------------------------------------------------------------

def is_fetchable(temperature: str) -> bool:
    """
    Return True if a memory with this temperature should be included
    in retrieval results.

    COLD and ARCHIVED are never fetched.

    Args:
        temperature: Temperature string.

    Returns:
        True if fetchable, False if should be skipped.
    """
    return temperature in ("PRIORITY_HOT", "HOT", "WARM")


def is_priority(temperature: str) -> bool:
    """Return True if temperature is PRIORITY_HOT."""
    return temperature == "PRIORITY_HOT"


def sort_key(temperature: str) -> int:
    """
    Return sort key for temperature-ordered retrieval.
    Lower number = higher priority.

    PRIORITY_HOT=0, HOT=1, WARM=2, COLD=3, ARCHIVED=4
    """
    order = {"PRIORITY_HOT": 0, "HOT": 1, "WARM": 2, "COLD": 3, "ARCHIVED": 4}
    return order.get(temperature, 99)


# ---------------------------------------------------------------------------
# Internal: extract thresholds from config dict
# ---------------------------------------------------------------------------

def _extract_thresholds(config: Optional[dict]) -> dict:
    """Extract threshold values from config, falling back to defaults."""
    if not config:
        return {}
    return {
        k: v for k, v in {
            "priority_hot_threshold": config.get("priority_hot_score"),
            "hot_threshold":          config.get("hot_score_threshold"),
            "cold_threshold":         config.get("archive_score"),
        }.items() if v is not None
    }