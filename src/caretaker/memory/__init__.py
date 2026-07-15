"""
caretaker.memory
Public API for conflict resolution, importance scoring, and temperature.
"""

from .conflict_checker import (
    is_replaceable,
    check_conflict,
    resolve_conflict,
    full_conflict_pipeline,
)
from .importance_scorer import score_importance, score_batch
from .temperature_engine import (
    assign_temperature,
    apply_decay,
    reheat,
    batch_decay,
    get_search_tiers,
)

__all__ = [
    "is_replaceable",
    "check_conflict",
    "resolve_conflict",
    "full_conflict_pipeline",
    "score_importance",
    "score_batch",
    "assign_temperature",
    "apply_decay",
    "reheat",
    "batch_decay",
    "get_search_tiers",
]
