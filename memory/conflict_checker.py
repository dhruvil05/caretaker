"""
memory/conflict_checker.py
---------------------------
Detects and resolves conflicting memory units after every new capture.

Conflict rules (from architecture doc):
  REPLACEABLE fact_type:
    → New memory replaces old.
    → Old marked OUTDATED. old.superseded_by = new.id.
    → Both kept in DB (full audit trail).

  ADDITIVE fact_type:
    → No conflict. Both stay ACTIVE.
    → New memory added alongside old.

Edge case — Two memories, same TYPE, different SUBTYPE:
    → Treat as ADDITIVE (different subtypes = different things).

Detection method:
    1. Query SQLite for ACTIVE memories with same TYPE.
    2. Check keyword overlap > 60% between new and existing.
    3. Apply FACT_TYPE rule.

Phase 2 — Memory Brain
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum keyword overlap ratio to consider a conflict
CONFLICT_OVERLAP_THRESHOLD: float = 0.40


# ---------------------------------------------------------------------------
# Keyword overlap calculator
# ---------------------------------------------------------------------------

def _keyword_overlap(keywords_a: list[str], keywords_b: list[str]) -> float:
    """
    Calculate overlap ratio between two keyword lists.

    Overlap = |intersection| / |union|  (Jaccard similarity)

    Args:
        keywords_a: Keywords of memory A.
        keywords_b: Keywords of memory B.

    Returns:
        Float 0.0–1.0. Higher = more overlap.
    """
    if not keywords_a or not keywords_b:
        return 0.0

    set_a = set(k.lower() for k in keywords_a)
    set_b = set(k.lower() for k in keywords_b)

    intersection = set_a & set_b
    union        = set_a | set_b

    if not union:
        return 0.0

    return len(intersection) / len(union)


def _text_overlap(text_a: str, text_b: str) -> float:
    """
    Fallback overlap check using word tokens when keywords unavailable.
    Same Jaccard logic on word tokens.
    """
    if not text_a or not text_b:
        return 0.0

    stop = {"a","an","the","and","or","in","on","at","to","for","of","with",
            "is","are","was","were","i","my","me","we","you","it","this","that"}

    words_a = set(w.lower() for w in text_a.split() if w.lower() not in stop and len(w) > 2)
    words_b = set(w.lower() for w in text_b.split() if w.lower() not in stop and len(w) > 2)

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union        = words_a | words_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Conflict resolution result
# ---------------------------------------------------------------------------

class ConflictResult:
    """Result of a single conflict check."""

    def __init__(
        self,
        conflict_found:   bool,
        fact_type:        str,
        existing_id:      Optional[str] = None,
        action:           str = "none",
        overlap_score:    float = 0.0,
    ):
        """
        Args:
            conflict_found: True if a conflicting memory was found.
            fact_type:      REPLACEABLE or ADDITIVE.
            existing_id:    ID of the conflicting existing memory (if found).
            action:         What was done: 'replaced', 'kept_both', 'none'.
            overlap_score:  Overlap ratio that triggered the conflict.
        """
        self.conflict_found = conflict_found
        self.fact_type      = fact_type
        self.existing_id    = existing_id
        self.action         = action
        self.overlap_score  = overlap_score

    def __repr__(self):
        return (
            f"ConflictResult(found={self.conflict_found}, "
            f"fact_type={self.fact_type}, action={self.action}, "
            f"overlap={self.overlap_score:.2f})"
        )


# ---------------------------------------------------------------------------
# Core conflict checker — works with plain dicts (DB-agnostic)
# ---------------------------------------------------------------------------

def check_conflict(
    new_memory:        dict,
    existing_memories: list[dict],
    overlap_threshold: float = CONFLICT_OVERLAP_THRESHOLD,
) -> ConflictResult:
    """
    Check if new_memory conflicts with any existing ACTIVE memories.

    Does NOT write to DB — returns result for caller to act on.

    Args:
        new_memory:        Dict with keys: id, type, subtype, fact_type, keywords, full
        existing_memories: List of ACTIVE memory dicts with same TYPE from DB.
        overlap_threshold: Minimum keyword overlap to flag conflict.

    Returns:
        ConflictResult describing what should happen.
    """
    new_type     = new_memory.get("type", "").upper()
    new_subtype  = (new_memory.get("subtype") or "").lower()
    new_fact     = new_memory.get("fact_type", "ADDITIVE").upper()
    new_keywords = new_memory.get("keywords") or []
    new_full     = new_memory.get("full", "")

    best_overlap  = 0.0
    best_existing = None

    for existing in existing_memories:
        ex_id      = existing.get("id")
        ex_subtype = (existing.get("subtype") or "").lower()
        ex_keywords = existing.get("keywords") or []
        ex_full     = existing.get("full", "")

        # Edge case: same TYPE but different non-empty subtype → treat as ADDITIVE
        if new_subtype and ex_subtype and new_subtype != ex_subtype:
            logger.debug(
                "conflict_checker: skipping %s — different subtype (%s vs %s)",
                ex_id, new_subtype, ex_subtype,
            )
            continue

        # Calculate overlap
        if new_keywords and ex_keywords:
            overlap = _keyword_overlap(new_keywords, ex_keywords)
        else:
            # Fallback to text overlap when keywords not yet generated
            overlap = _text_overlap(new_full, ex_full)

        logger.debug(
            "conflict_checker: overlap %.2f with %s (threshold=%.2f)",
            overlap, ex_id, overlap_threshold,
        )

        if overlap > best_overlap:
            best_overlap  = overlap
            best_existing = existing

    # No conflict found
    if best_existing is None or best_overlap < overlap_threshold:
        return ConflictResult(
            conflict_found=False,
            fact_type=new_fact,
            action="none",
            overlap_score=best_overlap,
        )

    # Conflict found — apply FACT_TYPE rule
    existing_id = best_existing.get("id")

    if new_fact == "REPLACEABLE":
        logger.info(
            "conflict_checker: REPLACEABLE conflict — %s supersedes %s (overlap=%.2f)",
            new_memory.get("id"), existing_id, best_overlap,
        )
        return ConflictResult(
            conflict_found=True,
            fact_type="REPLACEABLE",
            existing_id=existing_id,
            action="replaced",
            overlap_score=best_overlap,
        )

    else:  # ADDITIVE
        logger.info(
            "conflict_checker: ADDITIVE overlap — keeping both %s and %s (overlap=%.2f)",
            new_memory.get("id"), existing_id, best_overlap,
        )
        return ConflictResult(
            conflict_found=True,
            fact_type="ADDITIVE",
            existing_id=existing_id,
            action="kept_both",
            overlap_score=best_overlap,
        )


# ---------------------------------------------------------------------------
# DB update helpers — called by capture engine after conflict check
# ---------------------------------------------------------------------------

def build_outdated_update(
    existing_id:   str,
    superseded_by: str,
) -> dict:
    """
    Build the SQLite update dict to mark an existing memory as OUTDATED.

    Args:
        existing_id:   ID of the memory to mark OUTDATED.
        superseded_by: ID of the new memory that replaces it.

    Returns:
        Dict of fields to update in SQLite.
    """
    return {
        "id":            existing_id,
        "status":        "OUTDATED",
        "superseded_by": superseded_by,
        "updated_at":    datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Convenience: run full conflict pipeline given a DB query function
# ---------------------------------------------------------------------------

async def run_conflict_check(
    new_memory:      dict,
    db_query_fn,
    db_update_fn,
    overlap_threshold: float = CONFLICT_OVERLAP_THRESHOLD,
) -> ConflictResult:
    """
    Run full conflict check + DB update in one call.

    Args:
        new_memory:       New memory dict (must include id, type, fact_type).
        db_query_fn:      Async fn(type: str) → list[dict] — returns ACTIVE memories of that type.
        db_update_fn:     Async fn(update: dict) → None — applies update to DB.
        overlap_threshold: Overlap ratio threshold.

    Returns:
        ConflictResult.
    """
    memory_type = new_memory.get("type", "")

    # Fetch ACTIVE memories with same type from DB
    existing = await db_query_fn(memory_type)

    # Remove the new memory itself (shouldn't conflict with itself)
    new_id   = new_memory.get("id")
    existing = [m for m in existing if m.get("id") != new_id]

    if not existing:
        return ConflictResult(conflict_found=False, fact_type=new_memory.get("fact_type","ADDITIVE"), action="none")

    # Check conflict
    result = check_conflict(new_memory, existing, overlap_threshold)

    # Apply DB update if REPLACEABLE conflict found
    if result.conflict_found and result.action == "replaced" and result.existing_id:
        update = build_outdated_update(
            existing_id=result.existing_id,
            superseded_by=new_id,
        )
        await db_update_fn(update)
        logger.info(
            "conflict_checker: DB updated — %s marked OUTDATED, superseded by %s",
            result.existing_id, new_id,
        )

    return result