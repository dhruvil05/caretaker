"""
memory/conflict_checker.py
Detects conflicts between new and existing memories.
Resolves them based on REPLACEABLE vs ADDITIVE type classification.

REPLACEABLE types (only one "truth" at a time):
  PROJECT, PREFERENCE, PERSONAL, DECISION, CORRECTION
  → Old memory marked OUTDATED. New memory becomes ACTIVE.
  → History preserved (old row stays in DB, status=OUTDATED)

ADDITIVE types (multiple truths can coexist):
  PROBLEM, LEARNING, EMOTION
  → Both old and new stay ACTIVE.
  → No conflict. Just accumulate.
"""

import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Types where only ONE active memory should exist per user
REPLACEABLE_TYPES = {"PROJECT", "PREFERENCE", "PERSONAL", "DECISION", "CORRECTION"}

# Types where multiple memories coexist (no replacement)
ADDITIVE_TYPES = {"PROBLEM", "LEARNING", "EMOTION"}


def is_replaceable(memory_type: str) -> bool:
    return memory_type.upper() in REPLACEABLE_TYPES


def check_conflict(
    new_memory: Dict,
    existing_memories: List[Dict],
    similarity_threshold: float = 0.75,
) -> Tuple[List[str], str]:
    """
    Check if new_memory conflicts with any existing active memories.

    Args:
        new_memory: dict with keys: memory_type, short, keywords, full_text
        existing_memories: list of active memory dicts from DB
        similarity_threshold: keyword overlap ratio to consider "same topic"

    Returns:
        (conflicting_ids: List[str], resolution: str)
        resolution is one of: "REPLACE", "ADDITIVE", "NONE"
    """
    mem_type = new_memory.get("memory_type", "").upper()

    # ADDITIVE types never conflict
    if mem_type in ADDITIVE_TYPES:
        return ([], "ADDITIVE")

    # REPLACEABLE: find same-type memories with keyword overlap
    same_type = [
        m for m in existing_memories
        if m.get("memory_type", "").upper() == mem_type
        and m.get("status") == "ACTIVE"
    ]

    if not same_type:
        return ([], "NONE")

    new_keywords = set(k.lower() for k in new_memory.get("keywords", []))
    conflicting_ids = []

    for existing in same_type:
        existing_keywords = set(k.lower() for k in existing.get("keywords", []))

        if not new_keywords or not existing_keywords:
            # No keywords to compare — assume conflict by type alone
            conflicting_ids.append(existing["memory_id"])
            continue

        # Jaccard similarity on keywords
        intersection = len(new_keywords & existing_keywords)
        union = len(new_keywords | existing_keywords)
        similarity = intersection / union if union > 0 else 0.0

        logger.debug(
            f"[ConflictChecker] Comparing new {mem_type} vs "
            f"existing {existing['memory_id']}: "
            f"keyword similarity={similarity:.2f}"
        )

        if similarity >= similarity_threshold:
            conflicting_ids.append(existing["memory_id"])

    if conflicting_ids:
        return (conflicting_ids, "REPLACE")

    return ([], "NONE")


def resolve_conflict(
    new_memory: Dict,
    conflicting_ids: List[str],
    resolution: str,
    local_db,
) -> Dict:
    """
    Apply conflict resolution to DB.

    REPLACE:
      - Mark all conflicting memory IDs as OUTDATED in SQLite
      - New memory will be inserted as ACTIVE by caller
    ADDITIVE / NONE:
      - Do nothing — new memory inserts normally

    Returns new_memory dict (unchanged, caller inserts it).
    """
    if resolution == "REPLACE" and conflicting_ids:
        for old_id in conflicting_ids:
            local_db.update_status(old_id, "OUTDATED")
            # 0.5: remember what replaced it, so the dashboard can show and undo it.
            if new_memory.get("memory_id") and hasattr(local_db, "update_memory_fields"):
                local_db.update_memory_fields(
                    old_id, {"superseded_by": new_memory["memory_id"]}, touch_updated_at=False
                )
            logger.info(
                f"[ConflictChecker] Marked memory_id={old_id} as OUTDATED "
                f"(replaced by new {new_memory.get('memory_type')} memory)"
            )

    elif resolution == "ADDITIVE":
        logger.debug(
            f"[ConflictChecker] ADDITIVE type {new_memory.get('memory_type')} — "
            f"no conflict resolution needed"
        )

    return new_memory


def _normalise_row(m: Dict) -> Dict:
    """Accept both database rows and the dicts used in tests."""
    out = dict(m)
    out.setdefault("memory_type", m.get("type", ""))
    out.setdefault("memory_id", m.get("id"))
    kw = m.get("keywords")
    if isinstance(kw, str):
        try:
            import json
            parsed = json.loads(kw)
            kw = parsed if isinstance(parsed, list) else [str(parsed)]
        except ValueError:
            kw = [k.strip() for k in kw.split(",") if k.strip()]
    out["keywords"] = [str(k) for k in (kw or [])]
    return out


def full_conflict_pipeline(
    new_memory: Dict,
    local_db,
    similarity_threshold: float = 0.75,
) -> Dict:
    """
    Convenience: fetch existing memories, check conflict, resolve, return new_memory.
    Caller still inserts new_memory after this returns.
    """
    mem_type = new_memory.get("memory_type", "")
    project = new_memory.get("project") or ""
    if hasattr(local_db, "search_fulltext"):
        # 0.5: only the best keyword matches, not every memory of this type.
        # Similarity is keyword overlap, so a memory sharing no words can't conflict.
        keywords = list(new_memory.get("keywords") or [])
        if keywords:
            existing = local_db.search_fulltext(
                keywords, limit=50, memory_type=mem_type,
                keywords_only=True, min_shared=2 if len(keywords) >= 2 else 1,
            )
        else:
            text = new_memory.get("full_text", "")
            existing = local_db.search_fulltext(text, limit=50, memory_type=mem_type) if text else []
    else:
        existing = local_db.get_active_by_type(mem_type)
    # Database rows use "type" and store keywords as JSON; check_conflict expects
    # "memory_type" and a list. Replacement never crosses projects.
    existing = [
        _normalise_row(m) for m in existing
        if (m.get("project") or "") == project
    ]

    conflicting_ids, resolution = check_conflict(
        new_memory=new_memory,
        existing_memories=existing,
        similarity_threshold=similarity_threshold,
    )

    resolve_conflict(
        new_memory=new_memory,
        conflicting_ids=conflicting_ids,
        resolution=resolution,
        local_db=local_db,
    )

    return new_memory