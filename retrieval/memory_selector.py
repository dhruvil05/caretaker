"""
retrieval/memory_selector.py
-----------------------------
Selects SHORT or FULL text for each memory based on:
  - use_full flag set by budget_engine
  - Temperature tier
  - Available token budget remaining
  - Query relevance score

Rules (P2-T11):
  - PRIORITY_HOT + high relevance score (>0.5) → prefer FULL text
  - HOT → SHORT by default, FULL if budget allows and score > 0.6
  - WARM → always SHORT
  - If total estimated tokens would exceed budget → downgrade FULL → SHORT

Also handles P2-T14: OUTDATED memories are stripped from final output.

Phase 2 — Retrieval Upgrade
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds for FULL text selection
# ---------------------------------------------------------------------------

PRIORITY_HOT_FULL_SCORE_THRESHOLD: float = 0.30  # PRIORITY_HOT: FULL if score > this
HOT_FULL_SCORE_THRESHOLD:          float = 0.60  # HOT: FULL only if score > this
SHORT_TOKEN_ESTIMATE:              int   = 60
FULL_TOKEN_ESTIMATE:               int   = 200


# ---------------------------------------------------------------------------
# MemorySelector
# ---------------------------------------------------------------------------

class MemorySelector:
    """
    Selects SHORT or FULL text representation for each memory in a
    retrieval result set, respecting token budget constraints.

    Usage:
        selector = MemorySelector(db_fetch_fn)
        selected = await selector.select(results, token_budget=600)
    """

    def __init__(self, db_fetch_fn=None):
        """
        Args:
            db_fetch_fn: Optional async fn(memory_id: str) → dict with 'full' field.
                         Used to fetch FULL text from SQLite.
                         If None, SHORT text from ChromaDB result is used for all.
        """
        self._db_fetch = db_fetch_fn

    # -----------------------------------------------------------------------
    # Main selection — async (may fetch FULL from DB)
    # -----------------------------------------------------------------------

    async def select(
        self,
        results:      list[dict],
        token_budget: int = 600,
        force_short:  bool = False,
    ) -> list[dict]:
        """
        Select SHORT or FULL text for each memory result.

        Modifies results in-place by adding 'content' and 'content_type' fields.

        Args:
            results:      List of memory dicts from BudgetEngine.retrieve().
            token_budget: Total token budget for this retrieval call.
            force_short:  If True, always use SHORT (L0 greeting path).

        Returns:
            Filtered + enriched list of memory dicts with 'content' field set.
            OUTDATED memories are removed (P2-T14).
        """
        # Strip OUTDATED (P2-T14)
        active = [r for r in results if r.get("status", "ACTIVE") != "OUTDATED"]

        if not active:
            return []

        if force_short:
            return self._apply_short_to_all(active)

        # Decide SHORT vs FULL per memory
        selections  = self._decide_selections(active)
        token_count = self._estimate_tokens(selections)

        # If over budget → downgrade FULL → SHORT until within budget
        if token_count > token_budget:
            selections = self._downgrade_to_budget(selections, token_budget)

        # Fetch FULL text from DB where needed
        enriched = await self._enrich(selections)

        logger.debug(
            "memory_selector: %d memories selected (~%d tokens, budget=%d)",
            len(enriched),
            self._estimate_tokens(enriched),
            token_budget,
        )

        return enriched

    # -----------------------------------------------------------------------
    # Sync version (for cases where DB fetch is not needed)
    # -----------------------------------------------------------------------

    def select_sync(
        self,
        results:      list[dict],
        token_budget: int = 600,
        force_short:  bool = False,
    ) -> list[dict]:
        """
        Sync version of select() — uses SHORT text only (no DB fetch).
        Use when FULL text is not required or DB is not available.
        """
        active = [r for r in results if r.get("status", "ACTIVE") != "OUTDATED"]

        if not active or force_short:
            return self._apply_short_to_all(active)

        selections  = self._decide_selections(active)
        # Force all to SHORT since no DB fetch available
        for s in selections:
            s["_use_full"] = False

        return self._apply_short_to_all(selections)

    # -----------------------------------------------------------------------
    # Internal: decide SHORT vs FULL per memory
    # -----------------------------------------------------------------------

    def _decide_selections(self, results: list[dict]) -> list[dict]:
        """
        Add _use_full flag to each result dict based on temperature + score.
        Does NOT fetch from DB yet.

        P2-T11: Deep question → FULL memory text.
        """
        out = []
        for r in results:
            r = dict(r)  # don't mutate original
            temp  = r.get("temperature", "HOT")
            score = r.get("score", 0.0)

            # Respect budget_engine's explicit use_full tag first
            if r.get("use_full"):
                r["_use_full"] = True

            elif temp == "PRIORITY_HOT" and score >= PRIORITY_HOT_FULL_SCORE_THRESHOLD:
                r["_use_full"] = True

            elif temp == "HOT" and score >= HOT_FULL_SCORE_THRESHOLD:
                r["_use_full"] = True

            else:
                r["_use_full"] = False

            out.append(r)

        return out

    # -----------------------------------------------------------------------
    # Internal: estimate token cost
    # -----------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(results: list[dict]) -> int:
        """Rough token estimate for a list of results."""
        total = 0
        for r in results:
            if r.get("_use_full", False):
                total += FULL_TOKEN_ESTIMATE
            else:
                total += SHORT_TOKEN_ESTIMATE
        return total

    # -----------------------------------------------------------------------
    # Internal: downgrade FULL → SHORT to fit budget
    # -----------------------------------------------------------------------

    @staticmethod
    def _downgrade_to_budget(results: list[dict], token_budget: int) -> list[dict]:
        """
        Downgrade FULL → SHORT starting from lowest-importance memories
        until total token estimate fits within budget.
        """
        # Sort: downgrade lowest importance first
        sortable = sorted(
            enumerate(results),
            key=lambda x: x[1].get("importance", 0.5),
        )

        total = MemorySelector._estimate_tokens(results)

        for orig_idx, r in sortable:
            if total <= token_budget:
                break
            if r.get("_use_full", False):
                results[orig_idx]["_use_full"] = False
                total -= (FULL_TOKEN_ESTIMATE - SHORT_TOKEN_ESTIMATE)

        return results

    # -----------------------------------------------------------------------
    # Internal: fetch FULL text and build content field
    # -----------------------------------------------------------------------

    async def _enrich(self, results: list[dict]) -> list[dict]:
        """
        Fetch FULL text from DB for _use_full=True memories.
        Sets 'content' and 'content_type' fields on each result.
        """
        enriched = []
        for r in results:
            r = dict(r)

            if r.get("_use_full") and self._db_fetch:
                try:
                    db_row = await self._db_fetch(r["id"])
                    if db_row and db_row.get("full"):
                        r["content"]      = db_row["full"]
                        r["content_type"] = "FULL"
                    else:
                        # DB returned nothing — fall back to SHORT
                        r["content"]      = r.get("short", "")
                        r["content_type"] = "SHORT"
                except Exception as exc:
                    logger.warning(
                        "memory_selector: DB fetch failed for %s: %s — using SHORT",
                        r["id"], exc,
                    )
                    r["content"]      = r.get("short", "")
                    r["content_type"] = "SHORT"
            else:
                r["content"]      = r.get("short", "")
                r["content_type"] = "SHORT"

            # Clean internal flag
            r.pop("_use_full", None)
            enriched.append(r)

        return enriched

    # -----------------------------------------------------------------------
    # Internal: apply SHORT to all
    # -----------------------------------------------------------------------

    @staticmethod
    def _apply_short_to_all(results: list[dict]) -> list[dict]:
        """Set content=short for all results. No DB fetch needed."""
        out = []
        for r in results:
            r = dict(r)
            r["content"]      = r.get("short", "")
            r["content_type"] = "SHORT"
            r.pop("_use_full", None)
            out.append(r)
        return out


# ---------------------------------------------------------------------------
# Convenience: format results into whisper string
# ---------------------------------------------------------------------------

def format_whisper(
    selected:    list[dict],
    max_tokens:  int = 600,
) -> str:
    """
    Format selected memories into a whisper string for the context window.

    P2-T14: OUTDATED memories are excluded.

    Args:
        selected:   List of enriched memory dicts from MemorySelector.
        max_tokens: Hard cap on whisper length.

    Returns:
        Formatted string: "[TYPE][TEMP] content\\n..."
    """
    lines  = []
    tokens = 0

    for r in selected:
        # P2-T14: skip OUTDATED
        if r.get("status") == "OUTDATED":
            continue

        line = f"[{r.get('type','?')}][{r.get('temperature','?')}] {r.get('content','')}"
        est  = len(line) // 4  # rough token estimate

        if tokens + est > max_tokens:
            break

        lines.append(line)
        tokens += est

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_selector: Optional[MemorySelector] = None


def get_selector() -> MemorySelector:
    """Return the global MemorySelector instance."""
    global _global_selector
    if _global_selector is None:
        raise RuntimeError("MemorySelector not initialised. Call init_selector() first.")
    return _global_selector


def init_selector(db_fetch_fn=None) -> MemorySelector:
    """
    Initialise the global MemorySelector singleton.

    Args:
        db_fetch_fn: Async fn(id: str) → dict with 'full' field.

    Returns:
        Initialised MemorySelector instance.
    """
    global _global_selector
    _global_selector = MemorySelector(db_fetch_fn)
    logger.info("memory_selector: global selector initialised")
    return _global_selector