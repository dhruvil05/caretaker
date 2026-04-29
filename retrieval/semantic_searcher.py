"""
retrieval/semantic_searcher.py
-------------------------------
Temperature-filtered semantic search layer over ChromaDB.

Wraps VectorDB.search() with Caretaker-specific logic:
  - Temperature tier ordering  (PRIORITY_HOT → HOT → WARM)
  - COLD always excluded       (P2-T09)
  - OUTDATED always excluded   (P2-T14)
  - Optional TYPE pre-filter
  - Deduplication by memory ID
  - Relevance score threshold

This is the ONLY entry point for semantic search in Phase 2.
budget_engine.py calls this to populate retrieval slots.

Phase 2 — Retrieval Upgrade
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier search order — PRIORITY_HOT first, then broaden
# ---------------------------------------------------------------------------

TIER_ORDER = [
    ["PRIORITY_HOT"],
    ["PRIORITY_HOT", "HOT"],
    ["PRIORITY_HOT", "HOT", "WARM"],
]

# Default minimum relevance score to include a result
DEFAULT_MIN_RELEVANCE: float = 0.10


# ---------------------------------------------------------------------------
# SemanticSearcher
# ---------------------------------------------------------------------------

class SemanticSearcher:
    """
    Temperature-aware semantic search over ChromaDB memory store.

    Usage:
        searcher = SemanticSearcher(vector_db)
        results  = searcher.search("what editor does the user prefer?", n=5)
    """

    def __init__(
        self,
        vector_db,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
    ):
        """
        Args:
            vector_db:     Initialised VectorDB instance.
            min_relevance: Minimum cosine similarity score to include result.
        """
        self._vdb          = vector_db
        self._min_relevance = min_relevance

    # -----------------------------------------------------------------------
    # Main search — temperature-tiered
    # -----------------------------------------------------------------------

    def search(
        self,
        query:       str,
        n:           int = 10,
        memory_type: Optional[str] = None,
        include_warm: bool = True,
    ) -> list[dict]:
        """
        Semantic search with temperature-tier ordering.

        Searches PRIORITY_HOT first. If fewer than n results found,
        broadens to HOT, then WARM (if include_warm=True).

        COLD and OUTDATED are NEVER returned (P2-T09, P2-T14).

        Args:
            query:        Natural language query string.
            n:            Max results to return.
            memory_type:  Optional TYPE filter.
            include_warm: Whether to include WARM tier results.

        Returns:
            List of result dicts sorted by: temperature tier, then relevance score.
            Each dict: {id, short, score, temperature, importance, type, keywords, status}
        """
        if not query or not query.strip():
            return []

        tiers = TIER_ORDER[2] if include_warm else TIER_ORDER[1]

        raw = self._vdb.search(
            query_text=query,
            n_results=n * 2,        # fetch extra — filtering may reduce count
            temperatures=tiers,
            memory_type=memory_type,
            min_relevance=self._min_relevance,
        )

        # Deduplicate by ID (ChromaDB can return duplicates in rare cases)
        seen = set()
        deduped = []
        for r in raw:
            if r["id"] not in seen:
                seen.add(r["id"])
                deduped.append(r)

        # Sort: temperature tier first, then score descending
        deduped.sort(key=lambda x: (_temp_sort_key(x["temperature"]), -x["score"]))

        result = deduped[:n]

        logger.debug(
            "semantic_searcher: query='%s' → %d results (type=%s warm=%s)",
            query[:50], len(result), memory_type, include_warm,
        )

        return result

    # -----------------------------------------------------------------------
    # Targeted search — specific TYPE only
    # -----------------------------------------------------------------------

    def search_by_type(
        self,
        query:       str,
        memory_type: str,
        n:           int = 5,
    ) -> list[dict]:
        """
        Search within a specific memory TYPE.

        Args:
            query:       Natural language query.
            memory_type: TYPE to filter by (PROJECT, PREFERENCE, etc.)
            n:           Max results.

        Returns:
            List of result dicts.
        """
        return self.search(query=query, n=n, memory_type=memory_type)

    # -----------------------------------------------------------------------
    # Priority-only search — PRIORITY_HOT tier only (fast path)
    # -----------------------------------------------------------------------

    def search_priority_only(
        self,
        query: str,
        n:     int = 5,
    ) -> list[dict]:
        """
        Fast search restricted to PRIORITY_HOT memories only.
        Used when budget is very tight (whisper mode, greeting).

        Args:
            query: Natural language query.
            n:     Max results.

        Returns:
            List of result dicts from PRIORITY_HOT tier only.
        """
        raw = self._vdb.search(
            query_text=query,
            n_results=n,
            temperatures=["PRIORITY_HOT"],
            min_relevance=self._min_relevance,
        )
        return raw[:n]

    # -----------------------------------------------------------------------
    # Multi-query search — merge results from multiple queries
    # -----------------------------------------------------------------------

    def search_multi(
        self,
        queries:      list[str],
        n_per_query:  int = 5,
        total_limit:  int = 10,
        include_warm: bool = True,
    ) -> list[dict]:
        """
        Run multiple queries and merge results by score.
        Useful when a single query may miss relevant memories.

        Args:
            queries:     List of query strings.
            n_per_query: Max results per query.
            total_limit: Max total results after merge.
            include_warm: Whether to include WARM tier.

        Returns:
            Deduplicated, merged list sorted by temperature then score.
        """
        seen    = {}   # id → best result dict

        for q in queries:
            results = self.search(q, n=n_per_query, include_warm=include_warm)
            for r in results:
                mid = r["id"]
                # Keep best score for each memory
                if mid not in seen or r["score"] > seen[mid]["score"]:
                    seen[mid] = r

        merged = list(seen.values())
        merged.sort(key=lambda x: (_temp_sort_key(x["temperature"]), -x["score"]))
        return merged[:total_limit]


# ---------------------------------------------------------------------------
# Module-level singleton helper
# ---------------------------------------------------------------------------

_global_searcher: Optional["SemanticSearcher"] = None


def get_searcher() -> "SemanticSearcher":
    """Return the global SemanticSearcher instance."""
    global _global_searcher
    if _global_searcher is None:
        raise RuntimeError(
            "SemanticSearcher not initialised. Call init_searcher() first."
        )
    return _global_searcher


def init_searcher(
    vector_db,
    min_relevance: float = DEFAULT_MIN_RELEVANCE,
) -> "SemanticSearcher":
    """
    Initialise the global SemanticSearcher singleton.
    Call once at server startup after init_vector_db().

    Args:
        vector_db:     Initialised VectorDB instance.
        min_relevance: Min cosine similarity to include results.

    Returns:
        Initialised SemanticSearcher instance.
    """
    global _global_searcher
    _global_searcher = SemanticSearcher(vector_db, min_relevance)
    logger.info("semantic_searcher: global searcher initialised")
    return _global_searcher


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _temp_sort_key(temperature: str) -> int:
    """Lower number = fetched first."""
    order = {"PRIORITY_HOT": 0, "HOT": 1, "WARM": 2, "COLD": 3}
    return order.get(temperature, 99)