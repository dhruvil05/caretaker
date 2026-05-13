"""
retrieval/semantic_searcher.py
Semantic search with temperature pre-filtering and multi-factor ranking.
Ranking formula: final_score = semantic_score × temp_weight × importance × recency_factor
"""

import time
import math
import logging
from typing import List, Dict, Optional

from memory.temperature_engine import get_search_tiers

logger = logging.getLogger(__name__)

# Temperature tier → weight multiplier in ranking
TEMPERATURE_WEIGHTS: Dict[str, float] = {
    "PRIORITY_HOT": 2.0,
    "HOT":          1.5,
    "WARM":         1.0,
    "COLD":         0.3,
}

# Cosine distance → similarity conversion
# ChromaDB returns distance (0=identical, 2=opposite for cosine)
# We convert: similarity = 1 - (distance / 2)
def _distance_to_similarity(distance: float) -> float:
    return max(0.0, 1.0 - (distance / 2.0))


def _recency_factor(last_accessed_at: float, decay_days: float = 30.0) -> float:
    """
    Exponential recency decay.
    Memory accessed today → 1.0
    Memory accessed 30d ago → ~0.37
    Memory accessed 90d ago → ~0.05
    """
    age_days = (time.time() - last_accessed_at) / 86400.0
    return math.exp(-age_days / decay_days)


class SemanticSearcher:
    """
    Phase 2 semantic search engine.
    Replaces Phase 1 keyword lookup.

    Flow:
      1. Pre-filter by temperature (skip COLD by default → 30-50% cheaper)
      2. Vector search in ChromaDB
      3. Fetch full records from SQLite for top hits
      4. Rank by: semantic_score × temp_weight × importance × recency
      5. Return top-N ranked memories
    """

    def __init__(self, vector_db, local_db, include_cold: bool = False):
        self.vector_db = vector_db
        self.local_db = local_db
        self.include_cold = include_cold

    def search(
        self,
        query: str,
        n_results: int = 10,
        force_include_cold: bool = False,
    ) -> List[Dict]:
        """
        Main search entry point.
        Returns ranked list of memory dicts enriched with 'relevance_score'.
        """
        if not query or not query.strip():
            return []

        # Step 1: Temperature pre-filter
        include_cold = force_include_cold or self.include_cold
        tiers = get_search_tiers(include_cold=include_cold)
        logger.debug(f"[SemanticSearcher] Searching tiers: {tiers}")

        # Step 2: Vector search
        vector_hits = self.vector_db.search(
            query=query,
            n_results=n_results * 2,  # Over-fetch for re-ranking
            temperature_filter=tiers,
        )

        if not vector_hits:
            logger.debug("[SemanticSearcher] No vector hits found.")
            return []

        # Step 3: Fetch full records from SQLite
        memory_ids = [h["memory_id"] for h in vector_hits]
        full_records = self.local_db.get_by_ids(memory_ids)

        # Build lookup: memory_id → full record
        record_map = {r["memory_id"]: r for r in full_records}

        # Step 4: Rank
        ranked = []
        for hit in vector_hits:
            mid = hit["memory_id"]
            record = record_map.get(mid)
            if not record:
                continue

            semantic_sim = _distance_to_similarity(hit.get("distance", 1.0))
            temp_weight = TEMPERATURE_WEIGHTS.get(hit.get("temperature", "WARM"), 1.0)
            importance = float(record.get("importance_score", 0.5))
            last_accessed = record.get("last_accessed_at", time.time())
            recency = _recency_factor(last_accessed)

            relevance_score = semantic_sim * temp_weight * importance * recency

            ranked.append({
                **record,
                "short": hit.get("short", ""),
                "relevance_score": round(relevance_score, 6),
                "semantic_similarity": round(semantic_sim, 4),
            })

        # Sort by relevance descending
        ranked.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Return top-N
        results = ranked[:n_results]

        logger.info(
            f"[SemanticSearcher] Query: {query[:50]!r} → "
            f"{len(results)} results (top score: "
            f"{(results[0]['relevance_score'] if results else 0):.4f})"
        )

        return results

    def search_by_type(
        self,
        query: str,
        memory_type: str,
        n_results: int = 5,
    ) -> List[Dict]:
        """Search within a specific memory type only."""
        all_results = self.search(query=query, n_results=n_results * 3)
        filtered = [r for r in all_results if r.get("memory_type", "").upper() == memory_type.upper()]
        return filtered[:n_results]