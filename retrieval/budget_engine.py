"""
retrieval/budget_engine.py
---------------------------
Smart token budget engine for context assembly — L0 through L5.

Each retrieval call gets a token budget. The engine allocates that budget
across memory tiers and query types to maximise relevance per token.

Budget Levels (from architecture doc):
  L0 — Greeting / ping          : ≤ 120 tokens  (PRIORITY_HOT SHORT only)
  L1 — Simple factual query     : ≤ 300 tokens  (HOT SHORT + top PRIORITY_HOT FULL)
  L2 — Standard query           : ≤ 600 tokens  (HOT + WARM SHORT, key FULL)
  L3 — Complex / multi-topic    : ≤ 1200 tokens (all tiers, more FULL)
  L4 — Deep research / project  : ≤ 2400 tokens (full context, all tiers)
  L5 — Unrestricted (explicit)  : ≤ 4000 tokens (everything relevant)

Budget allocation per level:
  - PRIORITY_HOT slots : always filled first
  - HOT slots          : filled next
  - WARM slots         : filled last if budget remains
  - SHORT vs FULL      : determined by memory_selector.py (called after budget)

Phase 2 — Retrieval Upgrade
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Budget level definitions
# ---------------------------------------------------------------------------

@dataclass
class BudgetLevel:
    level:              int
    name:               str
    token_limit:        int
    priority_hot_slots: int    # How many PRIORITY_HOT memories to fetch
    hot_slots:          int    # How many HOT memories to fetch
    warm_slots:         int    # How many WARM memories to fetch
    full_slots:         int    # How many memories get FULL text (rest get SHORT)


BUDGET_LEVELS: dict[int, BudgetLevel] = {
    0: BudgetLevel(0, "greeting",      120,   3,  0,  0,  0),
    1: BudgetLevel(1, "simple",        300,   3,  2,  0,  1),
    2: BudgetLevel(2, "standard",      600,   4,  3,  2,  2),
    3: BudgetLevel(3, "complex",      1200,   5,  5,  3,  4),
    4: BudgetLevel(4, "deep",         2400,   6,  6,  5,  8),
    5: BudgetLevel(5, "unrestricted", 4000,  10, 10,  8, 15),
}

# Rough token estimates
SHORT_TOKEN_ESTIMATE: int = 60    # max SHORT summary length
FULL_TOKEN_ESTIMATE:  int = 200   # average FULL memory length


# ---------------------------------------------------------------------------
# Query classifier — determines budget level from query text
# ---------------------------------------------------------------------------

# Patterns → suggested budget level
_QUERY_PATTERNS: list[tuple[re.Pattern, int]] = [
    # L0: greetings / pings
    (re.compile(
        r"^(hi|hello|hey|what'?s up|good morning|good evening|ping|sup|yo|greet)\b",
        re.IGNORECASE,
    ), 0),

    # L4/L5: explicit deep research requests
    (re.compile(
        r"\b(everything|full context|all memories|complete history|deep dive|research)\b",
        re.IGNORECASE,
    ), 4),

    # L3: multi-topic or complex questions
    (re.compile(
        r"\b(explain|compare|how does|what is the difference|walk me through|"
        r"summarise|overview|all.*about|tell me about|architecture)\b",
        re.IGNORECASE,
    ), 3),

    # L1: simple factual
    (re.compile(
        r"^(what|who|where|when|which|is|are|do|does|did|has|have)\b.{0,60}[?]?$",
        re.IGNORECASE,
    ), 1),
]

_DEFAULT_LEVEL: int = 2   # standard for unmatched queries


def classify_query(query: str) -> int:
    """
    Classify a query string into a budget level (0–5).

    Args:
        query: Raw query string from the user or agent.

    Returns:
        Integer budget level 0–5.
    """
    stripped = query.strip()

    for pattern, level in _QUERY_PATTERNS:
        if pattern.search(stripped):
            logger.debug("budget_engine: query classified as L%d", level)
            return level

    # Heuristic: longer queries tend to be more complex
    word_count = len(stripped.split())
    if word_count <= 5:
        return 1
    elif word_count <= 15:
        return 2
    elif word_count <= 30:
        return 3
    else:
        return 4


# ---------------------------------------------------------------------------
# Budget plan — output of budget allocation
# ---------------------------------------------------------------------------

@dataclass
class BudgetPlan:
    level:              int
    name:               str
    token_limit:        int
    priority_hot_slots: int
    hot_slots:          int
    warm_slots:         int
    full_slots:         int
    estimated_tokens:   int     # rough estimate before actual fetch
    queries:            list[str] = field(default_factory=list)  # search queries to run


# ---------------------------------------------------------------------------
# Budget engine
# ---------------------------------------------------------------------------

class BudgetEngine:
    """
    Allocates token budget for a retrieval call and coordinates
    with SemanticSearcher to build the final memory context.

    Usage:
        engine  = BudgetEngine(searcher)
        results = engine.retrieve("what editor does the user prefer?")
    """

    def __init__(
        self,
        searcher,
        default_level: int = _DEFAULT_LEVEL,
    ):
        """
        Args:
            searcher:      Initialised SemanticSearcher instance.
            default_level: Default budget level when auto-classify is off.
        """
        self._searcher      = searcher
        self._default_level = default_level

    # -----------------------------------------------------------------------
    # Main retrieval — auto-classifies query and builds context
    # -----------------------------------------------------------------------

    def retrieve(
        self,
        query:          str,
        level:          Optional[int] = None,
        memory_type:    Optional[str] = None,
        extra_queries:  Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve memories for a query within budget.

        Auto-classifies query into a budget level (unless level is given).
        Fetches PRIORITY_HOT → HOT → WARM within budget slots.

        Args:
            query:         User or agent query string.
            level:         Force a specific budget level (0–5). Auto if None.
            memory_type:   Optional TYPE filter.
            extra_queries: Additional queries to merge (multi-query mode).

        Returns:
            List of memory dicts with budget-appropriate content.
            Sorted: PRIORITY_HOT first, then HOT, then WARM — each by score.
        """
        # Determine budget level
        budget_level = level if level is not None else classify_query(query)
        budget = BUDGET_LEVELS.get(budget_level, BUDGET_LEVELS[_DEFAULT_LEVEL])

        logger.info(
            "budget_engine: L%d ('%s') — slots: P_HOT=%d HOT=%d WARM=%d full=%d",
            budget.level, budget.name,
            budget.priority_hot_slots, budget.hot_slots,
            budget.warm_slots, budget.full_slots,
        )

        # L0 greeting — priority-only, SHORT only
        if budget.level == 0:
            return self._retrieve_greeting(query, budget)

        # Standard retrieval — tier by tier
        all_queries = [query] + (extra_queries or [])
        results     = []
        seen_ids    = set()

        # 1. PRIORITY_HOT
        if budget.priority_hot_slots > 0:
            ph = self._searcher.search(
                query=query,
                n=budget.priority_hot_slots,
                memory_type=memory_type,
                include_warm=False,
            )
            ph_filtered = [r for r in ph if r["temperature"] == "PRIORITY_HOT"]
            for r in ph_filtered:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    results.append(r)

        # 2. HOT
        if budget.hot_slots > 0:
            hot = self._searcher.search(
                query=query,
                n=budget.priority_hot_slots + budget.hot_slots,
                memory_type=memory_type,
                include_warm=False,
            )
            hot_filtered = [r for r in hot if r["temperature"] == "HOT"]
            for r in hot_filtered[:budget.hot_slots]:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    results.append(r)

        # 3. WARM (if budget allows)
        if budget.warm_slots > 0:
            warm = self._searcher.search(
                query=query,
                n=budget.priority_hot_slots + budget.hot_slots + budget.warm_slots,
                memory_type=memory_type,
                include_warm=True,
            )
            warm_filtered = [r for r in warm if r["temperature"] == "WARM"]
            for r in warm_filtered[:budget.warm_slots]:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    results.append(r)

        # 4. Merge extra queries if provided
        if extra_queries:
            extra = self._searcher.search_multi(
                queries=extra_queries,
                n_per_query=3,
                total_limit=budget.hot_slots,
                include_warm=budget.warm_slots > 0,
            )
            for r in extra:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    results.append(r)

        # Tag each result with full_slots budget metadata
        self._tag_full_budget(results, budget.full_slots)

        logger.info(
            "budget_engine: retrieved %d memories for L%d query",
            len(results), budget.level,
        )

        return results

    # -----------------------------------------------------------------------
    # L0 greeting retrieval — ultra-lean, ≤120 tokens (P2-T10)
    # -----------------------------------------------------------------------

    def _retrieve_greeting(self, query: str, budget: BudgetLevel) -> list[dict]:
        """
        Greeting path: fetch PRIORITY_HOT SHORT only.
        Target ≤ 120 tokens total (P2-T10).
        """
        results = self._searcher.search_priority_only(
            query=query,
            n=budget.priority_hot_slots,
        )
        # All greeting results use SHORT (full_slot=0)
        self._tag_full_budget(results, full_slots=0)
        return results

    # -----------------------------------------------------------------------
    # Build budget plan (for inspection / debugging)
    # -----------------------------------------------------------------------

    def plan(self, query: str, level: Optional[int] = None) -> BudgetPlan:
        """
        Return a BudgetPlan without executing retrieval.
        Useful for debugging and tests.

        Args:
            query: Query string.
            level: Optional forced level.

        Returns:
            BudgetPlan dataclass.
        """
        budget_level = level if level is not None else classify_query(query)
        budget       = BUDGET_LEVELS.get(budget_level, BUDGET_LEVELS[_DEFAULT_LEVEL])

        est = (
            (budget.priority_hot_slots + budget.hot_slots + budget.warm_slots - budget.full_slots)
            * SHORT_TOKEN_ESTIMATE
            + budget.full_slots * FULL_TOKEN_ESTIMATE
        )

        return BudgetPlan(
            level=budget.level,
            name=budget.name,
            token_limit=budget.token_limit,
            priority_hot_slots=budget.priority_hot_slots,
            hot_slots=budget.hot_slots,
            warm_slots=budget.warm_slots,
            full_slots=budget.full_slots,
            estimated_tokens=max(0, est),
            queries=[query],
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _tag_full_budget(results: list[dict], full_slots: int) -> None:
        """
        Tag results with use_full flag.
        First `full_slots` PRIORITY_HOT results → use_full=True
        All others → use_full=False (use SHORT text)
        """
        full_count = 0
        for r in results:
            if full_count < full_slots and r["temperature"] == "PRIORITY_HOT":
                r["use_full"] = True
                full_count += 1
            else:
                r["use_full"] = False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_engine: Optional[BudgetEngine] = None


def get_budget_engine() -> BudgetEngine:
    """Return the global BudgetEngine instance."""
    global _global_engine
    if _global_engine is None:
        raise RuntimeError("BudgetEngine not initialised. Call init_budget_engine() first.")
    return _global_engine


def init_budget_engine(
    searcher,
    default_level: int = _DEFAULT_LEVEL,
) -> BudgetEngine:
    """
    Initialise the global BudgetEngine singleton.
    Call once at server startup after init_searcher().

    Args:
        searcher:      Initialised SemanticSearcher instance.
        default_level: Default budget level.

    Returns:
        Initialised BudgetEngine instance.
    """
    global _global_engine
    _global_engine = BudgetEngine(searcher, default_level)
    logger.info("budget_engine: global engine initialised (default_level=L%d)", default_level)
    return _global_engine