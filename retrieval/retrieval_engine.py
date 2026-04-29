import asyncio
from storage.local_db import (
    get_all_active_memories,
    increment_retrieval_count,
)

# ── Phase 1 imports (kept as fallback) ───────────────────────────────────────
from retrieval.keyword_extractor import extract_keywords

# ── Phase 2 imports ──────────────────────────────────────────────────────────
from retrieval.budget_engine import get_budget_engine, classify_query
from retrieval.semantic_searcher import get_searcher
from retrieval.memory_selector import get_selector, format_whisper
# ─────────────────────────────────────────────────────────────────────────────


def retrieve_context(message: str, agent_id: str = "claude") -> dict:
    """
    Main retrieval pipeline — upgraded for Phase 2.

    Changes vs Phase 1:
      - budget_engine replaces fixed calculate_budget()
      - semantic_searcher (ChromaDB) replaces keyword_match_score()
      - memory_selector chooses SHORT or FULL per memory
      - Falls back to Phase 1 keyword search if Phase 2 not initialised

    Returns dict compatible with Phase 1 injector (relevant, recent, budget, use_full, level, keywords).
    """
    return asyncio.run(_retrieve_async(message, agent_id))


async def _retrieve_async(message: str, agent_id: str = "claude") -> dict:

    keywords = extract_keywords(message)

    # ── Try Phase 2 path first ────────────────────────────────────────────────
    try:
        engine   = get_budget_engine()
        searcher = get_searcher()
        selector = get_selector()

        # Step 1: Budget engine — classify query + retrieve ranked memories
        results = engine.retrieve(query=message)

        # Step 2: Memory selector — choose SHORT or FULL per memory
        budget_level  = classify_query(message)
        token_budgets = {0: 120, 1: 300, 2: 600, 3: 1200, 4: 2400, 5: 4000}
        token_budget  = token_budgets.get(budget_level, 600)

        selected = await selector.select(results, token_budget=token_budget)

        # Step 3: Increment retrieval counts in SQLite
        for mem in selected:
            try:
                increment_retrieval_count(mem["id"])
            except Exception:
                pass

        # Step 4: Build use_full list for injector compatibility
        use_full = [r.get("content_type") == "FULL" for r in selected]

        print(f"[RETRIEVAL] Phase 2 — L{budget_level} | {len(selected)} memories | {token_budget} token budget")

        return {
            "relevant": selected,
            "recent":   [],           # Phase 3 handles recent per-agent
            "budget":   token_budget,
            "use_full": use_full,
            "level":    budget_level,
            "keywords": keywords,
        }

    except RuntimeError:
        # Phase 2 singletons not initialised — fall back to Phase 1 path
        print("[RETRIEVAL] Phase 2 not ready — falling back to Phase 1 keyword search")
        return _retrieve_phase1_fallback(message, keywords)

    except Exception as e:
        print(f"[RETRIEVAL] Phase 2 error ({e}) — falling back to Phase 1 keyword search")
        return _retrieve_phase1_fallback(message, keywords)


def _retrieve_phase1_fallback(message: str, keywords: list) -> dict:
    """
    Phase 1 keyword-based retrieval — used as fallback when Phase 2
    singletons (budget engine, searcher, selector) are not yet initialised.
    """
    from retrieval.budget_engine import calculate_budget

    all_mems = get_all_active_memories()
    hot_mems = [
        m for m in all_mems
        if m.get("temperature") in ("PRIORITY_HOT", "HOT", "WARM")
    ]

    scored = []
    for mem in hot_mems:
        score = _keyword_match_score(mem, keywords)
        scored.append((score, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    relevant = [m for _, m in scored if _ > 0][:10]

    if not relevant:
        relevant = hot_mems[:5]

    budget_info = calculate_budget(message, relevant)

    for mem in relevant:
        increment_retrieval_count(mem["id"])

    return {
        "relevant": relevant,
        "recent":   [],
        "budget":   budget_info.get("budget", 600),
        "use_full": budget_info.get("use_full", False),
        "level":    budget_info.get("level", 2),
        "keywords": keywords,
    }


def _keyword_match_score(memory: dict, keywords: list) -> float:
    """Phase 1 keyword overlap scorer — kept for fallback path."""
    import json
    mem_keywords = memory.get("keywords") or "[]"
    try:
        mem_kws = json.loads(mem_keywords)
    except Exception:
        mem_kws = []

    full_text = (memory.get("full") or "").lower()
    hits = sum(1 for kw in keywords if kw in full_text or kw in mem_kws)
    return hits / max(len(keywords), 1)