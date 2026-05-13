from retrieval.keyword_extractor import extract_keywords
from retrieval.budget_engine import calculate_budget
from storage.local_db import (
    get_memories_by_type,
    get_recent_memories,
    get_all_active_memories,
    increment_retrieval_count,
    touch_last_accessed,        # Phase 2
)


# ── Phase 1: keyword match scorer (kept as fallback) ──────────────────────────
def _keyword_match_score(memory: dict, keywords: list) -> float:
    mem_keywords = memory.get("keywords") or "[]"
    try:
        import json
        mem_kws = json.loads(mem_keywords)
    except Exception:
        mem_kws = []

    full_text = (memory.get("full") or "").lower()
    hits = sum(1 for kw in keywords if kw in full_text or kw in mem_kws)

    return hits / max(len(keywords), 1)


def retrieve_context(
    message: str,
    agent_id: str = "claude",
    semantic_searcher=None,   # Phase 2: optional SemanticSearcher instance
    memory_selector=None,     # Phase 2: optional memory_selector module
) -> dict:

    keywords = extract_keywords(message)

    # ── Phase 2: semantic search path ─────────────────────────────────────
    if semantic_searcher:
        relevant = semantic_searcher.search(query=message, n_results=20)

        # Phase 2: reheat accessed memories + update last_accessed_at
        for mem in relevant:
            increment_retrieval_count(mem["id"])
            touch_last_accessed(mem["id"])

            # Reheat temperature in memory engine + update DB
            from memory.temperature_engine import reheat
            from storage.local_db import update_temperature
            new_temp = reheat(mem.get("temperature", "WARM"))
            if new_temp != mem.get("temperature"):
                update_temperature(mem["id"], new_temp)
                mem["temperature"] = new_temp

        budget_info = calculate_budget(message, relevant)

        # Phase 2: memory selector picks SHORT or FULL per memory
        if memory_selector:
            selected, tokens_used = memory_selector.select_memory_forms(
                memories=relevant,
                token_budget=budget_info["budget"],
            )
        else:
            selected = relevant

        recent = get_recent_memories(limit=3)

        return {
            "relevant":   selected,
            "recent":     recent,
            "budget":     budget_info["budget"],
            "use_full":   budget_info["use_full"],
            "level":      budget_info["level"],
            "n_results":  budget_info.get("n_results", 10),  # Phase 2
            "keywords":   keywords,
        }

    # ── Phase 1 fallback: keyword search (unchanged) ──────────────────────
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

    recent = get_recent_memories(limit=3)

    return {
        "relevant":   relevant,
        "recent":     recent,
        "budget":     budget_info["budget"],
        "use_full":   budget_info["use_full"],
        "level":      budget_info["level"],
        "n_results":  budget_info.get("n_results", 10),  # Phase 2
        "keywords":   keywords,
    }