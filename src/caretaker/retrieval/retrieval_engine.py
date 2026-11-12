from caretaker.retrieval.keyword_extractor import extract_keywords
from caretaker.retrieval.budget_engine import calculate_budget
from caretaker.storage.local_db import (
    get_memories_by_type,
    get_recent_memories,
    get_top_memories,
    record_recalls,
    search_fulltext,
)

# Semantic similarity below this is "not really about this message": such
# memories are not injected as relevant and not counted/reheated as recalled.
# Scale: (1 + cosine) / 2, so unrelated text sits around 0.5. Configurable via
# "min_relevance" in config.json.
DEFAULT_MIN_RELEVANCE = 0.6


def _min_relevance() -> float:
    try:
        from caretaker.paths import load_config
        return float(load_config().get("min_relevance", DEFAULT_MIN_RELEVANCE))
    except (TypeError, ValueError):
        return DEFAULT_MIN_RELEVANCE


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


def _fill_missing_summaries(memories: list) -> list:
    """Memories whose summary isn't made yet get a quick one for this recall, so
    the budget can still pick the short form and cold memories aren't dropped."""
    from caretaker.compression.local_compressor import quick_summary
    for mem in memories:
        if not str(mem.get("short") or "").strip():
            full = mem.get("full") or mem.get("full_text") or ""
            if full.strip():
                mem["short"] = quick_summary(full)
                mem["_quick_summary"] = True
    return memories


def retrieve_context(
    message: str,
    agent_id: str = "claude",
    semantic_searcher=None,   # Phase 2: optional SemanticSearcher instance
    memory_selector=None,     # Phase 2: optional memory_selector module
    project: str = "",        # 0.5: '' = all memories; else that project + global
    dry_run: bool = False,    # 0.5: preview only — don't count recalls or reheat
) -> dict:
    in_scope = (lambda m: (m.get("project") or "") in (project, "")) if project else (lambda m: True)
    scope = project or None

    keywords = extract_keywords(message)

    # ── Phase 2: semantic search path ─────────────────────────────────────
    if semantic_searcher:
        # force_include_cold=True: a relevant memory that decayed to COLD would be
        # invisible otherwise. The rebalanced ranking demotes COLD, so it won't
        # dominate results — but it can now surface when genuinely relevant.
        # Ask for more when scoped, since other projects' hits are filtered out.
        hits = semantic_searcher.search(query=message, n_results=40 if project else 20, force_include_cold=True)
        threshold = _min_relevance()
        relevant = [m for m in hits if in_scope(m) and m.get("semantic_similarity", 1.0) >= threshold][:20]

        if relevant:
            # Reheat + count only memories that genuinely matched, in one transaction.
            from caretaker.memory.temperature_engine import reheat
            recalls = []
            for mem in relevant:
                new_temp = reheat(mem.get("temperature", "WARM"))
                changed = new_temp != mem.get("temperature")
                if changed:
                    mem["temperature"] = new_temp
                recalls.append((mem["id"], new_temp if changed else None))
            if not dry_run:
                record_recalls(recalls)

            _fill_missing_summaries(relevant)
            budget_info = calculate_budget(message, relevant)

            # Phase 2: memory selector picks SHORT or FULL per memory
            if memory_selector:
                selected, tokens_used = memory_selector.select_memory_forms(
                    memories=relevant,
                    token_budget=budget_info["budget"],
                )
            else:
                selected = relevant

            recent = get_recent_memories(limit=3, project=scope)

            return {
                "project":    project,
                "relevant":   selected,
                "recent":     recent,
                "budget":     budget_info["budget"],
                "use_full":   budget_info["use_full"],
                "level":      budget_info["level"],
                "n_results":  budget_info.get("n_results", 10),  # Phase 2
                "keywords":   keywords,
            }

        # Semantic search returned nothing (e.g. vector store empty/un-indexed).
        # Fall through to the keyword path below so the user still gets relevant
        # memories instead of only the 3 most-recent ones.

    # ── Keyword path (no vector search, or nothing semantically relevant) ──
    # 0.5: ranked full-text search instead of loading every memory.
    # Search with the extracted keywords AND the message's own words: the
    # extractor can drop rare names (e.g. "Zylo003"), which are exactly the
    # words BM25 ranks highest. Stopwords are removed in search_fulltext.
    relevant = search_fulltext(
        [*keywords, message], limit=10,
        temperatures=("PRIORITY_HOT", "HOT", "WARM"),
        project=scope,
    )
    if not dry_run:
        record_recalls([(m["id"], None) for m in relevant])

    if not relevant:
        # Nothing matched: give general background, but don't count it as recalled.
        relevant = [{**m, "_reason": "background"} for m in get_top_memories(limit=5, project=scope)]

    _fill_missing_summaries(relevant)
    budget_info = calculate_budget(message, relevant)

    recent = get_recent_memories(limit=3, project=scope)

    return {
        "project":    project,
        "relevant":   relevant,
        "recent":     recent,
        "budget":     budget_info["budget"],
        "use_full":   budget_info["use_full"],
        "level":      budget_info["level"],
        "n_results":  budget_info.get("n_results", 10),  # Phase 2
        "keywords":   keywords,
    }