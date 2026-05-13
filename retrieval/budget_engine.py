from retrieval.topic_detector import detect_topic


def calculate_budget(message: str, memories: list) -> dict:
    """
    Phase 2 — Dynamic budget engine (L0–L5).
    Preserves Phase 1 importance + temperature modifiers.
    Adds Phase 2: L0–L5 level classification, match quality modifier,
    use_full threshold, and n_results per level.
    """

    # ── Phase 1: topic detection (unchanged) ──────────────────────────────
    topic  = detect_topic(message)
    level  = int(str(topic["level"]).replace("L", "").replace("l", ""))  # Phase 2 fix
    budget = int(topic["budget"])

    # ── Phase 1: importance + temperature modifiers (unchanged) ───────────
    if memories:
        top_importance = max(m.get("importance", 0.5) for m in memories)

        if top_importance > 0.90:
            budget += 100
        elif top_importance < 0.50:
            budget -= 80

        all_warm      = all(m.get("temperature") == "WARM" for m in memories)
        any_priority  = any(m.get("temperature") == "PRIORITY_HOT" for m in memories)

        if all_warm:
            budget -= 60
        if any_priority:
            budget += 80

    # ── Phase 2: match quality modifier (NEW) ─────────────────────────────
    # Strong semantic match → +20%, Weak match → -30%
    if memories:
        top_score = max(m.get("relevance_score", 0.0) for m in memories)

        if top_score >= 0.8:
            budget = int(budget * 1.20)   # +20% — strong match, use more context
        elif top_score < 0.3:
            budget = int(budget * 0.70)   # -30% — weak match, don't waste tokens

    # ── Clamp (Phase 1 bounds preserved) ──────────────────────────────────
    budget = max(80, min(budget, 800))

    # ── Phase 2: use_full threshold (replaces simple Phase 1 boolean) ─────
    # Phase 1 used budget >= 480. Phase 2 uses L-level for finer control.
    use_full = level >= 3 or budget >= 480

    # ── Phase 2: n_results per level (NEW) ────────────────────────────────
    n_results_map = {
        0: 3,
        1: 5,
        2: 8,
        3: 12,
        4: 15,
        5: 20,
    }
    n_results = n_results_map.get(level, 8)

    return {
        # Phase 1 keys — unchanged so nothing breaks
        "level":    level,
        "budget":   budget,
        "use_full": use_full,

        # Phase 2 additions
        "n_results": n_results,
    }