from retrieval.topic_detector import detect_topic


def calculate_budget(message: str, memories: list) -> dict:
    topic    = detect_topic(message)
    level    = topic["level"]
    budget   = topic["budget"]

    if memories:
        top_importance = max(m.get("importance", 0.5) for m in memories)

        if top_importance > 0.90:
            budget += 100
        elif top_importance < 0.50:
            budget -= 80

        all_warm = all(m.get("temperature") == "WARM" for m in memories)
        any_priority = any(m.get("temperature") == "PRIORITY_HOT" for m in memories)

        if all_warm:
            budget -= 60
        if any_priority:
            budget += 80

    budget = max(80, min(budget, 800))

    use_full = budget >= 480

    return {
        "level":    level,
        "budget":   budget,
        "use_full": use_full,
    }