PROJECT_SIGNALS = [
    "building", "working on", "project", "app", "system",
    "developing", "creating", "making", "coding", "implementing",
]

PREFERENCE_SIGNALS = [
    "prefer", "like to use", "always use", "favorite", "i use",
    "my setup", "i work with", "i choose", "rather use",
]

PROBLEM_SIGNALS = [
    "error", "bug", "issue", "broken", "failing", "crash",
    "exception", "not working", "stuck", "problem", "help with",
]

DECISION_SIGNALS = [
    "decided", "choosing", "going with", "picked", "switched",
    "will use", "moved to", "replaced", "dropped",
]

LEARNING_SIGNALS = [
    "learning", "studying", "reading", "exploring", "practicing",
    "trying to understand", "getting into", "working through",
]

PERSONAL_SIGNALS = [
    "i am", "i'm a", "my name", "i work at", "i live",
    "my job", "my role", "i'm from", "my background",
]

EMOTION_SIGNALS = [
    "frustrated", "excited", "happy", "confused", "annoyed",
    "proud", "tired", "worried", "love", "hate",
]

CORRECTION_SIGNALS = [
    "actually", "no wait", "correction", "i meant", "wrong",
    "not that", "let me correct", "i was wrong", "changed my mind",
]

REPLACEABLE_TYPES = {"PROJECT", "PREFERENCE", "PERSONAL", "CORRECTION"}
ADDITIVE_TYPES    = {"PROBLEM", "DECISION", "LEARNING", "EMOTION"}


def classify_type(message: str) -> dict:
    msg_lower = message.lower()

    scores = {
        "PROJECT":    sum(1 for s in PROJECT_SIGNALS    if s in msg_lower),
        "PREFERENCE": sum(1 for s in PREFERENCE_SIGNALS if s in msg_lower),
        "PROBLEM":    sum(1 for s in PROBLEM_SIGNALS    if s in msg_lower),
        "DECISION":   sum(1 for s in DECISION_SIGNALS   if s in msg_lower),
        "LEARNING":   sum(1 for s in LEARNING_SIGNALS   if s in msg_lower),
        "PERSONAL":   sum(1 for s in PERSONAL_SIGNALS   if s in msg_lower),
        "EMOTION":    sum(1 for s in EMOTION_SIGNALS    if s in msg_lower),
        "CORRECTION": sum(1 for s in CORRECTION_SIGNALS if s in msg_lower),
    }

    best_type = max(scores, key=scores.get)

    if scores[best_type] == 0:
        best_type = "LEARNING"

    fact_type = "REPLACEABLE" if best_type in REPLACEABLE_TYPES else "ADDITIVE"

    importance = min(0.5 + (scores[best_type] * 0.1), 0.9)

    return {
        "type":       best_type,
        "subtype":    best_type.lower(),
        "fact_type":  fact_type,
        "importance": round(importance, 2),
    }