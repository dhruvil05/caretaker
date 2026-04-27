PROJECT_SIGNALS = [
    'building', 'working on', 'project', 'app', 'system',
    'developing', 'creating', 'making', 'coding', 'implementing',
]

PREFERENCE_SIGNALS = [
    'prefer', 'like to use', 'always use', 'favorite', 'i use',
    'my setup', 'i work with', 'i choose', 'rather use', 'i prefer',
]

PROBLEM_SIGNALS = [
    'error', 'bug', 'issue', 'broken', 'failing', 'crash',
    'exception', 'not working', 'stuck', 'problem', 'help with',
]

DECISION_SIGNALS = [
    'decided', 'going with', 'picked', 'switched',
    'will use', 'moved to', 'replaced', 'dropped', 'instead of',
]

LEARNING_SIGNALS = [
    'learning', 'studying', 'reading', 'exploring', 'practicing',
    'trying to understand', 'getting into', 'working through',
]

PERSONAL_SIGNALS = [
    'i am a', "i'm a", 'my name', 'i work at', 'i live',
    'my job', 'my role', "i'm from", 'my background',
]

EMOTION_SIGNALS = [
    'frustrated', 'excited', 'happy', 'confused', 'annoyed',
    'proud', 'tired', 'worried', 'love this', 'hate this',
]

CORRECTION_SIGNALS = [
    'actually', 'no wait', 'correction', 'i meant', 'wrong',
    'not that', 'let me correct', 'i was wrong', 'changed my mind',
]

REPLACEABLE_TYPES = {"PROJECT", "PREFERENCE", "PERSONAL", "CORRECTION"}
ADDITIVE_TYPES    = {"PROBLEM", "DECISION", "LEARNING", "EMOTION"}

SIGNAL_MAP = {
    "CORRECTION": CORRECTION_SIGNALS,
    "PERSONAL":   PERSONAL_SIGNALS,
    "EMOTION":    EMOTION_SIGNALS,
    "PROBLEM":    PROBLEM_SIGNALS,
    "DECISION":   DECISION_SIGNALS,
    "PREFERENCE": PREFERENCE_SIGNALS,
    "LEARNING":   LEARNING_SIGNALS,
    "PROJECT":    PROJECT_SIGNALS,
}


def classify_type(message: str) -> dict:
    msg_lower = message.lower()

    scores = {}
    for mtype, signals in SIGNAL_MAP.items():
        scores[mtype] = sum(1 for s in signals if s in msg_lower)

    best_type = max(scores, key=scores.get)

    if scores[best_type] == 0:
        best_type = "LEARNING"

    fact_type = "REPLACEABLE" if best_type in REPLACEABLE_TYPES else "ADDITIVE"

    importance = min(0.5 + (scores[best_type] * 0.1), 0.9)

    return {
        "type":      best_type,
        "subtype":   best_type.lower(),
        "fact_type": fact_type,
        "importance": round(importance, 2),
    }
