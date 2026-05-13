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

# ── Phase 2: Skip detection ───────────────────────────────────────────────────

# Messages starting with these words = questions = SKIP
QUESTION_STARTERS = [
    "what ", "who ", "where ", "when ", "why ", "how ",
    "do you", "can you", "could you", "did you", "does ",
    "is ", "are ", "have you", "tell me", "show me",
    "give me", "let me know", "any idea", "you know",
    "you must", "do we", "don't we", "should we",
    "will you", "would you", "shall we", "before starting",
    "now start", "for that", "okay ", "ok ", "sure",
    "now give", "now tell", "now show", "now let",
    "and give", "also give", "then give", "keep in mind",
    "make sure", "make a plan", "can we", "are you sure",
]

# Messages ending with these = questions = SKIP
QUESTION_ENDINGS = ["?"]

# Short filler messages = SKIP
FILLER_PHRASES = [
    "ok", "okay", "sure", "got it", "thanks", "thank you",
    "hi", "hello", "hey", "yes", "no", "maybe", "alright",
    "sounds good", "great", "nice", "cool", "done", "continue",
    "next", "go ahead", "proceed", "start", "begin",
]

# Min length to be worth saving
MIN_USEFUL_LENGTH = 20


def is_question_or_noise(message: str) -> bool:
    """
    Return True if message is a question, command, filler, or conversation noise.
    These should NOT be saved as memories.
    """
    msg = message.strip()
    msg_lower = msg.lower()

    # Too short
    if len(msg) < MIN_USEFUL_LENGTH:
        return True

    # Pure filler
    if msg_lower in FILLER_PHRASES:
        return True

    # Ends with question mark
    if msg_lower.endswith("?"):
        return True

    # Starts with question/command word
    for starter in QUESTION_STARTERS:
        if msg_lower.startswith(starter):
            return True

    # Contains ONLY question/command — no factual content
    # e.g. "now start with next and keep mind from now on..."
    command_only_signals = [
        "start with next", "give me files", "give me all",
        "let me know if", "keep in mind", "from now on",
        "as you finish", "finish give me", "according to docs",
        "what do you mean", "are you sure that",
    ]
    for signal in command_only_signals:
        if signal in msg_lower:
            return True

    return False


def classify_type(message: str) -> dict:
    msg_lower = message.lower()

    # ── Phase 2: skip questions, commands, fillers ─────────────────────────
    if is_question_or_noise(message):
        return {
            "type":       "SKIP",
            "subtype":    "noise",
            "fact_type":  "SKIP",
            "importance": 0.0,
        }

    # ── Phase 1: signal scoring (unchanged) ───────────────────────────────
    scores = {}
    for mtype, signals in SIGNAL_MAP.items():
        scores[mtype] = sum(1 for s in signals if s in msg_lower)

    best_type = max(scores, key=scores.get)

    if scores[best_type] == 0:
        best_type = "LEARNING"

    fact_type  = "REPLACEABLE" if best_type in REPLACEABLE_TYPES else "ADDITIVE"
    importance = min(0.5 + (scores[best_type] * 0.1), 0.9)

    return {
        "type":       best_type,
        "subtype":    best_type.lower(),
        "fact_type":  fact_type,
        "importance": round(importance, 2),
    }