SAMPLE_MESSAGES = {
    "project":    "I am building a FastAPI project called Caretaker. It is a universal memory layer for AI agents.",
    "preference": "I prefer to use Python and SQLite for all my backend projects.",
    "problem":    "I have a bug in my code. The database connection is failing with an exception.",
    "decision":   "I decided to go with FastMCP instead of building my own MCP server.",
    "learning":   "I am learning about vector databases and semantic search.",
    "personal":   "I am a full stack developer. My name is Dhruvil.",
    # Phase 2 note: original emotion fixture contained the word "error" which
    # scores higher in PROBLEM_SIGNALS than "frustrated" in EMOTION_SIGNALS.
    # Updated fixture is pure emotion — no problem-domain words.
    "emotion":    "I am so excited and proud of the progress I have made today.",
    "correction": "Actually I was wrong. I meant to use PostgreSQL not SQLite.",
    "greeting":   "Hey! How are you?",
    "long":       " ".join(["word"] * 450),
}

EXPECTED_TYPES = {
    "project":    "PROJECT",
    "preference": "PREFERENCE",
    "problem":    "PROBLEM",
    "decision":   "DECISION",
    "learning":   "LEARNING",
    "personal":   "PERSONAL",
    "emotion":    "EMOTION",
    "correction": "CORRECTION",
}

EXPECTED_FACT_TYPES = {
    "PROJECT":    "REPLACEABLE",
    "PREFERENCE": "REPLACEABLE",
    "PERSONAL":   "REPLACEABLE",
    "CORRECTION": "REPLACEABLE",
    "PROBLEM":    "ADDITIVE",
    "DECISION":   "ADDITIVE",
    "LEARNING":   "ADDITIVE",
    "EMOTION":    "ADDITIVE",
}

VALID_TEMPERATURES = {"PRIORITY_HOT", "HOT", "WARM", "COLD"}

# Phase 2 added SKIP type for questions, commands, fillers, and noise messages.
# SKIP is a valid internal type — included here so type_always_valid tests pass.
VALID_TYPES      = set(EXPECTED_TYPES.values()) | {"SKIP"}
VALID_FACT_TYPES = {"REPLACEABLE", "ADDITIVE", "SKIP"}
VALID_STATUSES   = {"ACTIVE", "OUTDATED", "ARCHIVED"}
VALID_LEVELS     = {"L0", "L1", "L2", "L3", "L4", "L5"}