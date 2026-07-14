L0_SIGNALS = ["hi", "hello", "hey", "what's up", "howdy", "greetings"]

L1_SIGNALS = [
    "quick", "simple", "list", "name",
]

L2_SIGNALS = [
    "help", "how to", "understand", "describe",
    "what is", "what are", "can you",
]

L3_SIGNALS = [
    "code", "build", "debug", "error", "implement", "fix",
    "write", "create", "develop", "function", "script",
]

L4_SIGNALS = [
    "architecture", "full flow", "design", "entire system",
    "structure", "plan", "overview", "deep dive", "all of",
    "phase", "pipeline", "integration",
]

L5_SIGNALS = [
    "remember everything", "full context", "everything about",
    "complete history", "all memories", "everything you know",
    "what do you know about me", "know about me",
    "tell me everything", "full picture",
]

# ── Phase 2: Memory/identity queries → always L4 minimum ──────────────────────
# These questions need LOTS of context to answer well
MEMORY_QUERY_SIGNALS = [
    "what do you know", "what do you remember",
    "what project", "what are we", "what were we",
    "where were we", "where did we", "last session",
    "previously", "last time", "we discussed",
    "what have we", "tell me about caretaker",
    "about me", "know about", "our project",
    "what we built", "what we made", "caretaker",
    "phase 2", "phase 1", "phase 3",
]

LEVEL_BUDGET = {
    "L0": 80,
    "L1": 120,
    "L2": 280,
    "L3": 480,
    "L4": 700,
    "L5": 800,
}


def detect_topic(message: str) -> dict:
    msg_lower = message.lower()

    # ── Phase 2: memory/identity queries → L4 minimum ─────────────────────
    if any(s in msg_lower for s in MEMORY_QUERY_SIGNALS):
        return {"level": "L4", "budget": LEVEL_BUDGET["L4"]}

    # ── Phase 1: signal matching (unchanged logic) ─────────────────────────
    if any(s in msg_lower for s in L5_SIGNALS):
        level = "L5"
    elif any(s in msg_lower for s in L4_SIGNALS):
        level = "L4"
    elif any(s in msg_lower for s in L3_SIGNALS):
        level = "L3"
    elif any(s in msg_lower for s in L2_SIGNALS):
        level = "L2"
    elif any(s in msg_lower for s in L1_SIGNALS):
        level = "L1"
    elif any(s in msg_lower for s in L0_SIGNALS):
        level = "L0"
    else:
        level = "L2"

    return {
        "level":  level,
        "budget": LEVEL_BUDGET[level],
    }