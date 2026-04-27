L0_SIGNALS = ["hi", "hello", "hey", "what's up", "howdy", "greetings"]

L1_SIGNALS = [
    "what", "when", "who", "where", "quick", "simple",
    "tell me", "show me", "list", "give me",
]

L2_SIGNALS = [
    "explain", "help", "how to", "understand", "why",
    "describe", "what is", "what are", "can you",
]

L3_SIGNALS = [
    "code", "build", "debug", "error", "implement", "fix",
    "write", "create", "develop", "function", "script",
]

L4_SIGNALS = [
    "architecture", "full flow", "design", "entire system",
    "structure", "plan", "overview", "deep dive", "all of",
]

L5_SIGNALS = [
    "remember everything", "full context", "everything about",
    "complete history", "all memories", "everything you know",
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