import re


TOOL_KEYWORDS = [
    "python", "javascript", "typescript", "fastapi", "django", "flask",
    "react", "nextjs", "vue", "svelte", "node", "express",
    "sqlite", "postgresql", "mysql", "mongodb", "redis", "supabase",
    "docker", "kubernetes", "git", "github", "linux", "windows",
    "claude", "openai", "gemini", "anthropic", "langchain",
    "chromadb", "fastmcp", "uv", "pip", "venv",
]

EMOTION_KEYWORDS = [
    "frustrated", "happy", "excited", "confused", "stuck",
    "love", "hate", "annoyed", "proud", "tired", "worried",
]

DECISION_KEYWORDS = [
    "decided", "choosing", "going with", "picked", "will use",
    "switched to", "moved to", "replaced", "dropped",
]

PROBLEM_KEYWORDS = [
    "error", "bug", "issue", "problem", "broken", "failing",
    "crash", "exception", "not working", "stuck on", "help with",
]

LEARNING_KEYWORDS = [
    "learning", "studying", "reading about", "trying to understand",
    "exploring", "practicing", "working on", "getting into",
]


def extract_entities(message: str) -> dict:
    msg_lower = message.lower()

    tools_found = [t for t in TOOL_KEYWORDS if t in msg_lower]

    emotions_found = [e for e in EMOTION_KEYWORDS if e in msg_lower]

    decisions_found = [d for d in DECISION_KEYWORDS if d in msg_lower]

    problems_found = [p for p in PROBLEM_KEYWORDS if p in msg_lower]

    learnings_found = [l for l in LEARNING_KEYWORDS if l in msg_lower]

    keywords = list(set(tools_found + emotions_found[:2]))
    words = re.findall(r'\b[a-zA-Z]{4,}\b', message)
    freq = {}
    for w in words:
        w_low = w.lower()
        if w_low not in ["this", "that", "with", "have", "from", "they", "will", "been", "were", "your"]:
            freq[w_low] = freq.get(w_low, 0) + 1
    top_words = sorted(freq, key=freq.get, reverse=True)[:5]
    keywords = list(set(keywords + top_words))[:7]

    return {
        "tools":     tools_found,
        "emotions":  emotions_found,
        "decisions": decisions_found,
        "problems":  problems_found,
        "learnings": learnings_found,
        "keywords":  keywords,
    }