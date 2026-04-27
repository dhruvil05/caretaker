import re

STOP_WORDS = {
    "this", "that", "with", "have", "from", "they", "will",
    "been", "were", "your", "about", "would", "could", "should",
    "what", "when", "where", "which", "there", "their", "then",
    "than", "just", "also", "some", "into", "more", "very",
}


def extract_keywords(message: str) -> list:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', message.lower())

    freq = {}
    for w in words:
        if w not in STOP_WORDS:
            freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq, key=freq.get, reverse=True)

    return sorted_words[:7]