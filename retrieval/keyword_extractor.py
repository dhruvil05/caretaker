import re

STOP_WORDS = {
    # Phase 1 originals
    "this", "that", "with", "have", "from", "they", "will",
    "been", "were", "your", "about", "would", "could", "should",
    "what", "when", "where", "which", "there", "their", "then",
    "than", "just", "also", "some", "into", "more", "very",

    # Phase 2: missing common words that caused bad keyword matching
    "you", "me", "my", "we", "our", "us", "i", "he", "she",
    "it", "its", "they", "them", "the", "a", "an", "and", "or",
    "but", "in", "on", "at", "to", "for", "of", "by", "as",
    "is", "are", "was", "am", "be", "do", "did", "does", "done",
    "get", "got", "has", "had", "not", "no", "nor", "so", "yet",
    "all", "any", "few", "can", "may", "might", "shall", "let",
    "know", "tell", "show", "give", "take", "make", "see", "say",
    "ask", "use", "go", "come", "how", "why", "who", "too",
    "now", "new", "old", "own", "same", "each", "both", "here",
    "over", "out", "up", "off", "right", "left", "back", "down",
    "working", "work", "need", "want", "like", "think", "look",
    "way", "well", "still", "even", "much", "many", "two", "one",
    "first", "last", "next", "after", "before", "again", "while",
    "every", "through", "between", "being", "having", "doing",
    "these", "those", "such", "only", "then", "else", "per",
    "find", "help", "try", "run", "start", "end", "add", "set",
    "put", "call", "name", "time", "day", "good", "great", "sure",
    "okay", "yes", "yep", "nope", "please", "thank", "thanks",
}


def extract_keywords(message: str) -> list:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', message.lower())

    freq = {}
    for w in words:
        if w not in STOP_WORDS:
            freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq, key=freq.get, reverse=True)

    return sorted_words[:7]