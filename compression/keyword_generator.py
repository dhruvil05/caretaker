"""
compression/keyword_generator.py
Extract 3-7 keywords from SHORT summary text.
Used as fallback or post-processing step.
Works without any API key — uses spaCy noun chunks + frequency scoring.
"""

import re
from typing import List


def extract_keywords(text: str, max_keywords: int = 7, min_keywords: int = 3) -> List[str]:
    """
    Extract 3-7 keywords from any text.
    Strategy:
      1. Try spaCy noun chunks (best quality, no API)
      2. Fallback to frequency-based extraction if spaCy unavailable
    """
    if not text or not text.strip():
        return []

    keywords = _extract_with_spacy(text)

    if len(keywords) < min_keywords:
        keywords = _extract_with_frequency(text)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique.append(kw)

    # Clamp to min/max range
    result = unique[:max_keywords]
    return result if len(result) >= min_keywords else result


def _extract_with_spacy(text: str) -> List[str]:
    """Extract keywords using spaCy noun chunks and named entities."""
    try:
        import spacy

        # Try to load a model — use smallest available
        for model_name in ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]:
            try:
                nlp = spacy.load(model_name)
                break
            except OSError:
                continue
        else:
            return []

        doc = nlp(text[:500])  # Limit input length for speed

        keywords = []

        # Named entities first (highest value)
        for ent in doc.ents:
            if ent.label_ not in ("CARDINAL", "ORDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY"):
                clean = ent.text.strip()
                if len(clean) > 2:
                    keywords.append(clean)

        # Noun chunks second
        for chunk in doc.noun_chunks:
            clean = chunk.root.text.strip()
            if len(clean) > 2 and clean.lower() not in ("this", "that", "it", "they", "we", "you", "i"):
                keywords.append(clean)

        return keywords

    except ImportError:
        return []
    except Exception:
        return []


def _extract_with_frequency(text: str) -> List[str]:
    """
    Fallback: simple frequency-based keyword extraction.
    No external dependencies needed.
    """
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
        "so", "yet", "both", "either", "neither", "each", "few", "more", "most",
        "other", "some", "such", "than", "too", "very", "just", "that", "this",
        "these", "those", "it", "its", "we", "our", "they", "their", "you",
        "your", "i", "my", "me", "he", "she", "his", "her", "him", "user",
        "using", "used", "use", "also", "about", "into", "if", "as", "up",
        "what", "when", "who", "how", "which", "there", "then", "now", "all",
    }

    # Tokenize: keep alphanumeric words only
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b', text)

    # Count frequency, skip stopwords
    freq: dict = {}
    for word in words:
        lower = word.lower()
        if lower not in STOPWORDS:
            freq[lower] = freq.get(lower, 0) + 1

    # Sort by frequency descending
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    # Return top words, preserving original casing from first occurrence
    word_map = {}
    for word in words:
        if word.lower() not in word_map:
            word_map[word.lower()] = word

    keywords = [word_map.get(w, w) for w, _ in sorted_words]
    return keywords