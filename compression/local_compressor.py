"""
compression/local_compressor.py
FREE local compression — no API key needed.
Uses spaCy + TextRank sentence scoring to generate SHORT summary.
Uses keyword_generator for KEYWORDS.
Quality: Medium. Speed: Very Fast. Cost: Zero.
"""

import re
import math
from typing import Tuple, List
from compression.keyword_generator import extract_keywords


def compress_local(full_text: str, memory_type: str, max_short_tokens: int = 60) -> Tuple[str, List[str]]:
    """
    Generate SHORT summary + KEYWORDS from full_text without any API call.
    Returns: (short: str, keywords: List[str])
    """
    if not full_text or not full_text.strip():
        return ("", [])

    # Try spaCy TextRank first
    short = _textrank_summary(full_text, max_sentences=2)

    # If spaCy not available or result too long, fall back to first-sentence extraction
    if not short:
        short = _first_sentence_summary(full_text)

    # Trim to approximate token limit (1 token ≈ 4 chars)
    max_chars = max_short_tokens * 4
    if len(short) > max_chars:
        short = short[:max_chars].rsplit(" ", 1)[0] + "..."

    # Generate keywords from SHORT (not full_text — keeps keywords tight)
    keywords = extract_keywords(short or full_text, max_keywords=7, min_keywords=3)

    return (short.strip(), keywords)


def _textrank_summary(text: str, max_sentences: int = 2) -> str:
    """
    TextRank-inspired sentence scoring using spaCy.
    Ranks sentences by similarity to document centroid.
    """
    try:
        import spacy

        for model_name in ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]:
            try:
                nlp = spacy.load(model_name)
                break
            except OSError:
                continue
        else:
            return ""

        doc = nlp(text[:1000])  # Limit for speed

        # Split into sentences
        sentences = [sent for sent in doc.sents if len(sent.text.strip()) > 10]

        if not sentences:
            return ""

        if len(sentences) == 1:
            return sentences[0].text.strip()

        # Score each sentence by similarity to all others (TextRank simplified)
        scores = {}
        for i, sent in enumerate(sentences):
            score = 0.0
            for j, other in enumerate(sentences):
                if i != j:
                    try:
                        sim = sent.similarity(other)
                        score += sim if not math.isnan(sim) else 0.0
                    except Exception:
                        pass
            scores[i] = score

        # Pick top N sentences, preserving original order
        top_indices = sorted(scores, key=lambda x: scores[x], reverse=True)[:max_sentences]
        top_indices.sort()  # Restore original order

        summary = " ".join(sentences[i].text.strip() for i in top_indices)
        return summary

    except ImportError:
        return ""
    except Exception:
        return ""


def _first_sentence_summary(text: str) -> str:
    """
    Ultra-simple fallback: take first 1-2 meaningful sentences.
    No dependencies required.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return text[:200].strip()

    # Take up to 2 sentences
    result = " ".join(sentences[:2])
    return result