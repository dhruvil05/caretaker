"""
compression/keyword_generator.py
---------------------------------
Extract 3-7 keywords from a SHORT memory summary.

Two modes:
  1. API mode  — calls Haiku via Anthropic SDK (accurate, async-safe)
  2. Local mode — fast regex/NLP fallback when API unavailable

Phase 2 — Compression Tribe
"""

import json
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop words — excluded from local keyword extraction
# ---------------------------------------------------------------------------

STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "he", "she", "they", "we", "i", "you",
    "my", "your", "our", "their", "his", "her", "user", "memory", "project",
    "thing", "using", "used", "use", "also", "over", "after", "because",
    "when", "if", "then", "so", "as", "than", "into", "about", "during",
    "now", "status", "via", "per", "within", "known", "context", "phase",
}

# ---------------------------------------------------------------------------
# Local keyword extractor — no API required
# ---------------------------------------------------------------------------

def _extract_local(short_text: str, min_kw: int = 3, max_kw: int = 7) -> list[str]:
    """
    Fast local keyword extraction using token scoring.

    Strategy:
    - Tokenise on word boundaries
    - Score tokens by: capitalisation, length, position
    - Remove stop words
    - Return top N by score

    Args:
        short_text: The SHORT memory summary (≤60 tokens).
        min_kw:     Minimum keywords to return.
        max_kw:     Maximum keywords to return.

    Returns:
        List of lowercase keyword strings. Length between min_kw and max_kw.
    """
    if not short_text or not short_text.strip():
        return []

    # Tokenise: letters, numbers, hyphens inside words
    raw_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]*[a-zA-Z0-9]|[a-zA-Z]{2,}", short_text)

    scored: dict[str, float] = {}

    for i, token in enumerate(raw_tokens):
        lower = token.lower()

        # Skip stop words and very short tokens
        if lower in STOP_WORDS or len(lower) < 3:
            continue

        score = 1.0

        # Boost: proper noun (starts with capital, not at sentence start)
        if token[0].isupper() and i > 0:
            score += 1.5

        # Boost: longer tokens tend to be more specific
        if len(lower) >= 6:
            score += 0.5
        if len(lower) >= 9:
            score += 0.5

        # Boost: tokens with digits (version numbers, tech terms)
        if re.search(r"\d", lower):
            score += 1.0

        # Boost: hyphenated compound words
        if "-" in lower:
            score += 0.5

        # Accumulate score if token appears multiple times
        if lower in scored:
            scored[lower] += score * 0.5  # diminishing return on repeat
        else:
            scored[lower] = score

    # Sort by score descending, take top max_kw
    sorted_kw = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    keywords = [kw for kw, _ in sorted_kw[:max_kw]]

    # Pad with highest-score leftovers if under minimum
    if len(keywords) < min_kw and len(sorted_kw) >= min_kw:
        keywords = [kw for kw, _ in sorted_kw[:min_kw]]

    return keywords


# ---------------------------------------------------------------------------
# Parse Haiku JSON output safely
# ---------------------------------------------------------------------------

def _parse_keyword_response(raw_response: str) -> Optional[list[str]]:
    """
    Parse the Haiku API response for keywords.
    Expects a JSON array of strings. Handles minor formatting issues.

    Args:
        raw_response: Raw text response from Haiku.

    Returns:
        List of keyword strings, or None if parsing fails.
    """
    if not raw_response:
        return None

    text = raw_response.strip()

    # Strip any accidental markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Find JSON array in the response
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return None

    try:
        keywords = json.loads(match.group())
        if isinstance(keywords, list):
            # Clean: lowercase, strip whitespace, remove empties
            cleaned = [str(k).lower().strip() for k in keywords if k]
            return cleaned[:7] if cleaned else None
    except json.JSONDecodeError:
        logger.warning("keyword_generator: JSON parse failed on: %s", text[:100])
        return None

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_keywords_api(
    short_text: str,
    anthropic_client,
    model: str = "claude-haiku-4-5-20251001",
    min_kw: int = 3,
    max_kw: int = 7,
) -> list[str]:
    """
    Extract keywords using Haiku API (async).

    Args:
        short_text:       The SHORT memory summary.
        anthropic_client: An initialised anthropic.AsyncAnthropic client.
        model:            Model string (default: Haiku).
        min_kw:           Minimum keywords.
        max_kw:           Maximum keywords.

    Returns:
        List of 3-7 keyword strings.
        Falls back to local extraction if API fails.
    """
    from compression.templates import build_keyword_prompt

    if not short_text or not short_text.strip():
        return []

    prompt = build_keyword_prompt(short_text)

    try:
        response = await anthropic_client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text if response.content else ""
        keywords = _parse_keyword_response(raw)

        if keywords and len(keywords) >= min_kw:
            logger.debug("keyword_generator: API returned %d keywords", len(keywords))
            return keywords[:max_kw]

        # API returned bad output — fall back to local
        logger.warning(
            "keyword_generator: API output unparseable, falling back to local extraction"
        )
        return _extract_local(short_text, min_kw, max_kw)

    except Exception as exc:
        logger.error("keyword_generator: API call failed: %s — using local fallback", exc)
        return _extract_local(short_text, min_kw, max_kw)


def generate_keywords_local(
    short_text: str,
    min_kw: int = 3,
    max_kw: int = 7,
) -> list[str]:
    """
    Extract keywords using fast local method (sync, no API).

    Use this:
    - During compression queue when API is rate-limited
    - As fallback when Haiku is unavailable
    - In tests to avoid API calls

    Args:
        short_text: The SHORT memory summary.
        min_kw:     Minimum keywords to return.
        max_kw:     Maximum keywords to return.

    Returns:
        List of 3-7 keyword strings.
    """
    return _extract_local(short_text, min_kw, max_kw)


def validate_keywords(keywords: list[str]) -> bool:
    """
    Validate that a keyword list meets the 3-7 requirement.

    Args:
        keywords: List to validate.

    Returns:
        True if valid (3-7 non-empty strings), False otherwise.
    """
    if not isinstance(keywords, list):
        return False
    valid = [k for k in keywords if isinstance(k, str) and k.strip()]
    return 3 <= len(valid) <= 7