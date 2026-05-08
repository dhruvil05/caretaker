"""
compression/compressor.py
Main compression router.
PATH 1 — Paid: Uses Haiku API for best quality SHORT + KEYWORDS.
PATH 2 — Free: Uses local spaCy + TextRank (no API key needed).
Decision is made from config.json → compression_model field.
"""

import json
import logging
from typing import Tuple, List

from compression.templates import get_template
from compression.keyword_generator import extract_keywords
from compression.local_compressor import compress_local

logger = logging.getLogger(__name__)


class Compressor:
    """
    Routes compression to Haiku API or local engine based on config.
    Usage:
        compressor = Compressor(config)
        short, keywords = compressor.compress(full_text, memory_type)
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = config.get("compression_model", "local")
        self.use_haiku = self._should_use_haiku()

        if self.use_haiku:
            logger.info(f"[Compressor] Mode: HAIKU API ({self.model})")
        else:
            logger.info("[Compressor] Mode: LOCAL (spaCy + TextRank) — no API key")

    def _should_use_haiku(self) -> bool:
        """
        Use Haiku only if:
        - compression_model is NOT 'local'
        - AND anthropic_api_key exists in config
        """
        if self.model == "local":
            return False
        api_key = self.config.get("anthropic_api_key", "").strip()
        return bool(api_key)

    def compress(self, full_text: str, memory_type: str) -> Tuple[str, List[str]]:
        """
        Compress full_text into SHORT + KEYWORDS.
        Returns: (short: str, keywords: List[str])
        Automatically falls back to local if Haiku fails.
        """
        if not full_text or not full_text.strip():
            return ("", [])

        if self.use_haiku:
            try:
                return self._compress_with_haiku(full_text, memory_type)
            except Exception as e:
                logger.warning(f"[Compressor] Haiku failed: {e}. Falling back to local.")
                return compress_local(full_text, memory_type)
        else:
            return compress_local(full_text, memory_type)

    def _compress_with_haiku(self, full_text: str, memory_type: str) -> Tuple[str, List[str]]:
        """
        Call Anthropic Haiku API with type-specific template.
        Returns (short, keywords) parsed from JSON response.
        """
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic SDK not installed. Run: pip install anthropic")

        template = get_template(memory_type)
        api_key = self.config.get("anthropic_api_key", "")

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"{template['user_prefix']}\n\n{full_text}"

        message = client.messages.create(
            model=self.model,
            max_tokens=200,
            system=template["system"],
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()
        return self._parse_response(raw, full_text, memory_type)

    def _parse_response(self, raw: str, full_text: str, memory_type: str) -> Tuple[str, List[str]]:
        """
        Parse JSON response from Haiku.
        Fallback to local if parse fails.
        """
        # Strip markdown code fences if present
        cleaned = raw.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(cleaned)
            short = str(data.get("short", "")).strip()
            keywords = data.get("keywords", [])

            # Validate
            if not short:
                raise ValueError("Empty short field")
            if not isinstance(keywords, list):
                keywords = extract_keywords(short)

            # Ensure keyword count bounds
            if len(keywords) < 3:
                keywords += extract_keywords(full_text, max_keywords=7 - len(keywords))
            keywords = keywords[:7]

            return (short, keywords)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"[Compressor] Failed to parse Haiku response: {e}. Using local fallback.")
            return compress_local(full_text, memory_type)