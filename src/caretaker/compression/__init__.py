"""
caretaker.compression
Public API for compressing memories (Haiku + local fallback).
"""

from .compressor import Compressor
from .keyword_generator import extract_keywords as extract_keywords_nlp
from .local_compressor import compress_local
from .templates import get_template

__all__ = [
    "Compressor",
    "extract_keywords_nlp",
    "compress_local",
    "get_template",
]
