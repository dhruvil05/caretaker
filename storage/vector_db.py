"""
storage/vector_db.py
ChromaDB vector store for semantic memory search.
IMPORTANT: Set env vars BEFORE import to prevent ONNX segfault on Windows.
Uses simple sentence-transformers embedding (no Haiku needed).
Falls back to TF-IDF bag-of-words if sentence-transformers unavailable.
"""

import os
import logging
from typing import List, Dict, Optional

# CRITICAL: Set these BEFORE importing chromadb to prevent ONNX segfault
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

logger = logging.getLogger(__name__)


def _build_embedding_function():
    """
    Build embedding function.
    Priority:
      1. sentence-transformers (best local quality)
      2. Simple TF-IDF style hash embedding (zero deps fallback)
    """
    try:
        from chromadb import EmbeddingFunction, Documents, Embeddings
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        class STEmbeddingFunction(EmbeddingFunction):
            def name(self) -> str:
                return "sentence-transformers-minilm"

            def __call__(self, input: Documents) -> Embeddings:
                return model.encode(list(input), normalize_embeddings=True).tolist()

            @classmethod
            def build_from_config(cls, config: dict) -> "STEmbeddingFunction":
                return cls()

            def get_config(self) -> dict:
                return {}

        logger.info("[VectorDB] Using sentence-transformers embedding (all-MiniLM-L6-v2)")
        return STEmbeddingFunction()

    except ImportError:
        logger.warning("[VectorDB] sentence-transformers not found. Using hash embedding fallback.")
        return _hash_embedding_function()


def _hash_embedding_function():
    """Zero-dependency fallback embedding using character n-gram hashing."""
    try:
        from chromadb import EmbeddingFunction, Documents, Embeddings
        import hashlib
        import math

        DIM = 384  # Match MiniLM dimension for compatibility

        class HashEmbeddingFunction(EmbeddingFunction):
            def name(self) -> str:
                return "hash-ngram-fallback"

            def __call__(self, input: Documents) -> Embeddings:
                results = []
                for text in input:
                    vec = [0.0] * DIM
                    text_lower = str(text).lower()
                    # Character 3-grams
                    for i in range(len(text_lower) - 2):
                        gram = text_lower[i:i+3]
                        h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                        idx = h % DIM
                        vec[idx] += 1.0
                    # L2 normalize
                    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
                    vec = [v / norm for v in vec]
                    results.append(vec)
                return results

            @classmethod
            def build_from_config(cls, config: dict) -> "HashEmbeddingFunction":
                return cls()

            def get_config(self) -> dict:
                return {}

        return HashEmbeddingFunction()

    except Exception as e:
        logger.error(f"[VectorDB] Failed to build hash embedding: {e}")
        raise


class VectorDB:
    """
    ChromaDB wrapper for semantic memory search.
    Collection stores SHORT text + keywords as metadata.
    """

    COLLECTION_NAME = "caretaker_memories"

    def __init__(self, persist_directory: str = "./data/chromadb"):
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
        self._embedding_fn = None

    def initialize(self):
        """Connect to ChromaDB and get/create collection."""
        try:
            import chromadb

            self._embedding_fn = _build_embedding_function()

            # Use PersistentClient directly — no Settings() wrapper (prevents ONNX load)
            self._client = chromadb.PersistentClient(path=self.persist_directory)

            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                f"[VectorDB] Initialized. Collection: {self.COLLECTION_NAME}, "
                f"Count: {self._collection.count()}"
            )

        except Exception as e:
            logger.error(f"[VectorDB] Initialization failed: {e}")
            raise

    def upsert(
        self,
        memory_id: str,
        short: str,
        keywords: List[str],
        temperature: str = "HOT",
        memory_type: str = "LEARNING",
        importance_score: float = 0.5,
    ):
        """
        Add or update a memory in ChromaDB.
        Document = SHORT text (what gets embedded).
        Metadata = temperature, type, keywords, score (used for filtering).
        """
        if not self._collection:
            raise RuntimeError("VectorDB not initialized. Call initialize() first.")

        if not short or not short.strip():
            logger.warning(f"[VectorDB] Skipping upsert for {memory_id} — empty SHORT text")
            return

        self._collection.upsert(
            ids=[memory_id],
            documents=[short],
            metadatas=[{
                "temperature": temperature,
                "memory_type": memory_type,
                "keywords": ", ".join(keywords),
                "importance_score": importance_score,
            }],
        )
        logger.debug(f"[VectorDB] Upserted memory_id={memory_id}")

    def search(
        self,
        query: str,
        n_results: int = 10,
        temperature_filter: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Semantic search over memories.
        temperature_filter: list of tiers to include e.g. ["PRIORITY_HOT","HOT","WARM"]
        Returns list of dicts: {memory_id, short, distance, temperature, memory_type, keywords}
        """
        if not self._collection:
            raise RuntimeError("VectorDB not initialized. Call initialize() first.")

        if self._collection.count() == 0:
            return []

        # Build where filter for temperature
        where = None
        if temperature_filter:
            if len(temperature_filter) == 1:
                where = {"temperature": {"$eq": temperature_filter[0]}}
            else:
                where = {"temperature": {"$in": temperature_filter}}

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, self._collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"[VectorDB] Search failed with filter, retrying without: {e}")
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )

        hits = []
        if results and results.get("ids"):
            for i, memory_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                hits.append({
                    "memory_id": memory_id,
                    "short": results["documents"][0][i] if results.get("documents") else "",
                    "distance": results["distances"][0][i] if results.get("distances") else 1.0,
                    "temperature": meta.get("temperature", "WARM"),
                    "memory_type": meta.get("memory_type", ""),
                    "keywords": meta.get("keywords", ""),
                    "importance_score": float(meta.get("importance_score", 0.5)),
                })

        return hits

    def delete(self, memory_id: str):
        """Remove a memory from ChromaDB."""
        if self._collection:
            self._collection.delete(ids=[memory_id])
            logger.debug(f"[VectorDB] Deleted memory_id={memory_id}")

    def count(self) -> int:
        return self._collection.count() if self._collection else 0