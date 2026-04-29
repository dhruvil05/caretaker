"""
storage/vector_db.py
---------------------
ChromaDB vector store for Caretaker Phase 2 semantic search.

Responsibilities:
  - Initialise persistent ChromaDB client (survives server restart → P2-T15)
  - Upsert memory embeddings after compression completes
  - Semantic similarity search filtered by temperature tier
  - Delete / mark entries when memory is OUTDATED or ARCHIVED

Collection schema (per document in ChromaDB):
  id        : memory UUID (same as SQLite)
  embedding : auto-generated from SHORT text via ChromaDB default embedding fn
  document  : SHORT summary text
  metadata  : {
      type        : str   (PROJECT, PREFERENCE, ...)
      temperature : str   (PRIORITY_HOT, HOT, WARM, COLD)
      importance  : float
      keywords    : str   (comma-separated, for metadata filtering)
      status      : str   (ACTIVE, OUTDATED, ARCHIVED)
      created_at  : str   (ISO datetime)
  }

Phase 2 — Storage Upgrade
"""

import logging
import os
from typing import Optional

import hashlib
import math
import chromadb
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight embedding function (no external model download required)
# Uses a fixed 256-dim bag-of-words style embedding with character n-grams.
# Sufficient for semantic similarity on short memory summaries (≤60 tokens).
# Production systems can swap this for sentence-transformers or OpenAI embeddings.
# ---------------------------------------------------------------------------

EMBED_DIM = 256

class _SimpleEmbeddingFunction(EmbeddingFunction):
    """
    Fast, dependency-free embedding function for short memory summaries.

    Strategy:
      - Tokenise text into 2-gram and 3-gram character shingles
      - Hash each shingle into a 256-dim float vector bucket
      - L2-normalise the result

    This gives meaningful cosine similarity for short texts without requiring
    any external model download (unlike ChromaDB's default ONNX model).
    """

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(doc) for doc in input]

    @staticmethod
    def _embed(text: str) -> list[float]:
        if not text:
            return [0.0] * EMBED_DIM

        tokens = text.lower().split()
        vec = [0.0] * EMBED_DIM

        for token in tokens:
            # character 2-grams and 3-grams
            for n in (2, 3):
                for i in range(len(token) - n + 1):
                    shingle = token[i:i+n]
                    h = int(hashlib.md5(shingle.encode()).hexdigest(), 16)
                    idx = h % EMBED_DIM
                    vec[idx] += 1.0

        # L2 normalise
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


_EMBEDDING_FN = _SimpleEmbeddingFunction()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH:        str = "./caretaker_chroma"
DEFAULT_COLLECTION:     str = "memories"
FETCHABLE_TEMPERATURES: set = {"PRIORITY_HOT", "HOT", "WARM"}


# ---------------------------------------------------------------------------
# VectorDB class
# ---------------------------------------------------------------------------

class VectorDB:
    """
    Persistent ChromaDB wrapper for Caretaker memory embeddings.

    Usage:
        vdb = VectorDB(path="./caretaker_chroma")
        vdb.init()
        vdb.upsert(memory_id, short_text, metadata)
        results = vdb.search("what editor does the user prefer?", n_results=5)
    """

    def __init__(
        self,
        path:            str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        """
        Args:
            path:            Directory path for persistent ChromaDB storage.
            collection_name: Name of the ChromaDB collection.
        """
        self._path            = path
        self._collection_name = collection_name
        self._client:     Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection]       = None

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def init(self) -> None:
        """
        Initialise persistent ChromaDB client and get/create collection.

        Creates the storage directory if it doesn't exist.
        Safe to call multiple times — idempotent.
        P2-T15: persistent storage survives server restart.
        """
        if self._collection is not None:
            logger.debug("vector_db: already initialised")
            return

        os.makedirs(self._path, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self._path,
            settings=Settings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=_EMBEDDING_FN,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
        )

        logger.info(
            "vector_db: initialised collection '%s' at '%s' (count=%d)",
            self._collection_name,
            self._path,
            self._collection.count(),
        )

    def _ensure_init(self) -> None:
        """Raise if init() was not called."""
        if self._collection is None:
            raise RuntimeError(
                "VectorDB not initialised. Call vdb.init() before using."
            )

    # -----------------------------------------------------------------------
    # Upsert — add or update a memory embedding
    # -----------------------------------------------------------------------

    def upsert(
        self,
        memory_id:   str,
        short_text:  str,
        metadata:    dict,
    ) -> None:
        """
        Add or update a memory in ChromaDB.

        Called after compression completes — SHORT text is used as the document
        for embedding generation.

        Args:
            memory_id:  UUID of the memory (must match SQLite id).
            short_text: Compressed SHORT summary (≤60 tokens).
            metadata:   Dict with keys: type, temperature, importance,
                        keywords (list or comma-str), status, created_at.
        """
        self._ensure_init()

        if not short_text or not short_text.strip():
            logger.warning("vector_db: upsert skipped — empty short_text for %s", memory_id)
            return

        # Normalise keywords to comma-string (ChromaDB metadata must be scalar)
        clean_meta = _normalise_metadata(metadata)

        self._collection.upsert(
            ids=[memory_id],
            documents=[short_text.strip()],
            metadatas=[clean_meta],
        )

        logger.debug(
            "vector_db: upserted %s (temperature=%s importance=%.2f)",
            memory_id,
            clean_meta.get("temperature", "?"),
            clean_meta.get("importance", 0.0),
        )

    # -----------------------------------------------------------------------
    # Search — semantic similarity with temperature filter
    # -----------------------------------------------------------------------

    def search(
        self,
        query_text:   str,
        n_results:    int = 10,
        temperatures: Optional[list[str]] = None,
        memory_type:  Optional[str] = None,
        min_relevance: float = 0.0,
    ) -> list[dict]:
        """
        Semantic similarity search with optional temperature + type filter.

        P2-T08: Most semantically relevant memory returned first.
        P2-T09: COLD memories NOT included unless explicitly requested.

        Args:
            query_text:    Natural language query string.
            n_results:     Max results to return.
            temperatures:  List of temperature tiers to include.
                           Defaults to PRIORITY_HOT + HOT + WARM (excludes COLD).
            memory_type:   Optional TYPE filter (PROJECT, PREFERENCE, etc.).
            min_relevance: Minimum cosine similarity score (0.0–1.0).

        Returns:
            List of result dicts sorted by relevance (highest first):
            [{id, short, score, temperature, importance, type, keywords, status}]
        """
        self._ensure_init()

        if not query_text or not query_text.strip():
            return []

        # Default: exclude COLD (P2-T09)
        if temperatures is None:
            temperatures = list(FETCHABLE_TEMPERATURES)

        # Build ChromaDB where filter
        where = _build_where_filter(temperatures, memory_type)

        try:
            query_kwargs = dict(
                query_texts=[query_text.strip()],
                n_results=min(n_results, max(1, self._collection.count())),
                include=["documents", "metadatas", "distances"],
            )
            if where:
                query_kwargs["where"] = where

            raw = self._collection.query(**query_kwargs)

        except Exception as exc:
            logger.error("vector_db: search failed: %s", exc)
            return []

        # Parse results
        results = []
        ids       = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for mem_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            # ChromaDB cosine distance → similarity: score = 1 - distance
            score = round(1.0 - dist, 4)

            if score < min_relevance:
                continue

            keywords_raw = meta.get("keywords", "")
            keywords = (
                [k.strip() for k in keywords_raw.split(",") if k.strip()]
                if isinstance(keywords_raw, str)
                else keywords_raw
            )

            results.append({
                "id":          mem_id,
                "short":       doc,
                "score":       score,
                "temperature": meta.get("temperature", "HOT"),
                "importance":  float(meta.get("importance", 0.5)),
                "type":        meta.get("type", ""),
                "keywords":    keywords,
                "status":      meta.get("status", "ACTIVE"),
            })

        # Sort by score descending (P2-T08)
        results.sort(key=lambda x: x["score"], reverse=True)

        logger.debug(
            "vector_db: search '%s' → %d results (temp_filter=%s)",
            query_text[:50], len(results), temperatures,
        )

        return results

    # -----------------------------------------------------------------------
    # Update metadata — temperature change, status change
    # -----------------------------------------------------------------------

    def update_metadata(self, memory_id: str, updates: dict) -> None:
        """
        Update metadata fields for an existing memory.
        Used when temperature changes (decay) or status changes (OUTDATED).

        Args:
            memory_id: UUID of the memory.
            updates:   Dict of metadata fields to update.
        """
        self._ensure_init()

        try:
            existing = self._collection.get(ids=[memory_id], include=["metadatas", "documents"])
        except Exception as exc:
            logger.warning("vector_db: update_metadata get failed for %s: %s", memory_id, exc)
            return

        if not existing["ids"]:
            logger.warning("vector_db: update_metadata — %s not found", memory_id)
            return

        current_meta = existing["metadatas"][0].copy()
        current_meta.update(_normalise_metadata(updates))

        self._collection.update(
            ids=[memory_id],
            metadatas=[current_meta],
        )

        logger.debug("vector_db: updated metadata for %s: %s", memory_id, list(updates.keys()))

    # -----------------------------------------------------------------------
    # Delete
    # -----------------------------------------------------------------------

    def delete(self, memory_id: str) -> None:
        """
        Remove a memory from ChromaDB entirely.
        Used when memory is permanently deleted (not just OUTDATED).

        Args:
            memory_id: UUID of the memory to delete.
        """
        self._ensure_init()

        try:
            self._collection.delete(ids=[memory_id])
            logger.info("vector_db: deleted %s", memory_id)
        except Exception as exc:
            logger.warning("vector_db: delete failed for %s: %s", memory_id, exc)

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Total number of memories in ChromaDB."""
        self._ensure_init()
        return self._collection.count()

    def stats(self) -> dict:
        """Return collection stats."""
        self._ensure_init()
        return {
            "collection": self._collection_name,
            "path":       self._path,
            "count":      self._collection.count(),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_vdb: Optional[VectorDB] = None


def get_vector_db() -> VectorDB:
    """Return the global VectorDB instance."""
    global _global_vdb
    if _global_vdb is None:
        raise RuntimeError("VectorDB not initialised. Call init_vector_db() first.")
    return _global_vdb


def init_vector_db(
    path:            str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
) -> VectorDB:
    """
    Initialise the global VectorDB singleton.
    Call once at server startup before any upserts or searches.

    Args:
        path:            Persistent storage directory.
        collection_name: ChromaDB collection name.

    Returns:
        Initialised VectorDB instance.
    """
    global _global_vdb
    _global_vdb = VectorDB(path=path, collection_name=collection_name)
    _global_vdb.init()
    return _global_vdb


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_metadata(meta: dict) -> dict:
    """
    Normalise metadata dict so all values are ChromaDB-safe scalars
    (str, int, float, bool). Lists are converted to comma-separated strings.
    """
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, list):
            clean[k] = ",".join(str(i) for i in v)
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def _build_where_filter(
    temperatures: list[str],
    memory_type:  Optional[str],
) -> Optional[dict]:
    """
    Build a ChromaDB `where` filter dict from temperature list and type.

    Returns None if no filter needed (avoid passing empty where to ChromaDB).
    """
    conditions = []

    # Temperature filter
    if temperatures:
        if len(temperatures) == 1:
            conditions.append({"temperature": {"$eq": temperatures[0]}})
        else:
            conditions.append({"temperature": {"$in": temperatures}})

    # Always exclude OUTDATED and ARCHIVED from search
    conditions.append({"status": {"$eq": "ACTIVE"}})

    # Optional type filter
    if memory_type:
        conditions.append({"type": {"$eq": memory_type.upper()}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}