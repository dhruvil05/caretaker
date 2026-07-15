"""
caretaker.storage
Public API for the local SQLite store, the vector DB, cloud sync, and encryption.
"""

from .cloud_sync import CloudSync
from .encrypt import Encryptor, encrypt_memory_dict, decrypt_memory_dict
from .local_db import (
    get_connection,
    run_migrations,
    save_memory,
    get_memories_by_type,
    get_recent_memories,
    get_memory_by_id,
    update_memory_status,
    increment_retrieval_count,
    get_all_active_memories,
    update_compression,
    update_status,
    update_temperature,
    touch_last_accessed,
    get_active_by_type,
    get_by_ids,
    get_all_for_decay,
    get_all_memories,
    update_memory_fields,
    archive_memory,
    restore_memory,
    get_stats,
    search_memories_by_keyword,
    upsert_memory,
    get_memories_by_agent,
    get_duplicate_candidates,
)
from .vector_db import VectorDB

__all__ = [
    "CloudSync",
    "Encryptor",
    "encrypt_memory_dict",
    "decrypt_memory_dict",
    "get_connection",
    "run_migrations",
    "save_memory",
    "get_memories_by_type",
    "get_recent_memories",
    "get_memory_by_id",
    "update_memory_status",
    "increment_retrieval_count",
    "get_all_active_memories",
    "update_compression",
    "update_status",
    "update_temperature",
    "touch_last_accessed",
    "get_active_by_type",
    "get_by_ids",
    "get_all_for_decay",
    "get_all_memories",
    "update_memory_fields",
    "archive_memory",
    "restore_memory",
    "get_stats",
    "search_memories_by_keyword",
    "upsert_memory",
    "get_memories_by_agent",
    "get_duplicate_candidates",
    "VectorDB",
]
