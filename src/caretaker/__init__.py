"""
caretaker
=========
Universal agent memory layer — local-first, agent-agnostic, cloud-backed.

This file flattens the package so common functions/classes can be imported
directly, e.g.:

    from caretaker import save_memory, retrieve_context, Compressor

Note: `extract_keywords` (retrieval) and `extract_keywords_nlp` (compression)
share similar names but come from different modules — both are exposed here
under distinct names to avoid clashing.
"""

from .capture import (
    get_temperature,
    count_tokens_approx,
    run_capture,
    extract_entities,
    estimate_tokens,
    is_long_message,
    handle_long_message,
    process_long_message,
    is_question_or_noise,
    classify_type,
)

from .compression import (
    Compressor,
    extract_keywords_nlp,
    compress_local,
    get_template,
)

from .memory import (
    is_replaceable,
    check_conflict,
    resolve_conflict,
    full_conflict_pipeline,
    score_importance,
    score_batch,
    assign_temperature,
    apply_decay,
    reheat,
    batch_decay,
    get_search_tiers,
)

from .retrieval import (
    calculate_budget,
    extract_keywords,
    select_memory_forms,
    format_for_context,
    retrieve_context,
    SemanticSearcher,
    detect_topic,
)

from .scheduler import (
    CompressionJob,
    CompressionQueue,
    MaintenanceRunner,
    NightlyMaintenance,
    CaretakerScheduler,
)

from .storage import (
    CloudSync,
    Encryptor,
    encrypt_memory_dict,
    decrypt_memory_dict,
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
    VectorDB,
)

from .mcp_server import (
    caretaker_get_context,
    caretaker_save_message,
    get_context,
    save_message,
    adapt,
    normalise_agent_id,
    get_supported_agents,
    is_known_agent,
    get_agent_info,
    build_whisper,
)

__version__ = "0.3.2"

__all__ = [
    # capture
    "get_temperature", "count_tokens_approx", "run_capture", "extract_entities",
    "estimate_tokens", "is_long_message", "handle_long_message",
    "process_long_message", "is_question_or_noise", "classify_type",
    # compression
    "Compressor", "extract_keywords_nlp", "compress_local", "get_template",
    # memory
    "is_replaceable", "check_conflict", "resolve_conflict",
    "full_conflict_pipeline", "score_importance", "score_batch",
    "assign_temperature", "apply_decay", "reheat", "batch_decay",
    "get_search_tiers",
    # retrieval
    "calculate_budget", "extract_keywords", "select_memory_forms",
    "format_for_context", "retrieve_context", "SemanticSearcher",
    "detect_topic",
    # scheduler
    "CompressionJob", "CompressionQueue", "MaintenanceRunner",
    "NightlyMaintenance", "CaretakerScheduler",
    # storage
    "CloudSync", "Encryptor", "encrypt_memory_dict", "decrypt_memory_dict",
    "get_connection", "run_migrations", "save_memory", "get_memories_by_type",
    "get_recent_memories", "get_memory_by_id", "update_memory_status",
    "increment_retrieval_count", "get_all_active_memories",
    "update_compression", "update_status", "update_temperature",
    "touch_last_accessed", "get_active_by_type", "get_by_ids",
    "get_all_for_decay", "get_all_memories", "update_memory_fields",
    "archive_memory", "restore_memory", "get_stats",
    "search_memories_by_keyword", "upsert_memory", "get_memories_by_agent",
    "get_duplicate_candidates", "VectorDB",
    # mcp_server
    "caretaker_get_context", "caretaker_save_message", "get_context",
    "save_message", "adapt", "normalise_agent_id", "get_supported_agents",
    "is_known_agent", "get_agent_info", "build_whisper",
]
