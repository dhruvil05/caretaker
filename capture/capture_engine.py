import uuid
import json
from datetime import datetime, timezone

from capture.entity_extractor import extract_entities
from capture.type_classifier import classify_type
from storage.local_db import save_memory

# ── Phase 2 imports ────────────────────────────────────────────────────────────
from memory.importance_scorer import score_importance
from memory.temperature_engine import assign_temperature
from memory.conflict_checker import full_conflict_pipeline
from capture.long_message_handler import is_long_message, process_long_message


def get_temperature(importance: float) -> str:
    """Phase 1 temperature assignment — kept for fallback."""
    if importance >= 0.7:
        return "PRIORITY_HOT"
    elif importance >= 0.5:
        return "HOT"
    elif importance >= 0.2:
        return "WARM"
    else:
        return "COLD"


def count_tokens_approx(text: str) -> int:
    return len(text.split())


def run_capture(
    message: str,
    agent_id: str = "claude",
    compressor=None,       # Phase 2: optional Compressor instance
    compression_queue=None, # Phase 2: optional CompressionQueue instance
    local_db=None,          # Phase 2: optional db reference for conflict check
) -> dict:

    # ── Phase 1: classify + extract (unchanged) ───────────────────────────
    entities   = extract_entities(message)
    classified = classify_type(message)
    # ADD THIS:
    if classified["type"] == "SKIP":
        print(f"[CAPTURE] Skipped — question or noise.")
        return {"id": "skipped", "type": "SKIP", "temperature": "COLD"}
    now        = datetime.now(timezone.utc).isoformat()

    # ── Phase 2: importance score (replaces Phase 1 classified["importance"] directly) ──
    importance = score_importance(message, classified["type"])

    # ── Phase 2: temperature via temperature_engine (replaces get_temperature()) ─
    temperature = assign_temperature(importance, classified["type"])

    # ── Phase 2: long message handler (replaces Phase 1 _handle_long_message) ──
    if is_long_message(message) and compressor:
        processed_chunks = process_long_message(
            text=message,
            memory_type=classified["type"],
            compressor=compressor,
        )
    else:
        # Phase 1 path: no compressor available or short message
        processed_chunks = [{
            "full_text":   message.strip(),
            "short":       None,
            "keywords":    entities["keywords"],
            "memory_type": classified["type"],
            "chunk_index": 0,
        }]

    saved_memories = []

    for chunk in processed_chunks:
        memory = {
            "id":              str(uuid.uuid4()),
            "source_agent":    agent_id,
            "keywords":        json.dumps(chunk.get("keywords") or entities["keywords"]),
            "short":           chunk.get("short"),
            "full":            chunk["full_text"],
            "type":            classified["type"],
            "subtype":         classified["subtype"],
            "fact_type":       classified["fact_type"],
            # Phase 2: use scored importance instead of classifier importance
            "importance":      importance,
            "decay_score":     1.0,
            # Phase 2: use temperature_engine result
            "temperature":     temperature,
            "status":          "ACTIVE",
            "superseded_by":   None,
            "retrieval_count": 0,
            "created_at":      now,
            "updated_at":      now,
            "last_used":       now,
        }

        # ── Phase 2: conflict resolution before save ───────────────────────
        if local_db:
            full_conflict_pipeline(
                new_memory={
                    "memory_id":   memory["id"],
                    "memory_type": memory["type"],
                    "keywords":    chunk.get("keywords") or entities["keywords"],
                    "short":       chunk.get("short") or "",
                    "full_text":   chunk["full_text"],
                },
                local_db=local_db,
            )

        success = save_memory(memory)

        if success:
            print(f"[CAPTURE] Saved memory {memory['id']} | type={memory['type']} | temp={memory['temperature']}")

            # ── Phase 2: enqueue compression if short not yet generated ───
            if compression_queue and not memory.get("short"):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(
                            compression_queue.enqueue(
                                memory_id=memory["id"],
                                full_text=chunk["full_text"],
                                memory_type=memory["type"],
                            )
                        )
                    else:
                        loop.run_until_complete(
                            compression_queue.enqueue(
                                memory_id=memory["id"],
                                full_text=chunk["full_text"],
                                memory_type=memory["type"],
                            )
                        )
                except Exception as e:
                    print(f"[CAPTURE] Compression queue enqueue failed: {e}")
        else:
            print(f"[CAPTURE] Failed to save memory")

        saved_memories.append(memory)

    # Return first memory for backward compatibility with Phase 1 callers
    return saved_memories[0] if saved_memories else {}


# ── Phase 1 helper — kept for reference, replaced by long_message_handler ─────
def _handle_long_message(message: str) -> str:
    words      = message.split()
    chunks     = []
    chunk_size = 300

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    if len(chunks) == 1:
        return chunks[0]

    compressed = f"[LONG MESSAGE - {len(chunks)} parts] " + " | ".join(
        chunk[:100] + "..." for chunk in chunks
    )
    return compressed