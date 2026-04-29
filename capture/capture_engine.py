import uuid
import json
import asyncio
from datetime import datetime, timezone

from capture.entity_extractor import extract_entities
from capture.type_classifier import classify_type
from storage.local_db import save_memory, get_memories_by_type, update_memory

# ── Phase 2 imports ──────────────────────────────────────────────────────────
from capture.long_message_handler import handle_long_message, _count_tokens
from memory.importance_scorer import score_importance
from memory.temperature_engine import assign_initial_temperature
from memory.conflict_checker import check_conflict, build_outdated_update
from storage.vector_db import get_vector_db
from scheduler.compression_queue import get_queue
# ─────────────────────────────────────────────────────────────────────────────


def run_capture(message: str, agent_id: str = "claude") -> list[dict]:
    """
    Main capture pipeline — upgraded for Phase 2.

    Changes vs Phase 1:
      - long_message_handler replaces simple _handle_long_message
      - importance_scorer replaces hardcoded importance from classifier
      - temperature_engine replaces get_temperature()
      - conflict_checker marks OUTDATED when REPLACEABLE conflict found
      - compression_queue enqueues background SHORT+KEYWORDS generation
      - vector_db upserted immediately with placeholder, updated after compression

    Returns list of saved memory dicts (1 normally, 2+ if message was SPLIT).
    """
    return asyncio.run(_run_capture_async(message, agent_id))


async def _run_capture_async(message: str, agent_id: str = "claude") -> list[dict]:
    saved_memories = []

    # ── Step 1: Long message handler (Phase 2) ────────────────────────────────
    handler_result = await handle_long_message(message)

    segments   = handler_result.segments    # 1 segment normally, 2+ if SPLIT
    strategy   = handler_result.strategy   # passthrough | split | compress

    if strategy != "passthrough":
        print(f"[CAPTURE] Long message strategy={strategy} → {len(segments)} segment(s)")

    # ── Process each segment as its own memory unit ───────────────────────────
    for segment in segments:

        # ── Step 2: Entity extraction + type classification ───────────────────
        entities   = extract_entities(segment)
        classified = classify_type(segment)

        # ── Step 3: Importance scoring (Phase 2) ─────────────────────────────
        importance = score_importance(
            memory_type=classified["type"],
            fact_type=classified["fact_type"],
            full_text=segment,
        )

        # ── Step 4: Temperature assignment (Phase 2) ──────────────────────────
        temp_result = assign_initial_temperature(importance)
        temperature = temp_result.temperature

        now = datetime.now(timezone.utc).isoformat()
        memory_id = str(uuid.uuid4())

        memory = {
            "id":              memory_id,
            "source_agent":    agent_id,
            "keywords":        json.dumps(entities["keywords"]),
            "short":           None,        # filled async by compression queue
            "full":            segment,
            "type":            classified["type"],
            "subtype":         classified.get("subtype"),
            "fact_type":       classified["fact_type"],
            "status":          "ACTIVE",
            "superseded_by":   None,
            "importance":      importance,
            "decay_score":     1.0,
            "temperature":     temperature,
            "retrieval_count": 0,
            "created_at":      now,
            "updated_at":      now,
            "last_used":       now,
        }

        # ── Step 5: Conflict check (Phase 2) ──────────────────────────────────
        try:
            existing = get_memories_by_type(classified["type"])
            existing_active = [
                m for m in existing
                if m.get("status") == "ACTIVE" and m.get("id") != memory_id
            ]

            if existing_active:
                # Add keywords list for overlap check
                memory_for_check = dict(memory)
                memory_for_check["keywords"] = entities["keywords"]  # list not JSON

                conflict = check_conflict(memory_for_check, existing_active)

                if conflict.conflict_found and conflict.action == "replaced":
                    outdated_update = build_outdated_update(
                        existing_id=conflict.existing_id,
                        superseded_by=memory_id,
                    )
                    update_memory(conflict.existing_id, {
                        "status":        "OUTDATED",
                        "superseded_by": memory_id,
                        "updated_at":    now,
                    })
                    print(f"[CAPTURE] Conflict resolved: {conflict.existing_id} → OUTDATED")

        except Exception as e:
            print(f"[CAPTURE] Conflict check error (non-fatal): {e}")

        # ── Step 6: Save to SQLite ─────────────────────────────────────────────
        success = save_memory(memory)

        if success:
            print(f"[CAPTURE] Saved {memory_id} | type={memory['type']} | temp={temperature} | importance={importance:.2f}")
        else:
            print(f"[CAPTURE] Failed to save memory {memory_id}")
            continue

        # ── Step 7: Enqueue compression (Phase 2 — non-blocking) ──────────────
        try:
            queue = get_queue()
            await queue.enqueue(
                memory_id=memory_id,
                memory_type=classified["type"],
                full_text=segment,
            )
        except RuntimeError:
            # Queue not initialised (e.g. during tests) — skip silently
            pass
        except Exception as e:
            print(f"[CAPTURE] Compression enqueue error (non-fatal): {e}")

        saved_memories.append(memory)

    return saved_memories