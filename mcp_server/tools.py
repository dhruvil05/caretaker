from retrieval.retrieval_engine import retrieve_context
from capture.capture_engine import run_capture
from mcp_server.injector import build_whisper

# ── Phase 2 imports ──────────────────────────────────────────────────────────
from storage.vector_db import get_vector_db
from storage.local_db import update_memory
from datetime import datetime, timezone
import json
# ─────────────────────────────────────────────────────────────────────────────


def get_context(message: str, agent_id: str = "claude") -> str:
    """
    MCP tool: get_context
    Upgraded for Phase 2 — uses semantic search + smart budget.
    Falls back to Phase 1 keyword search if Phase 2 not ready.
    """
    try:
        context = retrieve_context(message, agent_id)
        whisper = build_whisper(context)

        final = f"""IMPORTANT - YOU HAVE MEMORY. READ THIS CAREFULLY:

{whisper}

INSTRUCTION: You already know everything above. Use this memory naturally in your response. Do not say you cannot remember. Do not ask user to remind you. You ALREADY know this information."""

        return final

    except Exception as e:
        print(f"[TOOLS] get_context error: {e}")
        return "[CARETAKER] Memory unavailable."


def save_message(message: str, agent_id: str = "claude") -> str:
    """
    MCP tool: save_message
    Upgraded for Phase 2:
      - run_capture now uses long_message_handler, importance_scorer,
        temperature_engine, conflict_checker internally
      - compression_queue is enqueued inside run_capture
      - After capture, registers on_complete callback so when Haiku
        finishes SHORT+KEYWORDS, SQLite + ChromaDB are both updated
    """
    try:
        memories = run_capture(message, agent_id)

        if not memories:
            return "[CARETAKER] Nothing to save."

        # Register compression callback for each memory
        # (safe no-op if queue not initialised)
        _register_compression_callbacks(memories)

        # Return confirmation for first memory (standard single-message case)
        memory = memories[0]
        suffix = f" (+{len(memories)-1} more)" if len(memories) > 1 else ""
        return f"[CARETAKER] Memory saved. id={memory['id']} type={memory['type']}{suffix}"

    except Exception as e:
        print(f"[TOOLS] save_message error: {e}")
        return "[CARETAKER] Failed to save memory."


# ── Phase 2: Compression completion callback ─────────────────────────────────

def _register_compression_callbacks(memories: list[dict]) -> None:
    """
    Register an async on_complete callback on the compression queue
    for each newly captured memory.

    When Haiku finishes:
      1. Updates SQLite: short + keywords fields
      2. Upserts ChromaDB with SHORT embedding

    Safe no-op if queue not initialised (e.g. tests, Phase 1 mode).
    """
    try:
        from scheduler.compression_queue import get_queue
        queue = get_queue()

        # Only register if queue doesn't already have a callback
        if queue._on_complete is None:
            queue._on_complete = _on_compression_complete

    except RuntimeError:
        pass   # Queue not initialised — skip
    except Exception as e:
        print(f"[TOOLS] Callback registration error (non-fatal): {e}")


async def _on_compression_complete(
    memory_id: str,
    short:     str,
    keywords:  list[str],
    success:   bool,
) -> None:
    """
    Called by compression queue after Haiku generates SHORT + KEYWORDS.

    Updates:
      1. SQLite  — short + keywords fields, status stays ACTIVE
      2. ChromaDB — upsert with SHORT text + metadata for semantic search
    """
    if not success or not short:
        print(f"[TOOLS] Compression failed for {memory_id} — skipping DB update")
        return

    now = datetime.now(timezone.utc).isoformat()

    # ── 1. Update SQLite ──────────────────────────────────────────────────────
    try:
        update_memory(memory_id, {
            "short":      short,
            "keywords":   json.dumps(keywords),
            "updated_at": now,
        })
        print(f"[TOOLS] SQLite updated: {memory_id} — short + keywords written")
    except Exception as e:
        print(f"[TOOLS] SQLite update error for {memory_id}: {e}")

    # ── 2. Upsert ChromaDB ────────────────────────────────────────────────────
    try:
        from storage.local_db import get_memory_by_id
        mem = get_memory_by_id(memory_id)

        if mem:
            vdb = get_vector_db()
            vdb.upsert(
                memory_id=memory_id,
                short_text=short,
                metadata={
                    "type":        mem.get("type", ""),
                    "temperature": mem.get("temperature", "HOT"),
                    "importance":  float(mem.get("importance", 0.5)),
                    "keywords":    keywords,
                    "status":      mem.get("status", "ACTIVE"),
                    "created_at":  mem.get("created_at", now),
                },
            )
            print(f"[TOOLS] ChromaDB upserted: {memory_id}")

    except RuntimeError:
        pass   # VectorDB not initialised — skip
    except Exception as e:
        print(f"[TOOLS] ChromaDB upsert error for {memory_id}: {e}")