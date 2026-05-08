from retrieval.retrieval_engine import retrieve_context
from capture.capture_engine import run_capture
from mcp_server.injector import build_whisper


def get_context(
    message: str,
    agent_id: str = "claude",
    # Phase 2: optional — passed from server.py if available
    semantic_searcher=None,
    memory_selector=None,
) -> str:
    try:
        print(f"[TOOLS] get_context for message: {message}")
        # Phase 2: pass semantic_searcher + memory_selector to retrieval engine
        context = retrieve_context(
            message,
            agent_id,
            semantic_searcher=semantic_searcher,
            memory_selector=memory_selector,
        )
        whisper = build_whisper(context)
        print(f"[TOOLS] Whisper:\n{whisper}")  # ← ADD THIS LINE for debugging
        # Phase 1: final whisper format — unchanged
        final = f"""IMPORTANT - YOU HAVE MEMORY. READ THIS CAREFULLY:

{whisper}

INSTRUCTION: You already know everything above. Use this memory naturally in your response. Do not say you cannot remember. Do not ask user to remind you. You ALREADY know this information."""

        return final
    except Exception as e:
        print(f"[TOOLS] get_context error: {e}")
        import traceback
        traceback.print_exc()  # ← ADD THIS LINE temporarily
        return "[CARETAKER] Memory unavailable."


def save_message(
    message: str,
    agent_id: str = "claude",
    # Phase 2: optional — passed from server.py if available
    compressor=None,
    compression_queue=None,
    local_db=None,
) -> str:
    try:
        # Phase 2: pass compressor + compression_queue + local_db to capture engine
        memory = run_capture(
            message,
            agent_id,
            compressor=compressor,
            compression_queue=compression_queue,
            local_db=local_db,
        )
        # Phase 1: return format — unchanged
        return f"[CARETAKER] Memory saved. id={memory['id']} type={memory['type']}"
    except Exception as e:
        print(f"[TOOLS] save_message error: {e}")
        return "[CARETAKER] Failed to save memory."