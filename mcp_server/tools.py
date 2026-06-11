from retrieval.retrieval_engine import retrieve_context
from capture.capture_engine import run_capture
from mcp_server.injector import build_whisper
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def get_context(
    message: str,
    agent_id: str = "claude",
    # Phase 2: optional — passed from server.py if available
    semantic_searcher=None,
    memory_selector=None,
) -> str:
    try:
        print(f"[TOOLS] get_context for message: {message} (agent={agent_id})")
        # Phase 2: pass semantic_searcher + memory_selector to retrieval engine
        context = retrieve_context(
            message,
            agent_id,
            semantic_searcher=semantic_searcher,
            memory_selector=memory_selector,
        )
        whisper = build_whisper(context)

        # Phase 3: format whisper per agent type via agent_adapter
        try:
            from mcp_server.agent_adapter import adapt, normalise_agent_id
            canonical = normalise_agent_id(agent_id)
            final = adapt(whisper, agent_id)
            print(f"[TOOLS] Whisper adapted for agent={canonical}")
        except Exception as adapt_err:
            # Fallback to Phase 1/2 format if adapter fails
            print(f"[TOOLS] agent_adapter failed ({adapt_err}), using default format")
            final = f"""IMPORTANT - YOU HAVE MEMORY. READ THIS CAREFULLY:

{whisper}

INSTRUCTION: You already know everything above. Use this memory naturally in your response. Do not say you cannot remember. Do not ask user to remind you. You ALREADY know this information."""

        print(f"[TOOLS] Whisper:\n{final}")
        return final
    except Exception as e:
        print(f"[TOOLS] get_context error: {e}")
        import traceback
        traceback.print_exc()
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