from retrieval.retrieval_engine import retrieve_context
from capture.capture_engine import run_capture
from mcp_server.injector import build_whisper


def get_context(message: str, agent_id: str = "claude") -> str:
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
    try:
        memory = run_capture(message, agent_id)
        return f"[CARETAKER] Memory saved. id={memory['id']} type={memory['type']}"
    except Exception as e:
        print(f"[TOOLS] save_message error: {e}")
        return "[CARETAKER] Failed to save memory."