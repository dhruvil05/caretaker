import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from storage.local_db import run_migrations
from capture.capture_engine import run_capture
from mcp_server.tools import get_context, save_message
from mcp_server.injector import build_whisper


@pytest.fixture(autouse=True)
def setup_db():
    run_migrations()


class TestInjector:

    def test_build_whisper_returns_string(self):
        from retrieval.retrieval_engine import retrieve_context
        context = retrieve_context("What project am I building?")
        whisper = build_whisper(context)
        assert isinstance(whisper, str)

    def test_whisper_has_caretaker_header(self):
        from retrieval.retrieval_engine import retrieve_context
        context = retrieve_context("What project am I building?")
        whisper = build_whisper(context)
        assert "[CARETAKER CONTEXT]"     in whisper
        assert "[END CARETAKER CONTEXT]" in whisper

    def test_whisper_has_core_identity_section(self):
        from retrieval.retrieval_engine import retrieve_context
        context = retrieve_context("What project am I building?")
        whisper = build_whisper(context)
        assert "CORE IDENTITY" in whisper

    def test_whisper_has_recent_sessions_section(self):
        from retrieval.retrieval_engine import retrieve_context
        context = retrieve_context("What project am I building?")
        whisper = build_whisper(context)
        assert "RECENT SESSIONS" in whisper

    def test_whisper_has_relevant_memory_section(self):
        from retrieval.retrieval_engine import retrieve_context
        context = retrieve_context("What project am I building?")
        whisper = build_whisper(context)
        assert "RELEVANT MEMORY" in whisper

    def test_whisper_contains_memory_after_save(self):
        from retrieval.retrieval_engine import retrieve_context
        run_capture("I am building Caretaker memory system.", "claude")
        context = retrieve_context("What am I building?")
        whisper = build_whisper(context)
        assert "Caretaker" in whisper

    def test_whisper_shows_project_in_identity(self):
        from retrieval.retrieval_engine import retrieve_context
        run_capture("I am building a FastAPI project called Caretaker.", "claude")
        context = retrieve_context("What is my project?")
        whisper = build_whisper(context)
        assert "Active Project" in whisper or "Caretaker" in whisper

    def test_whisper_no_crash_empty_db(self):
        from retrieval.retrieval_engine import retrieve_context
        context = retrieve_context("random message")
        whisper = build_whisper(context)
        assert isinstance(whisper, str)
        assert len(whisper) > 0


class TestMCPTools:

    def test_get_context_returns_string(self):
        result = get_context("What project am I building?")
        assert isinstance(result, str)

    def test_get_context_has_caretaker_header(self):
        result = get_context("What project am I building?")
        assert "[CARETAKER CONTEXT]" in result

    def test_get_context_has_instruction(self):
        result = get_context("What project am I building?")
        assert "IMPORTANT" in result or "INSTRUCTION" in result or "memory" in result.lower()

    def test_get_context_no_crash_empty_db(self):
        result = get_context("random message")
        assert isinstance(result, str)

    def test_save_message_returns_string(self):
        result = save_message("I am building Caretaker project.")
        assert isinstance(result, str)

    def test_save_message_confirms_save(self):
        result = save_message("I prefer Python for all projects.")
        assert "saved" in result.lower() or "Memory" in result

    def test_save_message_contains_memory_id(self):
        result = save_message("I am a full stack developer.")
        assert "id=" in result

    def test_save_message_contains_type(self):
        result = save_message("I am building a React app.")
        assert "type=" in result

    def test_get_context_after_save_has_memory(self):
        save_message("I am building Caretaker universal memory layer.")
        result = get_context("What am I building?")
        assert "Caretaker" in result

    def test_save_message_custom_agent(self):
        result = save_message("I prefer TypeScript.", agent_id="gpt4")
        assert isinstance(result, str)
        assert "saved" in result.lower() or "Memory" in result

    def test_get_context_custom_agent(self):
        result = get_context("What do I prefer?", agent_id="gemini")
        assert isinstance(result, str)
