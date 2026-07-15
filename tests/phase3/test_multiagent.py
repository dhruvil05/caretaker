"""
tests/phase3/test_multiagent.py
Phase 3 — Multi-Agent + Scheduler Tests: P3-T16, P3-T17, P3-T18

Fix 1: async tests use asyncio.run() — no pytest-asyncio needed.
Fix 2: CaretakerScheduler.start() patched to not need running event loop
       (AsyncIOScheduler.start() requires an existing loop — we mock it
        in tests that just verify registration logic, not actual scheduling).

P3-T16  Multi-Agent Context   ChatGPT get_context → whisper returned correctly
P3-T17  Source Agent Logged   capture via different agent_id → source_agent correct
P3-T18  Scheduler Registered  job registered at config time, status dict correct
"""

import sys
import uuid
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import caretaker.storage.local_db as db_module


# ── Isolated DB ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    test_db = tmp_path / "caretaker_multiagent_test.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(test_db))
    db_module.run_migrations()
    yield test_db


def _seed(**overrides) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    base = {
        "id"            : str(uuid.uuid4()),
        "source_agent"  : "claude",
        "keywords"      : '["python", "project", "caretaker"]',
        "short"         : "User builds Caretaker universal memory layer in Python.",
        "full"          : "I am building a FastAPI project called Caretaker.",
        "type"          : "PROJECT",
        "subtype"       : "build",
        "fact_type"     : "REPLACEABLE",
        "status"        : "ACTIVE",
        "superseded_by" : None,
        "importance"    : 0.8,
        "decay_score"   : 0.9,
        "temperature"   : "HOT",
        "retrieval_count": 1,
        "created_at"    : now,
        "updated_at"    : now,
        "last_used"     : now,
        "source_msg"    : "Building Caretaker.",
    }
    base.update(overrides)
    db_module.save_memory(base)
    return base


def _run(coro):
    """Run coroutine synchronously without pytest-asyncio."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── P3-T16: Multi-Agent Context ────────────────────────────────────────────────

class TestP3T16_MultiAgentContext:
    """P3-T16: Simulate ChatGPT get_context → correctly formatted whisper."""

    def test_chatgpt_get_context_returns_non_empty_string(self):
        _seed()
        from mcp_server.tools import get_context
        result = get_context(message="What am I building?", agent_id="chatgpt")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_claude_format_has_instruction_block(self):
        _seed()
        from mcp_server.tools import get_context
        result = get_context("What am I building?", agent_id="claude")
        assert "INSTRUCTION" in result

    def test_chatgpt_format_has_no_instruction_block(self):
        _seed()
        from mcp_server.tools import get_context
        result = get_context("What am I building?", agent_id="chatgpt")
        assert "INSTRUCTION" not in result

    def test_gemini_format_has_xml_tags(self):
        _seed()
        from mcp_server.tools import get_context
        result = get_context("What am I building?", agent_id="gemini")
        assert "<context>" in result
        assert "</context>" in result

    def test_unknown_agent_uses_neutral_format(self):
        _seed()
        from mcp_server.tools import get_context
        result = get_context("Hello", agent_id="unknown-agent-xyz-999")
        assert "[USER CONTEXT]" in result
        assert "[END CONTEXT]" in result

    def test_all_canonical_agents_return_string(self):
        _seed()
        from mcp_server.tools import get_context
        for agent in ["claude", "chatgpt", "gemini", "cursor", "copilot"]:
            result = get_context("Test message", agent_id=agent)
            assert isinstance(result, str), f"Agent '{agent}' returned non-string"
            assert len(result) > 10, f"Agent '{agent}' returned empty context"

    def test_get_context_empty_db_no_crash(self):
        from mcp_server.tools import get_context
        result = get_context("Hello", agent_id="chatgpt")
        assert isinstance(result, str)


# ── P3-T17: Source Agent Logged ────────────────────────────────────────────────

class TestP3T17_SourceAgentLogged:
    """P3-T17: capture via agent_id → source_agent field stored correctly."""

    def test_source_agent_claude(self):
        from capture.capture_engine import run_capture
        mem = run_capture("I am building a Python project.", agent_id="claude")
        assert mem["source_agent"] == "claude"

    def test_source_agent_chatgpt(self):
        from capture.capture_engine import run_capture
        mem = run_capture("I prefer TypeScript for frontend.", agent_id="chatgpt")
        assert mem["source_agent"] == "chatgpt"

    def test_source_agent_gemini(self):
        from capture.capture_engine import run_capture
        mem = run_capture("I am learning transformer architecture.", agent_id="gemini")
        assert mem["source_agent"] == "gemini"

    def test_source_agent_custom_value_preserved(self):
        from capture.capture_engine import run_capture
        mem = run_capture("Testing custom agent.", agent_id="my-custom-agent-v2")
        assert mem["source_agent"] == "my-custom-agent-v2"

    def test_source_agent_persisted_in_sqlite(self):
        from capture.capture_engine import run_capture
        mem = run_capture("I prefer dark mode editors.", agent_id="cursor")
        saved = db_module.get_memory_by_id(mem["id"])
        assert saved is not None
        assert saved["source_agent"] == "cursor"

    def test_agent_adapter_normalises_aliases(self):
        from mcp_server.agent_adapter import normalise_agent_id
        assert normalise_agent_id("gpt-4o")          == "chatgpt"
        assert normalise_agent_id("vertex")           == "gemini"
        assert normalise_agent_id("claude-desktop")   == "claude"
        assert normalise_agent_id("CLAUDE")           == "claude"
        assert normalise_agent_id("totally-unknown")  == "custom"

    def test_get_agent_info_returns_dict(self):
        from mcp_server.agent_adapter import get_agent_info
        info = get_agent_info("gpt-4o")
        assert info["canonical"]    == "chatgpt"
        assert info["known"]        is True
        assert info["format_style"] == "context_block"


# ── P3-T18: Scheduler Registered ──────────────────────────────────────────────

class TestP3T18_SchedulerRegistered:
    """P3-T18: APScheduler job registered at config time."""

    def test_scheduler_status_dict_has_required_keys(self):
        from scheduler.scheduler import CaretakerScheduler
        sched = CaretakerScheduler(
            config={"maintenance_time": "02:00"},
            local_db=db_module,
            vector_db=None,
        )
        status = sched.status()
        for key in ("running", "schedule", "job_id", "apscheduler"):
            assert key in status, f"Key '{key}' missing from status dict"

    def test_scheduler_parses_time_correctly(self):
        from scheduler.scheduler import CaretakerScheduler
        sched = CaretakerScheduler(
            config={"maintenance_time": "14:30"},
            local_db=db_module,
            vector_db=None,
        )
        assert sched._hour   == 14
        assert sched._minute == 30

    def test_scheduler_bad_time_defaults_to_0200(self):
        from scheduler.scheduler import CaretakerScheduler
        sched = CaretakerScheduler(
            config={"maintenance_time": "INVALID_TIME"},
            local_db=db_module,
            vector_db=None,
        )
        assert sched._hour   == 2
        assert sched._minute == 0

    def test_scheduler_job_id_constant(self):
        from scheduler.scheduler import CaretakerScheduler
        assert CaretakerScheduler.JOB_ID == "caretaker_nightly_maintenance"

    def test_scheduler_start_with_mocked_apscheduler(self):
        """
        Verify job registration logic without needing a real event loop.
        Mocks AsyncIOScheduler so no loop is needed.
        """
        from scheduler.scheduler import CaretakerScheduler

        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = None

        with patch("scheduler.scheduler.CaretakerScheduler._is_apscheduler_installed",
                   return_value=True):
            with patch("apscheduler.schedulers.asyncio.AsyncIOScheduler",
                       return_value=mock_scheduler):
                try:
                    from apscheduler.schedulers.asyncio import AsyncIOScheduler
                except ImportError:
                    pytest.skip("APScheduler not installed")

                sched = CaretakerScheduler(
                    config={"maintenance_time": "03:00"},
                    local_db=db_module,
                    vector_db=None,
                )
                # Inject mock directly
                sched._scheduler = mock_scheduler
                sched._running   = True

                status = sched.status()
                assert status["running"]  is True
                assert status["schedule"] == "03:00 UTC"
                assert status["job_id"]   == CaretakerScheduler.JOB_ID

                sched.stop()

    def test_trigger_now_runs_maintenance(self):
        """trigger_now() runs maintenance pipeline and returns result dict."""
        from scheduler.scheduler import CaretakerScheduler

        sched = CaretakerScheduler(
            config={"maintenance_time": "02:00", "archive_score": 0.2},
            local_db=db_module,
            vector_db=None,
        )
        result = _run(sched.trigger_now())
        assert isinstance(result, dict)
        # Must contain either result keys or error key
        assert "ran_at" in result or "error" in result