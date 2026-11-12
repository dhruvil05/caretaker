"""
tests/phase3/test_privacy_flows.py
Sensitive-data protection across every path text takes: save, recall, search,
summariser, import, dashboard edit, existing-data scan and the review actions.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone

import pytest

import caretaker.storage.local_db as db_module

KEY = "sk-ant-" + "api03-" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1Qq0Pp9"


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "privacy_flows.db"))
    db_module.run_migrations()


def raw_row(**overrides):
    """Insert a memory directly, as if saved before 0.5 (no protection applied)."""
    now = datetime.now(timezone.utc).isoformat()
    mem = {
        "id": str(uuid.uuid4()), "source_agent": "claude", "keywords": None, "short": None,
        "full": f"Orbit deploys with key {KEY}", "type": "PROJECT", "subtype": None,
        "fact_type": "ADDITIVE", "status": "ACTIVE", "superseded_by": None, "importance": 0.8,
        "decay_score": 1.0, "temperature": "HOT", "retrieval_count": 0,
        "created_at": now, "updated_at": now, "last_used": None,
    }
    mem.update(overrides)
    db_module.save_memory(mem)
    return mem


def all_text():
    with db_module.get_connection() as conn:
        return " ".join(f"{r['full']} {r['short'] or ''}" for r in conn.execute("SELECT full, short FROM memories"))


# ── Save ───────────────────────────────────────────────────────────────────────

def test_save_redacts_before_storage_and_reports_it():
    from caretaker.mcp_server.tools import save_message
    reply = save_message(f"I am building Orbit and the Anthropic key is {KEY}")
    assert "privacy: 1 Anthropic API key redacted" in reply
    assert KEY not in all_text() and "[REDACTED: Anthropic API key]" in all_text()
    with db_module.get_connection() as conn:
        row = dict(conn.execute("SELECT sensitivity, privacy_findings FROM memories").fetchone())
    assert row["sensitivity"] == "redacted"
    assert json.loads(row["privacy_findings"])[0]["kind"] == "anthropic_api_key"


def test_save_keeps_and_flags_contact_details():
    from caretaker.mcp_server.tools import save_message
    reply = save_message("I am building Orbit, reach me at dhruvil@example.com")
    assert "flagged for review" in reply
    assert "dhruvil@example.com" in all_text()
    with db_module.get_connection() as conn:
        assert conn.execute("SELECT sensitivity FROM memories").fetchone()[0] == "flagged"


def test_long_message_summariser_never_sees_the_secret(monkeypatch):
    from caretaker.capture import capture_engine
    seen = []

    class SpyCompressor:
        def compress(self, text, memory_type):
            seen.append(text)
            return "summary", ["orbit"]

    monkeypatch.setattr(capture_engine, "is_long_message", lambda m: True)
    monkeypatch.setattr(capture_engine, "process_long_message",
                        lambda text, memory_type, compressor: [{"full_text": text, "short": compressor.compress(text, memory_type)[0],
                                                                "keywords": ["orbit"], "memory_type": memory_type, "chunk_index": 0}])
    capture_engine.run_capture(f"I am building Orbit. Key: {KEY}. " * 3, compressor=SpyCompressor())
    assert seen and all(KEY not in t for t in seen)


def test_background_summariser_redacts_old_memories():
    from caretaker.scheduler.compression_queue import CompressionQueue
    seen = []

    class SpyCompressor:
        def compress(self, text, memory_type):
            seen.append(text)
            return "short", ["k"]

    class FakeLocalDB:
        def update_compression(self, *a, **k): pass
        def update_status(self, *a, **k): pass
        def update_memory_fields(self, *a, **k): return True

    class FakeVectorDB:
        def upsert(self, **k): pass

    q = CompressionQueue(SpyCompressor(), FakeLocalDB(), FakeVectorDB())

    async def run():
        from caretaker.scheduler.compression_queue import CompressionJob
        await q._process_job(CompressionJob(memory_id="m1", full_text=f"old memory with {KEY}", memory_type="PROJECT"))

    asyncio.run(run())
    assert seen and KEY not in seen[0]


# ── Recall / search ────────────────────────────────────────────────────────────

def test_recall_redacts_secrets_saved_before_0_5():
    from caretaker.mcp_server.tools import get_context
    raw_row()
    out = get_context("How does Orbit deploy?")
    assert "Orbit" in out and KEY not in out


def test_search_and_list_output_redacts():
    from caretaker.mcp_server.tools import list_memories, search_memories
    raw_row()
    assert KEY not in search_memories("Orbit") and "REDACTED" in search_memories("Orbit")
    assert KEY not in list_memories()


# ── Existing data scan + review actions ────────────────────────────────────────

def test_scan_reports_then_applies():
    from caretaker.privacy.review import scan_memories
    secret = raw_row()
    email = raw_row(full="contact me at a@b.co")
    clean = raw_row(full="I prefer Python")

    report = scan_memories(apply=False)
    assert report["with_findings"] == 2 and report["changed"] == 0
    assert KEY in all_text()                                   # report-only changes nothing
    assert KEY not in json.dumps(report)                       # report never contains the secret

    result = scan_memories(apply=True)
    assert result["changed"] == 1 and KEY not in all_text()
    assert db_module.get_memory_by_id(secret["id"])["sensitivity"] == "redacted"
    assert db_module.get_memory_by_id(email["id"])["sensitivity"] == "flagged"
    assert db_module.get_memory_by_id(clean["id"])["sensitivity"] == "none"
    assert scan_memories(apply=True)["changed"] == 0           # idempotent


def test_mark_not_sensitive_and_redact_actions():
    from caretaker.privacy.review import mark_not_sensitive, redact_memory, scan_memories
    a = raw_row(full="team inbox is team@example.com")
    b = raw_row(full="my personal mail is me@example.com")
    scan_memories(apply=True)

    assert mark_not_sensitive(a["id"])
    assert db_module.get_memory_by_id(a["id"])["sensitivity"] == "none"
    assert all(i["id"] != a["id"] for i in scan_memories()["items"])   # not flagged again

    updated = redact_memory(b["id"])
    assert "me@example.com" not in updated["full"] and updated["sensitivity"] == "redacted"


def test_user_redaction_also_removes_medium_confidence_items():
    """Asking to remove the sensitive part must not leave a phone number behind."""
    from caretaker.privacy.review import redact_memory, scan_memories
    m = raw_row(full="Reach me at me@example.com or +91 98765 43210")
    scan_memories(apply=True)
    updated = redact_memory(m["id"])
    assert "me@example.com" not in updated["full"] and "98765" not in updated["full"]


# ── Import / dashboard edit ────────────────────────────────────────────────────

def test_import_applies_policy(tmp_path):
    from click.testing import CliRunner
    from caretaker.cli.main import cli
    import caretaker.storage.vector_db as vdb_module

    class NoVectorDB:
        def __init__(self, *a, **k): pass
        def initialize(self): pass
        def add(self, *a, **k): pass

    vdb_module_VectorDB = vdb_module.VectorDB
    vdb_module.VectorDB = NoVectorDB
    try:
        now = datetime.now(timezone.utc).isoformat()
        mem = {"id": "imp-1", "source_agent": "claude", "keywords": None, "short": None,
               "full": f"imported note with {KEY}", "type": "PROJECT", "subtype": None, "fact_type": "ADDITIVE",
               "status": "ACTIVE", "superseded_by": None, "importance": 0.5, "decay_score": 1.0,
               "temperature": "HOT", "retrieval_count": 0, "created_at": now, "updated_at": now, "last_used": None}
        path = tmp_path / "in.json"
        path.write_text(json.dumps({"caretaker_export": True, "memories": [mem]}), encoding="utf-8")
        result = CliRunner().invoke(cli, ["import", str(path), "--force"])
        assert result.exit_code == 0, result.output
    finally:
        vdb_module.VectorDB = vdb_module_VectorDB
    saved = db_module.get_memory_by_id("imp-1")
    assert KEY not in saved["full"] and saved["sensitivity"] == "redacted"


def test_dashboard_edit_redacts_pasted_key():
    from starlette.testclient import TestClient
    from caretaker.dashboard.app import create_app
    mem = raw_row(full="Orbit notes")
    client = TestClient(create_app("t", allowed_hosts=["testserver"]))
    r = client.patch(f"/api/memories/{mem['id']}", headers={"X-Caretaker-Token": "t"},
                     json={"full": f"Orbit notes, key {KEY}"})
    assert r.status_code == 200
    body = r.json()
    assert KEY not in body["full"] and body["sensitivity"] == "redacted"
    assert "Anthropic API key redacted" in body["privacy_note"]
    assert body["privacy_findings"][0]["kind"] == "anthropic_api_key"


def test_cli_privacy_scan_never_prints_secret():
    from click.testing import CliRunner
    from caretaker.cli.main import cli
    raw_row()
    out = CliRunner().invoke(cli, ["privacy", "scan"]).output
    assert "Anthropic API key" in out and "will be redacted" in out and KEY not in out
    out = CliRunner().invoke(cli, ["privacy", "scan", "--apply", "--force"]).output
    assert "1 memories changed" in out and KEY not in all_text()
