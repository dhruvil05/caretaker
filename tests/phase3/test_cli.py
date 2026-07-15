"""
tests/phase3/test_cli.py
Phase 3 — CLI Tests: P3-T01 through P3-T10

Tests all CLI commands using Click's CliRunner (no terminal needed).
Runs against isolated SQLite DB per test via monkeypatch.

P3-T01  CLI Starts    --help shows output, no errors
P3-T02  CLI List      list returns rows sorted by temperature
P3-T03  CLI View      view <id> shows all fields
P3-T04  CLI Search    search 'python' returns results
P3-T05  CLI Edit      edit short field → DB updated
P3-T06  CLI Delete    delete → status ARCHIVED, record still exists
P3-T07  CLI Restore   archive then restore → status ACTIVE
P3-T08  CLI Stats     stats shows counts by type/status/temperature
P3-T09  CLI Export    export produces valid JSON with all memories
P3-T10  CLI Import    export → wipe → import → all memories restored
"""

import sys
import json
import uuid
import re
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from click.testing import CliRunner
from caretaker.cli.main import cli


# ── Isolated DB per test ───────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    import caretaker.storage.local_db as db_module
    test_db = tmp_path / "caretaker_test.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(test_db))
    db_module.run_migrations()
    yield test_db


def _make_memory(**overrides) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    base = {
        "id"            : str(uuid.uuid4()),
        "source_agent"  : "claude",
        "keywords"      : '["python", "caretaker", "project"]',
        "short"         : "User builds Caretaker memory system in Python.",
        "full"          : "I am building a FastAPI project called Caretaker. It is a universal memory layer for AI agents.",
        "type"          : "PROJECT",
        "subtype"       : "build",
        "fact_type"     : "REPLACEABLE",
        "status"        : "ACTIVE",
        "superseded_by" : None,
        "importance"    : 0.8,
        "decay_score"   : 0.9,
        "temperature"   : "HOT",
        "retrieval_count": 2,
        "created_at"    : now,
        "updated_at"    : now,
        "last_used"     : now,
        "source_msg"    : "I am building a FastAPI project.",
    }
    base.update(overrides)
    return base


def _seed(**overrides) -> dict:
    from caretaker.storage.local_db import save_memory
    mem = _make_memory(**overrides)
    save_memory(mem)
    return mem


# ── P3-T01 ─────────────────────────────────────────────────────────────────────

class TestP3T01_CLIStarts:

    def test_help_exits_zero(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_help_lists_all_commands(self):
        result = CliRunner().invoke(cli, ["--help"])
        for cmd in ["list", "view", "search", "edit", "delete",
                    "restore", "stats", "export", "import", "sync", "config"]:
            assert cmd in result.output, f"'{cmd}' missing from help"

    def test_no_exception_on_start(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exception is None


# ── P3-T02 ─────────────────────────────────────────────────────────────────────

class TestP3T02_CLIList:

    def test_list_shows_memories(self):
        _seed()
        result = CliRunner().invoke(cli, ["list"])
        assert result.exit_code == 0
        assert result.exception is None

    def test_list_filter_by_type(self):
        _seed(type="PROJECT")
        _seed(type="PREFERENCE", short="Prefers Python.",
              full="I prefer Python for all work.", keywords='["python","preference"]')
        result = CliRunner().invoke(cli, ["list", "--type", "PROJECT"])
        assert result.exit_code == 0
        assert "PROJECT" in result.output

    def test_list_outdated_flag(self):
        _seed(status="OUTDATED")
        result = CliRunner().invoke(cli, ["list", "--outdated"])
        assert result.exit_code == 0
        assert "OUTDATED" in result.output

    def test_list_hot_before_warm(self):
        _seed(temperature="WARM", importance=0.3)
        _seed(temperature="HOT",  importance=0.8)
        result = CliRunner().invoke(cli, ["list"])
        assert result.output.index("HOT") < result.output.index("WARM")

    def test_list_empty_db_no_crash(self):
        result = CliRunner().invoke(cli, ["list"])
        assert result.exit_code == 0
        assert result.exception is None


# ── P3-T03 ─────────────────────────────────────────────────────────────────────

class TestP3T03_CLIView:

    def test_view_shows_all_key_fields(self):
        mem = _seed()
        result = CliRunner().invoke(cli, ["view", mem["id"]])
        assert result.exit_code == 0
        for val in [mem["id"], mem["type"], mem["status"], mem["temperature"]]:
            assert val in result.output

    def test_view_shows_full_and_short_labels(self):
        mem = _seed()
        result = CliRunner().invoke(cli, ["view", mem["id"]])
        assert "FULL" in result.output
        assert "SHORT" in result.output

    def test_view_partial_id_prefix_works(self):
        mem = _seed()
        result = CliRunner().invoke(cli, ["view", mem["id"][:8]])
        assert result.exit_code == 0
        assert mem["id"] in result.output

    def test_view_missing_id_gives_error_message(self):
        result = CliRunner().invoke(cli, ["view", "not-a-real-id"])
        assert result.exit_code == 0
        assert "not found" in result.output.lower()


# ── P3-T04 ─────────────────────────────────────────────────────────────────────

class TestP3T04_CLISearch:

    def test_search_no_crash(self):
        _seed()
        result = CliRunner().invoke(cli, ["search", "python project"])
        assert result.exit_code == 0
        assert result.exception is None

    def test_search_empty_db_no_crash(self):
        result = CliRunner().invoke(cli, ["search", "anything at all"])
        assert result.exit_code == 0
        assert result.exception is None

    def test_search_output_has_query_echo(self):
        _seed()
        result = CliRunner().invoke(cli, ["search", "caretaker memory"])
        import re
        assert result.exit_code == 0
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "caretaker memory" in clean


# ── P3-T05 ─────────────────────────────────────────────────────────────────────

class TestP3T05_CLIEdit:

    def test_edit_updates_short_in_db(self, monkeypatch):
        mem = _seed(short="Original short text here.")
        new_short = "Completely new short summary after edit."

        import subprocess as _sub

        def fake_run(cmd, check=False, **kw):
            path = cmd[1]
            with open(path, "r") as f:
                content = f.read()
            updated = re.sub(
                r'\[SHORT\](.*?)\[/SHORT\]',
                f'[SHORT]\n{new_short}\n[/SHORT]',
                content, flags=re.DOTALL
            )
            with open(path, "w") as f:
                f.write(updated)

        monkeypatch.setattr(_sub, "run", fake_run)

        result = CliRunner().invoke(cli, ["edit", mem["id"], "--field", "short"])
        assert result.exit_code == 0

        from caretaker.storage.local_db import get_memory_by_id
        updated = get_memory_by_id(mem["id"])
        assert updated["short"] == new_short

    def test_edit_no_change_skips(self, monkeypatch):
        mem = _seed(short="Unchanged text.")

        import subprocess as _sub
        monkeypatch.setattr(_sub, "run", lambda *a, **k: None)

        result = CliRunner().invoke(cli, ["edit", mem["id"], "--field", "short"])
        assert result.exit_code == 0
        assert result.exception is None


# ── P3-T06 ─────────────────────────────────────────────────────────────────────

class TestP3T06_CLIDelete:

    def test_delete_sets_archived_not_hard_deleted(self):
        mem = _seed()
        CliRunner().invoke(cli, ["delete", mem["id"], "--force"])

        from caretaker.storage.local_db import get_memory_by_id
        row = get_memory_by_id(mem["id"])
        assert row is not None, "Memory was hard-deleted — should only be ARCHIVED"
        assert row["status"] == "ARCHIVED"

    def test_delete_cancel_leaves_active(self):
        mem = _seed()
        CliRunner().invoke(cli, ["delete", mem["id"]], input="n\n")

        from caretaker.storage.local_db import get_memory_by_id
        row = get_memory_by_id(mem["id"])
        assert row["status"] == "ACTIVE"

    def test_delete_already_archived_warns(self):
        mem = _seed(status="ARCHIVED")
        result = CliRunner().invoke(cli, ["delete", mem["id"], "--force"])
        assert result.exit_code == 0
        assert "already" in result.output.lower() or "ARCHIVED" in result.output

    def test_delete_not_found_no_crash(self):
        result = CliRunner().invoke(cli, ["delete", "no-such-id", "--force"])
        assert result.exit_code == 0
        assert result.exception is None


# ── P3-T07 ─────────────────────────────────────────────────────────────────────

class TestP3T07_CLIRestore:

    def test_restore_archived_to_active(self):
        mem = _seed(status="ARCHIVED")
        CliRunner().invoke(cli, ["restore", mem["id"]])

        from caretaker.storage.local_db import get_memory_by_id
        row = get_memory_by_id(mem["id"])
        assert row["status"] == "ACTIVE"

    def test_restore_outdated_to_active(self):
        mem = _seed(status="OUTDATED")
        CliRunner().invoke(cli, ["restore", mem["id"]])

        from caretaker.storage.local_db import get_memory_by_id
        row = get_memory_by_id(mem["id"])
        assert row["status"] == "ACTIVE"

    def test_restore_clears_superseded_by(self):
        mem = _seed(status="OUTDATED", superseded_by=str(uuid.uuid4()))
        CliRunner().invoke(cli, ["restore", mem["id"]])

        from caretaker.storage.local_db import get_memory_by_id
        row = get_memory_by_id(mem["id"])
        assert row["superseded_by"] is None

    def test_restore_already_active_no_crash(self):
        mem = _seed(status="ACTIVE")
        result = CliRunner().invoke(cli, ["restore", mem["id"]])
        assert result.exit_code == 0
        assert result.exception is None


# ── P3-T08 ─────────────────────────────────────────────────────────────────────

class TestP3T08_CLIStats:

    def test_stats_runs(self):
        result = CliRunner().invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert result.exception is None

    def test_stats_shows_status_counts(self):
        _seed(status="ACTIVE")
        _seed(status="ARCHIVED")
        result = CliRunner().invoke(cli, ["stats"])
        assert "ACTIVE" in result.output
        assert "ARCHIVED" in result.output

    def test_stats_shows_temperature(self):
        _seed(temperature="HOT")
        _seed(temperature="WARM")
        result = CliRunner().invoke(cli, ["stats"])
        assert "HOT" in result.output
        assert "WARM" in result.output

    def test_stats_shows_type(self):
        _seed(type="PROJECT")
        result = CliRunner().invoke(cli, ["stats"])
        assert "PROJECT" in result.output

    def test_stats_shows_agent(self):
        _seed(source_agent="claude")
        result = CliRunner().invoke(cli, ["stats"])
        assert "claude" in result.output


# ── P3-T09 ─────────────────────────────────────────────────────────────────────

class TestP3T09_CLIExport:

    def test_export_creates_file(self, tmp_path):
        _seed()
        out = tmp_path / "test_export.json"
        result = CliRunner().invoke(cli, ["export", "--file", str(out)])
        assert result.exit_code == 0
        assert out.exists()

    def test_export_valid_json(self, tmp_path):
        _seed()
        out = tmp_path / "test_export.json"
        CliRunner().invoke(cli, ["export", "--file", str(out)])
        data = json.loads(out.read_text())
        assert data.get("caretaker_export") is True
        assert isinstance(data.get("memories"), list)

    def test_export_contains_all_memories(self, tmp_path):
        m1 = _seed()
        m2 = _seed(type="PREFERENCE", short="Prefers Python.",
                   full="I prefer Python.", keywords='["python"]')
        out = tmp_path / "test_all.json"
        CliRunner().invoke(cli, ["export", "--file", str(out)])
        data = json.loads(out.read_text())
        ids = [m["id"] for m in data["memories"]]
        assert m1["id"] in ids
        assert m2["id"] in ids

    def test_export_has_metadata_fields(self, tmp_path):
        _seed()
        out = tmp_path / "test_meta.json"
        CliRunner().invoke(cli, ["export", "--file", str(out)])
        data = json.loads(out.read_text())
        assert "exported_at" in data
        assert "total" in data
        assert "version" in data


# ── P3-T10 ─────────────────────────────────────────────────────────────────────

class TestP3T10_CLIImport:

    def test_import_restores_all_memories(self, tmp_path):
        import caretaker.storage.local_db as db_module

        m1 = _seed()
        m2 = _seed(type="PREFERENCE", short="Prefers Python.",
                   full="I prefer Python.", keywords='["python"]')
        m3 = _seed(type="LEARNING", short="Learning Rust.",
                   full="I am learning Rust.", keywords='["rust"]')

        out = tmp_path / "restore_test.json"
        CliRunner().invoke(cli, ["export", "--file", str(out)])

        # Wipe all records
        with db_module.get_connection() as conn:
            conn.execute("DELETE FROM memories")

        # Re-import
        result = CliRunner().invoke(cli, ["import", str(out), "--force"])
        assert result.exit_code == 0
        assert result.exception is None

        from caretaker.storage.local_db import get_memory_by_id
        for mem in [m1, m2, m3]:
            row = get_memory_by_id(mem["id"])
            assert row is not None, f"Memory {mem['id'][:8]} not restored"
            assert row["type"] == mem["type"]

    def test_import_skips_existing(self, tmp_path):
        mem = _seed()
        out = tmp_path / "skip_test.json"
        CliRunner().invoke(cli, ["export", "--file", str(out)])

        result = CliRunner().invoke(
            cli, ["import", str(out), "--force", "--skip-existing"]
        )
        assert result.exit_code == 0
        assert "Skipped" in result.output or "skipped" in result.output.lower()

    def test_import_nonexistent_file_no_crash(self):
        result = CliRunner().invoke(cli, ["import", "no_such_file.json", "--force"])
        assert result.exit_code == 0
        assert result.exception is None