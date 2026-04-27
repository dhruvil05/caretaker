import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import uuid
from datetime import datetime, timezone
from storage.local_db import (
    run_migrations,
    save_memory,
    get_memories_by_type,
    get_recent_memories,
    get_memory_by_id,
    update_memory_status,
    increment_retrieval_count,
    get_all_active_memories,
)


def make_memory(overrides: dict = {}) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    base = {
        "id":              str(uuid.uuid4()),
        "source_agent":    "claude",
        "keywords":        '["test", "caretaker"]',
        "short":           None,
        "full":            "Test memory for unit testing purposes.",
        "type":            "PROJECT",
        "subtype":         "project",
        "fact_type":       "REPLACEABLE",
        "status":          "ACTIVE",
        "superseded_by":   None,
        "importance":      0.7,
        "decay_score":     1.0,
        "temperature":     "PRIORITY_HOT",
        "retrieval_count": 0,
        "created_at":      now,
        "updated_at":      now,
        "last_used":       now,
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def setup_db():
    run_migrations()


class TestSaveMemory:

    def test_save_returns_true(self):
        mem = make_memory()
        result = save_memory(mem)
        assert result is True

    def test_saved_memory_retrievable(self):
        mem = make_memory()
        save_memory(mem)
        found = get_memory_by_id(mem["id"])
        assert found is not None
        assert found["id"] == mem["id"]

    def test_saved_memory_content_correct(self):
        mem = make_memory()
        save_memory(mem)
        found = get_memory_by_id(mem["id"])
        assert found["full"]        == mem["full"]
        assert found["type"]        == mem["type"]
        assert found["temperature"] == mem["temperature"]
        assert found["importance"]  == mem["importance"]

    def test_duplicate_id_returns_false(self):
        mem = make_memory()
        save_memory(mem)
        result = save_memory(mem)
        assert result is False


class TestGetMemoriesByType:

    def test_returns_list(self):
        result = get_memories_by_type("PROJECT")
        assert isinstance(result, list)

    def test_filters_by_type(self):
        mem = make_memory({"type": "PREFERENCE", "subtype": "preference"})
        save_memory(mem)
        results = get_memories_by_type("PREFERENCE")
        types = [m["type"] for m in results]
        assert all(t == "PREFERENCE" for t in types)

    def test_only_returns_active(self):
        mem = make_memory({"status": "ARCHIVED"})
        save_memory(mem)
        results = get_memories_by_type("PROJECT", status="ACTIVE")
        ids = [m["id"] for m in results]
        assert mem["id"] not in ids

    def test_no_type_returns_all_active(self):
        mem1 = make_memory({"type": "PROJECT"})
        mem2 = make_memory({"type": "LEARNING"})
        save_memory(mem1)
        save_memory(mem2)
        results = get_memories_by_type()
        assert len(results) >= 2


class TestGetRecentMemories:

    def test_returns_list(self):
        result = get_recent_memories()
        assert isinstance(result, list)

    def test_respects_limit(self):
        for _ in range(5):
            save_memory(make_memory())
        result = get_recent_memories(limit=3)
        assert len(result) <= 3

    def test_only_returns_active(self):
        mem = make_memory({"status": "ARCHIVED"})
        save_memory(mem)
        results = get_recent_memories(limit=10)
        ids = [m["id"] for m in results]
        assert mem["id"] not in ids


class TestGetMemoryById:

    def test_returns_none_for_missing(self):
        result = get_memory_by_id("nonexistent-id-xyz")
        assert result is None

    def test_returns_dict_for_existing(self):
        mem = make_memory()
        save_memory(mem)
        result = get_memory_by_id(mem["id"])
        assert isinstance(result, dict)


class TestUpdateMemoryStatus:

    def test_status_updated_to_outdated(self):
        mem = make_memory()
        save_memory(mem)
        update_memory_status(mem["id"], "OUTDATED")
        found = get_memory_by_id(mem["id"])
        assert found["status"] == "OUTDATED"

    def test_superseded_by_saved(self):
        mem1 = make_memory()
        mem2 = make_memory()
        save_memory(mem1)
        save_memory(mem2)
        update_memory_status(mem1["id"], "OUTDATED", superseded_by=mem2["id"])
        found = get_memory_by_id(mem1["id"])
        assert found["superseded_by"] == mem2["id"]


class TestIncrementRetrievalCount:

    def test_count_increments(self):
        mem = make_memory()
        save_memory(mem)
        increment_retrieval_count(mem["id"])
        found = get_memory_by_id(mem["id"])
        assert found["retrieval_count"] == 1

    def test_count_increments_multiple_times(self):
        mem = make_memory()
        save_memory(mem)
        increment_retrieval_count(mem["id"])
        increment_retrieval_count(mem["id"])
        increment_retrieval_count(mem["id"])
        found = get_memory_by_id(mem["id"])
        assert found["retrieval_count"] == 3


class TestGetAllActiveMemories:

    def test_returns_list(self):
        result = get_all_active_memories()
        assert isinstance(result, list)

    def test_excludes_archived(self):
        mem = make_memory({"status": "ARCHIVED"})
        save_memory(mem)
        results = get_all_active_memories()
        ids = [m["id"] for m in results]
        assert mem["id"] not in ids

    def test_includes_active(self):
        mem = make_memory()
        save_memory(mem)
        results = get_all_active_memories()
        ids = [m["id"] for m in results]
        assert mem["id"] in ids
