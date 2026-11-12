"""
tests/phase3/test_summaries.py
Memories get a summary even on the default install (no vector index), lost
summary jobs are picked up again, and recall works well before a summary exists.
"""

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

import caretaker.storage.local_db as db


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "summaries.db"))
    monkeypatch.setenv("CARETAKER_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("CARETAKER_DATA_DIR", str(tmp_path / "data"))
    db._FTS_CACHE.clear()
    db.run_migrations()


def mem(full, **kw):
    now = datetime.now(timezone.utc).isoformat()
    m = {"id": str(uuid.uuid4()), "source_agent": "claude", "keywords": None, "short": None, "full": full,
         "type": "PROJECT", "subtype": None, "fact_type": "ADDITIVE", "status": "ACTIVE", "superseded_by": None,
         "importance": 0.6, "decay_score": 1.0, "temperature": "HOT", "retrieval_count": 0,
         "created_at": now, "updated_at": now, "last_used": None}
    m.update(kw)
    db.save_memory(m)
    return m


class FirstWords:
    calls = 0

    def compress(self, text, memory_type):
        FirstWords.calls += 1
        return " ".join(text.split()[:4]), ["k"]


async def drain(q, timeout=10.0):
    await q.start()
    await asyncio.wait_for(q._queue.join(), timeout)
    await q.stop()


# ── Backlog ────────────────────────────────────────────────────────────────────

def test_backlog_lists_only_active_memories_without_a_summary():
    a = mem("needs a summary", importance=0.4)
    b = mem("more important, also needs one", importance=0.9)
    mem("already summarised", short="done")
    mem("archived", status="ARCHIVED")
    mem("   ")
    ids = [r["id"] for r in db.get_unsummarised(10)]
    assert ids == [b["id"], a["id"]]


def test_queue_works_without_vector_index_and_drains_backlog():
    from caretaker.scheduler.compression_queue import CompressionQueue
    m1 = mem("Orbit deploys on Fridays after the release review")
    m2 = mem("Billing exports run nightly at two in the morning")
    q = CompressionQueue(FirstWords(), db)          # no vector_db: default install
    assert q.enqueue_backlog() == 2
    q.submit(m1["id"], m1["full"], "PROJECT")       # a save racing the backlog
    assert q.queue_size == 2                         # duplicate skipped
    asyncio.run(drain(q))
    assert db.get_memory_by_id(m1["id"])["short"] == "Orbit deploys on Fridays"
    assert db.get_memory_by_id(m2["id"])["short"] == "Billing exports run nightly"
    assert db.get_unsummarised(10) == []


def test_failed_summary_retries_a_limited_number_of_times():
    from caretaker.scheduler.compression_queue import CompressionQueue
    calls = []

    class Broken:
        def compress(self, text, memory_type):
            calls.append(1)
            raise RuntimeError("model unavailable")

    m = mem("text that cannot be summarised right now")
    q = CompressionQueue(Broken(), db)
    q.BASE_BACKOFF = 0.001
    q.submit(m["id"], m["full"], "PROJECT")
    asyncio.run(drain(q))
    assert len(calls) == q.MAX_RETRIES              # used to restart at 0 and retry forever
    assert db.get_memory_by_id(m["id"])["status"] == "ACTIVE"


def test_server_starts_queue_and_backlog_without_vector_index():
    import inspect
    from caretaker.mcp_server import server
    src = inspect.getsource(server)
    assert "if _compressor and _vector_db:" not in src
    assert "enqueue_backlog()" in src


# ── Recall before a summary exists ─────────────────────────────────────────────

def test_quick_summary_is_short_and_cut_at_a_word():
    from caretaker.compression.local_compressor import quick_summary
    long = "Orbit is our deployment tool and " + "it handles many environments " * 30
    s = quick_summary(long)
    assert len(s) <= 241 and s.endswith("…") and not s[:-1].endswith(" ")
    assert quick_summary("Use tabs. Not spaces. Ever. Really.") == "Use tabs. Not spaces. Ever."
    assert quick_summary("One. " + "x" * 300) == "One."
    assert quick_summary("") == ""


def test_recall_fills_missing_summary_and_keeps_cold_memories():
    from caretaker.retrieval import memory_selector
    from caretaker.retrieval.retrieval_engine import _fill_missing_summaries
    cold = {"id": "c", "temperature": "COLD", "short": None, "full": "Cold fact about Orbit. More detail here.",
            "full_text": "Cold fact about Orbit. More detail here.", "relevance_score": 0.9}
    _fill_missing_summaries([cold])
    assert cold["short"] == "Cold fact about Orbit. More detail here." and cold["_quick_summary"]
    selected, _ = memory_selector.select_memory_forms([cold], token_budget=1000)
    assert selected and selected[0]["selected_form"] == "SHORT"


def test_keyword_recall_gives_unsummarised_memories_a_summary():
    from caretaker.retrieval.retrieval_engine import retrieve_context
    mem("Zylo003 is the codename for the new billing service. It replaces the old invoicing cron.")
    ctx = retrieve_context("what is Zylo003?", dry_run=True)
    assert ctx["relevant"] and ctx["relevant"][0]["short"].startswith("Zylo003 is the codename")
    assert db.get_unsummarised(10)                  # nothing was written back


def test_injector_keeps_most_of_a_summary():
    from caretaker.mcp_server.injector import SHORT_MAX_CHARS, build_whisper
    short = "Deploys go through the release checklist " * 5        # ~200 chars
    out = build_whisper({"relevant": [{"id": "x", "type": "PROJECT", "temperature": "WARM", "status": "ACTIVE",
                                       "short": short, "full": short + " and more", "selected_form": "SHORT"}],
                         "recent": [], "use_full": False})
    assert SHORT_MAX_CHARS >= 200
    assert short.strip()[:190] in out
