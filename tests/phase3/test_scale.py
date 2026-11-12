"""
tests/phase3/test_scale.py
0.5 scale work: full-text search (and its LIKE fallback), batched recall
writes, the relevance threshold, replacement/dedup that use index neighbours
and never cross projects, and SQLite-only CLI writes.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

import caretaker.storage.local_db as db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "scale.db"))
    db._FTS_CACHE.clear()
    db.run_migrations()


def mem(full, **kw):
    now = kw.pop("when", datetime.now(timezone.utc)).isoformat()
    m = {
        "id": str(uuid.uuid4()), "source_agent": "claude", "keywords": json.dumps(kw.pop("keywords", [])),
        "short": None, "full": full, "type": "PROJECT", "subtype": None, "fact_type": "REPLACEABLE",
        "status": "ACTIVE", "superseded_by": None, "importance": 0.6, "decay_score": 1.0,
        "temperature": "HOT", "retrieval_count": 0, "created_at": now, "updated_at": now, "last_used": None,
    }
    m.update(kw)
    db.save_memory(m)
    return m


# ── Full-text search ───────────────────────────────────────────────────────────

class TestFulltext:

    def test_ranked_prefix_search_with_filters(self):
        orbit = mem("Orbit uses PostgreSQL for scheduling data")
        mem("Nimbus is a weather app")
        archived = mem("Old Orbit prototype", status="ARCHIVED")
        warm = mem("Orbit dashboard redesign", temperature="COLD")

        ids = [r["id"] for r in db.search_fulltext("orbit postgres")]
        assert ids[0] == orbit["id"]                       # both terms match → ranked first
        assert archived["id"] not in ids                   # ACTIVE only by default
        assert warm["id"] not in [r["id"] for r in db.search_fulltext("orbit", temperatures=("HOT",))]
        assert all("match_score" in r for r in db.search_fulltext("orbit"))

    def test_project_filter_includes_global(self):
        a = mem("Orbit deploy notes", project="orbit")
        g = mem("I prefer concise answers about deploy", project="")
        other = mem("CRM deploy notes", project="crm")
        ids = {r["id"] for r in db.search_fulltext("deploy", project="orbit")}
        assert ids == {a["id"], g["id"]} and other["id"] not in ids

    def test_query_syntax_is_neutralised(self):
        mem("Orbit notes")
        for q in ['orbit" OR full:*', "NEAR(", "*", "AND OR NOT", '""']:
            db.search_fulltext(q)          # must not raise

    def test_like_fallback_without_fts(self, monkeypatch):
        orbit = mem("Orbit uses PostgreSQL")
        monkeypatch.setattr(db, "_has_fulltext", lambda conn: False)
        assert [r["id"] for r in db.search_fulltext("postgresql orbit")] == [orbit["id"]]


# ── Recall writes + threshold ──────────────────────────────────────────────────

def test_record_recalls_updates_in_one_go():
    a, b = mem("a note"), mem("b note", temperature="WARM")
    db.record_recalls([(a["id"], None), (b["id"], "HOT")])
    ra, rb = db.get_memory_by_id(a["id"]), db.get_memory_by_id(b["id"])
    assert ra["retrieval_count"] == 1 and ra["last_used"] and ra["last_accessed_at"]
    assert rb["temperature"] == "HOT" and rb["retrieval_count"] == 1


class _FakeSearcher:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, n_results=20, force_include_cold=False):
        return [dict(h) for h in self.hits]


def test_semantic_threshold_only_counts_real_matches():
    from caretaker.retrieval.retrieval_engine import retrieve_context
    strong, weak = mem("Orbit runs on FastAPI"), mem("Weather in Ahmedabad")
    hits = [{**strong, "semantic_similarity": 0.82}, {**weak, "semantic_similarity": 0.51}]
    result = retrieve_context("what does Orbit run on?", semantic_searcher=_FakeSearcher(hits))
    assert [m["id"] for m in result["relevant"]] == [strong["id"]]
    assert db.get_memory_by_id(strong["id"])["retrieval_count"] == 1
    assert db.get_memory_by_id(weak["id"])["retrieval_count"] == 0


def test_no_semantic_match_falls_back_to_keywords_then_background():
    from caretaker.retrieval.retrieval_engine import retrieve_context
    kw = mem("Orbit deploys with Docker")
    weak_only = [{**kw, "semantic_similarity": 0.4}]
    result = retrieve_context("orbit docker", semantic_searcher=_FakeSearcher(weak_only))
    assert [m["id"] for m in result["relevant"]] == [kw["id"]]          # found by keywords
    assert db.get_memory_by_id(kw["id"])["retrieval_count"] == 1

    other = mem("I like tea")
    result = retrieve_context("zzz unrelated question")
    assert result["relevant"]                                          # background context…
    assert db.get_memory_by_id(other["id"])["retrieval_count"] == 0    # …not counted as recalled


def test_min_relevance_is_configurable(monkeypatch):
    from caretaker.retrieval import retrieval_engine
    monkeypatch.setattr("caretaker.paths.load_config", lambda: {"min_relevance": 0.3})
    assert retrieval_engine._min_relevance() == 0.3


# ── Replacement and dedup ──────────────────────────────────────────────────────

def test_replacement_works_on_real_rows_and_stays_in_project():
    from caretaker.memory.conflict_checker import full_conflict_pipeline
    old = mem("Orbit database is MySQL", keywords=["orbit", "database"], project="orbit")
    other_project = mem("CRM database is MySQL", keywords=["orbit", "database"], project="crm")
    new_id = str(uuid.uuid4())
    full_conflict_pipeline(
        new_memory={"memory_id": new_id, "memory_type": "PROJECT", "keywords": ["orbit", "database"],
                    "full_text": "Orbit database is PostgreSQL", "project": "orbit"},
        local_db=db,
    )
    replaced = db.get_memory_by_id(old["id"])
    assert replaced["status"] == "OUTDATED" and replaced["superseded_by"] == new_id
    assert db.get_memory_by_id(other_project["id"])["status"] == "ACTIVE"


def _run_dedup(config):
    from caretaker.scheduler.nightly_maintenance import NightlyMaintenance
    return asyncio.run(NightlyMaintenance(local_db=db, vector_db=None, config=config)._task_deduplication())


def test_dedup_merges_recent_neighbours_within_a_project():
    kws = ["orbit", "fastapi", "postgres"]
    older = mem("Orbit uses FastAPI and Postgres", keywords=kws, when=datetime.now(timezone.utc) - timedelta(hours=5))
    newer = mem("Orbit is FastAPI + Postgres", keywords=kws)
    crm = mem("CRM uses FastAPI and Postgres", keywords=kws, project="crm")
    assert _run_dedup({}) == 1
    assert db.get_memory_by_id(older["id"])["status"] == "OUTDATED"
    assert db.get_memory_by_id(older["id"])["superseded_by"] == newer["id"]
    assert db.get_memory_by_id(crm["id"])["status"] == "ACTIVE"


def test_dedup_is_incremental_unless_lookback_is_zero():
    kws = ["nimbus", "weather", "react"]
    long_ago = datetime.now(timezone.utc) - timedelta(days=30)
    a = mem("Nimbus weather app in React", keywords=kws, when=long_ago)
    mem("Nimbus is a React weather app", keywords=kws, when=long_ago - timedelta(days=1))
    assert _run_dedup({}) == 0                              # nothing changed recently
    assert _run_dedup({"dedup_lookback_days": 0}) == 1      # full sweep
    assert db.get_memory_by_id(a["id"])["status"] == "ACTIVE"


# ── CLI never writes to ChromaDB ───────────────────────────────────────────────

def test_cli_delete_and_restore_do_not_open_vector_db(monkeypatch):
    from click.testing import CliRunner
    from caretaker.cli.main import cli
    import caretaker.storage.vector_db as vdb

    class Forbidden:
        def __init__(self, *a, **k):
            raise AssertionError("CLI must not open ChromaDB")

    monkeypatch.setattr(vdb, "VectorDB", Forbidden)
    m = mem("Orbit note", short="Orbit note")
    r = CliRunner().invoke(cli, ["delete", m["id"][:8], "--force"])
    assert r.exit_code == 0 and r.exception is None, r.output
    assert db.get_memory_by_id(m["id"])["status"] == "ARCHIVED"
    r = CliRunner().invoke(cli, ["restore", m["id"][:8]])
    assert r.exit_code == 0 and r.exception is None, r.output
    assert db.get_memory_by_id(m["id"])["status"] == "ACTIVE"
