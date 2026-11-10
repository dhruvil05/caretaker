"""
tests/phase3/test_dashboard.py
Local dashboard: access control, security headers, API behaviour, and the
SQLite-is-source-of-truth guarantees it relies on.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from starlette.testclient import TestClient

import caretaker.storage.local_db as db_module

TOKEN = "test-token-123"


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "dashboard_test.db"))
    db_module.run_migrations()


@pytest.fixture
def client():
    from caretaker.dashboard.app import create_app
    return TestClient(create_app(TOKEN, allowed_hosts=["testserver"]))


def auth(**extra):
    return {"X-Caretaker-Token": TOKEN, **extra}


def seed(**overrides) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    mem = {
        "id": str(uuid.uuid4()), "source_agent": "claude", "keywords": '["python"]',
        "short": "Builds Caretaker.", "full": "I am building Caretaker, a memory layer.",
        "type": "PROJECT", "subtype": None, "fact_type": "REPLACEABLE", "status": "ACTIVE",
        "superseded_by": None, "importance": 0.6, "decay_score": 1.0, "temperature": "HOT",
        "retrieval_count": 0, "created_at": now, "updated_at": now, "last_used": None,
    }
    mem.update(overrides)
    db_module.save_memory(mem)
    return mem


# ── Access control & headers ───────────────────────────────────────────────────

class TestAccess:

    def test_api_requires_token(self, client):
        assert client.get("/api/overview").status_code == 401
        assert client.get("/api/overview", headers={"X-Caretaker-Token": "wrong"}).status_code == 401
        assert client.get("/api/overview", headers=auth()).status_code == 200

    def test_writes_require_token(self, client):
        mem = seed()
        assert client.post(f"/api/memories/{mem['id']}/archive").status_code == 401
        assert db_module.get_memory_by_id(mem["id"])["status"] == "ACTIVE"

    def test_token_in_query_string_is_not_accepted(self, client):
        assert client.get(f"/api/overview?token={TOKEN}").status_code == 401

    def test_foreign_host_header_is_rejected(self, client):
        # DNS-rebinding: a page on evil.example resolving to 127.0.0.1
        r = client.get("/api/overview", headers=auth(host="evil.example"))
        assert r.status_code == 400

    def test_page_loads_without_token_and_has_strict_csp(self, client):
        r = client.get("/")
        assert r.status_code == 200
        csp = r.headers["content-security-policy"]
        assert "default-src 'none'" in csp and "script-src 'self'" in csp
        assert "unsafe-inline" not in csp
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["referrer-policy"] == "no-referrer"

    def test_only_known_static_files_are_served(self, client):
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/app.py").status_code == 404
        assert client.get("/static/..%2Fapp.py").status_code == 404


# ── Overview ───────────────────────────────────────────────────────────────────

class TestOverview:

    def test_counts_and_workflow(self, client):
        seed(type="PROJECT", retrieval_count=3, last_used=datetime.now(timezone.utc).isoformat())
        seed(type="PREFERENCE", temperature="WARM", short=None)
        seed(type="PROJECT", status="ARCHIVED")
        seed(type="DECISION", status="OUTDATED")
        d = client.get("/api/overview", headers=auth()).json()

        assert d["totals"] == {"total": 4, "active": 2, "archived": 1, "outdated": 1,
                               "recalls": 3, "awaiting_compression": 1}
        assert {r["key"]: r["count"] for r in d["by_type"]} == {"PROJECT": 1, "PREFERENCE": 1}
        assert [r["key"] for r in d["by_temperature"]] == ["HOT", "WARM"]
        stages = {w["stage"]: w["count"] for w in d["workflow"]}
        assert stages["Captured"] == 4 and stages["Compressed"] == 1 and stages["Recalled"] == 1
        assert len(d["growth"]) == 30 and d["growth"][-1]["count"] == 4
        assert d["top_recalled"][0]["retrieval_count"] == 3

    def test_privacy_never_exposes_secrets(self, client, monkeypatch):
        monkeypatch.setenv("CARETAKER_ANTHROPIC_API_KEY", "sk-secret-value")
        monkeypatch.setenv("CARETAKER_ENCRYPT_KEY", "very secret passphrase")
        monkeypatch.setenv("CARETAKER_COMPRESSION_MODEL", "claude-haiku-4-5-20251001")
        r = client.get("/api/overview", headers=auth())
        assert "sk-secret-value" not in r.text and "very secret passphrase" not in r.text
        assert r.json()["privacy"]["compression_on_device"] is False


# ── Memories ───────────────────────────────────────────────────────────────────

class TestMemories:

    def test_filter_search_and_paging(self, client):
        seed(type="PROJECT", full="Working on Orbit, a FastAPI app")
        seed(type="PREFERENCE", full="Prefers Python over JavaScript")
        seed(type="PROJECT", status="ARCHIVED", full="Old project Nimbus")

        d = client.get("/api/memories", headers=auth(), params={"status": "ACTIVE"}).json()
        assert d["total"] == 2
        d = client.get("/api/memories", headers=auth(), params={"q": "orbit"}).json()
        assert d["total"] == 1 and "Orbit" in d["items"][0]["full"]
        d = client.get("/api/memories", headers=auth(), params={"type": "PROJECT", "status": "ARCHIVED"}).json()
        assert d["total"] == 1
        d = client.get("/api/memories", headers=auth(), params={"limit": 1, "offset": 1}).json()
        assert d["total"] == 3 and len(d["items"]) == 1
        assert set(d["facets"]) == {"type", "temperature", "status", "agent"}
        assert d["items"][0]["keywords"] == ["python"]

    def test_unknown_sort_and_bad_paging(self, client):
        seed()
        assert client.get("/api/memories", headers=auth(), params={"sort": "id; DROP TABLE memories"}).status_code == 200
        assert client.get("/api/memories", headers=auth(), params={"limit": "x"}).status_code == 400
        assert client.get("/api/memories", headers=auth(), params={"limit": 9999}).json()["limit"] == 200

    def test_edit_memory(self, client):
        mem = seed(importance=0.3, temperature="WARM")
        r = client.patch(f"/api/memories/{mem['id']}", headers=auth(),
                         json={"short": "Builds Orbit.", "full": "I am building Orbit.", "importance": 0.9})
        assert r.status_code == 200, r.text
        saved = db_module.get_memory_by_id(mem["id"])
        assert saved["short"] == "Builds Orbit." and saved["full"] == "I am building Orbit."
        assert saved["importance"] == 0.9
        from caretaker.memory.temperature_engine import assign_temperature
        assert saved["temperature"] == assign_temperature(0.9, "WARM")   # same rule as `caretaker score`
        assert saved["temperature"] != "WARM"

    @pytest.mark.parametrize("body", [
        {"importance": 2}, {"importance": "high"}, {"full": "   "}, {"short": 123},
        {"status": "ARCHIVED"}, {}, {"short": "x" * 20_001},
    ])
    def test_edit_rejects_bad_input(self, client, body):
        mem = seed()
        assert client.patch(f"/api/memories/{mem['id']}", headers=auth(), json=body).status_code == 400
        assert db_module.get_memory_by_id(mem["id"])["full"] == mem["full"]

    def test_archive_and_restore(self, client):
        mem = seed()
        assert client.post(f"/api/memories/{mem['id']}/archive", headers=auth()).json()["status"] == "ARCHIVED"
        assert client.post(f"/api/memories/{mem['id']}/restore", headers=auth()).json()["status"] == "ACTIVE"

    def test_unknown_memory_is_404(self, client):
        assert client.get("/api/memories/nope", headers=auth()).status_code == 404
        assert client.post("/api/memories/nope/archive", headers=auth()).status_code == 404


# ── Activity & maintenance log ─────────────────────────────────────────────────

def test_activity_and_maintenance_log(client, tmp_path, monkeypatch):
    import caretaker.paths as paths
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    (paths.logs_dir() / "maintenance.log").write_text(
        "\n" + "=" * 60 + "\n2026-11-10 02:00:01 UTC\n  Decayed          : 2\n  Reindexed        : 5\n  Elapsed          : 0.40s\n",
        encoding="utf-8",
    )
    seed(last_used=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(), retrieval_count=2)
    d = client.get("/api/activity", headers=auth()).json()
    assert len(d["recent_saves"]) == 1 and len(d["recently_recalled"]) == 1
    assert d["recent_saves"][0]["text"] == "Builds Caretaker."
    assert d["maintenance_runs"] == [{"time": "2026-11-10 02:00:01 UTC",
                                      "values": {"Decayed": "2", "Reindexed": "5", "Elapsed": "0.40s"}}]


# ── Source-of-truth guarantees ─────────────────────────────────────────────────

def test_semantic_search_skips_archived_memories():
    from caretaker.retrieval.semantic_searcher import SemanticSearcher
    active, archived = seed(full="active one"), seed(full="archived one", status="ARCHIVED")

    class FakeVectorDB:
        def search(self, query, n_results=10, temperature_filter=None):
            return [{"memory_id": m["id"], "short": m["short"], "distance": 0.1,
                     "temperature": "HOT", "memory_type": "PROJECT", "keywords": "", "importance_score": 0.5}
                    for m in (active, archived)]

        def count(self):
            return 2

    results = SemanticSearcher(vector_db=FakeVectorDB(), local_db=db_module).search("one", n_results=5)
    ids = {r.get("id") or r.get("memory_id") for r in results}
    assert active["id"] in ids and archived["id"] not in ids


def test_vector_db_add_and_update_map_onto_upsert(monkeypatch):
    from caretaker.storage.vector_db import VectorDB
    calls = []
    vdb = VectorDB.__new__(VectorDB)
    monkeypatch.setattr(vdb, "upsert", lambda **kw: calls.append(kw), raising=False)
    vdb.add(memory_id="m1", text="hello", metadata={"type": "PROJECT", "temperature": "HOT", "keywords": "a, b"})
    vdb.update(memory_id="m1", text="hello again", metadata={"type": "PROJECT"})
    assert calls[0] == {"memory_id": "m1", "short": "hello", "keywords": ["a", "b"],
                        "temperature": "HOT", "memory_type": "PROJECT", "importance_score": 0.5}
    assert calls[1]["short"] == "hello again" and calls[1]["temperature"] == "WARM"
