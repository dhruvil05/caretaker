"""
tests/phase3/test_dashboard_v05.py
0.5 dashboard API: project filter, preview (dry run), privacy page + actions +
policy, review (replaced/stale) with undo, bulk actions, setup.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from starlette.testclient import TestClient

import caretaker.storage.local_db as db

TOKEN = "tok"
KEY = "sk-ant-" + "api03-" + "Qw1Er2Ty3Ui4Op5As6Df7Gh8Jk9Lz0Xc1"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "dash05.db"))
    monkeypatch.setenv("CARETAKER_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("CARETAKER_DATA_DIR", str(tmp_path / "data"))
    db._FTS_CACHE.clear()
    db.run_migrations()


@pytest.fixture
def client():
    from caretaker.dashboard.app import create_app
    return TestClient(create_app(TOKEN, allowed_hosts=["testserver"]))


H = {"X-Caretaker-Token": TOKEN}


def mem(full, **kw):
    when = kw.pop("when", datetime.now(timezone.utc)).isoformat()
    m = {"id": str(uuid.uuid4()), "source_agent": "claude", "keywords": None, "short": None, "full": full,
         "type": "PROJECT", "subtype": None, "fact_type": "ADDITIVE", "status": "ACTIVE", "superseded_by": None,
         "importance": 0.6, "decay_score": 1.0, "temperature": "HOT", "retrieval_count": 0,
         "created_at": when, "updated_at": when, "last_used": None}
    m.update(kw)
    db.save_memory(m)
    return m


def test_all_new_endpoints_require_token(client):
    for method, path in [("get", "/api/projects"), ("get", "/api/preview?message=x"), ("get", "/api/privacy"),
                         ("post", "/api/privacy/scan"), ("put", "/api/privacy/policy"), ("get", "/api/review"),
                         ("post", "/api/memories/bulk"), ("get", "/api/setup")]:
        assert getattr(client, method)(path).status_code == 401, path


def test_projects_and_project_filter(client):
    mem("Orbit note", project="orbit")
    mem("CRM note", project="crm")
    mem("Global note")
    projects = {p["project"]: p["count"] for p in client.get("/api/projects", headers=H).json()}
    assert projects == {"orbit": 1, "crm": 1, "": 1}
    assert client.get("/api/overview?project=orbit", headers=H).json()["totals"]["active"] == 1
    assert client.get("/api/overview?project=__global__", headers=H).json()["totals"]["active"] == 1
    assert client.get("/api/overview", headers=H).json()["totals"]["active"] == 3
    items = client.get("/api/memories?project=crm", headers=H).json()["items"]
    assert [m["full"] for m in items] == ["CRM note"]
    acts = client.get("/api/activity?project=orbit", headers=H).json()["recent_saves"]
    assert [a["text"] for a in acts] == ["Orbit note"]


def test_preview_is_a_dry_run_and_redacts(client):
    m = mem(f"Orbit deploys with Docker using key {KEY}", project="orbit")
    r = client.get("/api/preview", params={"message": "how does orbit deploy with docker", "project": "orbit"}, headers=H)
    assert r.status_code == 200, r.text
    d = r.json()
    assert "Docker" in d["text"] and KEY not in d["text"]
    assert d["memories"][0]["id"] == m["id"] and "keyword match" in d["memories"][0]["reason"]
    assert db.get_memory_by_id(m["id"])["retrieval_count"] == 0          # not counted
    assert client.get("/api/preview", headers=H).status_code == 400


def test_privacy_page_scan_and_actions(client):
    secret = mem(f"deploy key {KEY}")
    email = mem("reach me at me@example.com")
    report = client.post("/api/privacy/scan", json={}, headers=H).json()
    assert report["with_findings"] == 2 and report["will_change"] == 1 and not report["applied"]
    assert KEY not in json.dumps(report)
    applied = client.post("/api/privacy/scan", json={"apply": True}, headers=H).json()
    assert applied["changed"] == 1 and KEY not in db.get_memory_by_id(secret["id"])["full"]

    page = client.get("/api/privacy", headers=H).json()
    assert page["counts"]["flagged"] == 1 and page["counts"]["redacted"] == 1
    assert [q["id"] for q in page["queue"]] == [email["id"]]
    assert page["policy"]["contact"] == "flag" and KEY not in json.dumps(page)

    r = client.post(f"/api/memories/{email['id']}/privacy", json={"action": "redact"}, headers=H).json()
    assert "me@example.com" not in r["full"] and r["sensitivity"] == "redacted"
    assert client.post(f"/api/memories/{email['id']}/privacy", json={"action": "nope"}, headers=H).status_code == 400


def test_privacy_policy_update_validates_and_saves(client, tmp_path):
    bad = client.put("/api/privacy/policy", json={"contact": "delete"}, headers=H)
    assert bad.status_code == 400
    bad_re = client.put("/api/privacy/policy", json={"custom_patterns": [{"pattern": "(["}]}, headers=H)
    assert bad_re.status_code == 400
    ok = client.put("/api/privacy/policy", headers=H, json={
        "contact": "redact", "custom_patterns": [{"pattern": r"ACME-\d+", "label": "client id"}],
        "allowlist": ["team@example.com"]})
    assert ok.status_code == 200 and ok.json()["policy"]["contact"] == "redact"
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["privacy"]
    assert saved["contact"] == "redact" and saved["allowlist"] == ["team@example.com"]


def test_review_replaced_with_undo_and_stale(client):
    new = mem("Orbit database is PostgreSQL")
    old = mem("Orbit database is MySQL", status="OUTDATED", superseded_by=new["id"])
    stale = mem("Old idea nobody used", when=datetime.now(timezone.utc) - timedelta(days=90))
    d = client.get("/api/review", headers=H).json()
    assert d["replaced"][0]["id"] == old["id"]
    assert d["replaced"][0]["replaced_by"]["text"] == "Orbit database is PostgreSQL"
    assert [s["id"] for s in d["stale"]] == [stale["id"]]
    undo = client.post(f"/api/memories/{old['id']}/restore", headers=H).json()
    assert undo["status"] == "ACTIVE" and undo["superseded_by"] is None


def test_bulk_actions(client):
    a, b = mem("a"), mem("b")
    r = client.post("/api/memories/bulk", json={"ids": [a["id"], b["id"], "missing"], "action": "set_project",
                                                "value": "Orbit App"}, headers=H).json()
    assert r == {"action": "set_project", "requested": 3, "updated": 2}
    assert db.get_memory_by_id(a["id"])["project"] == "orbit-app"
    client.post("/api/memories/bulk", json={"ids": [a["id"]], "action": "archive"}, headers=H)
    assert db.get_memory_by_id(a["id"])["status"] == "ARCHIVED"
    client.post("/api/memories/bulk", json={"ids": [b["id"]], "action": "set_type", "value": "DECISION"}, headers=H)
    assert db.get_memory_by_id(b["id"])["type"] == "DECISION"
    for body in [{"ids": [], "action": "archive"}, {"ids": [a["id"]], "action": "drop_table"},
                 {"ids": [a["id"]], "action": "set_type", "value": "EVIL"}]:
        assert client.post("/api/memories/bulk", json=body, headers=H).status_code == 400


def test_patch_project_and_type(client):
    m = mem("note")
    r = client.patch(f"/api/memories/{m['id']}", json={"project": "My CRM", "type": "DECISION"}, headers=H).json()
    assert r["project"] == "my-crm" and r["type"] == "DECISION"
    assert client.patch(f"/api/memories/{m['id']}", json={"type": "NOPE"}, headers=H).status_code == 400


def test_setup_lists_checks_and_snippets(client):
    d = client.get("/api/setup", headers=H).json()
    ids = {c["id"] for c in d["checks"]}
    assert {"data_dir", "database", "semantic", "compression", "clients"} <= ids
    apps = {s["app"] for s in d["snippets"]}
    assert {"Claude Desktop", "Cursor", "VS Code", "Claude Code"} <= apps
    claude = next(s for s in d["snippets"] if s["app"] == "Claude Desktop")
    assert '"caretaker"' in claude["json"]


def test_review_follows_project_filter(client):
    new = mem("Orbit database is PostgreSQL", project="orbit")
    mem("Orbit database is MySQL", project="orbit", status="OUTDATED", superseded_by=new["id"])
    old = datetime.now(timezone.utc) - timedelta(days=90)
    crm_stale = mem("CRM idea nobody used", project="crm", when=old)
    d = client.get("/api/review?project=crm", headers=H).json()
    assert d["replaced"] == [] and [s["id"] for s in d["stale"]] == [crm_stale["id"]]
    assert len(client.get("/api/review?project=orbit", headers=H).json()["replaced"]) == 1


def test_summary_counts_for_navigation(client):
    new = mem("new fact")
    mem("old fact", status="OUTDATED", superseded_by=new["id"])
    mem("contact me@example.com", sensitivity="flagged")
    mem("unused", when=datetime.now(timezone.utc) - timedelta(days=90))
    assert client.get("/api/summary").status_code == 401
    body = client.get("/api/summary", headers=H).json()
    assert {k: body[k] for k in ("privacy_review", "replaced", "stale")} == {"privacy_review": 1, "replaced": 1, "stale": 1}
    assert body["data_flow"]["compression_on_device"] is True
    assert body["data_flow"]["cloud_backup"] is False


def test_data_flow_matches_what_the_compressor_does():
    """The sidebar must not claim 'on this device' wrongly, nor warn when nothing is sent."""
    from caretaker.dashboard.api_v05 import data_flow
    assert data_flow({})["compression_on_device"] is True
    # A cloud model without a key falls back to local summaries: nothing is sent.
    assert data_flow({"compression_model": "claude-haiku"})["compression_on_device"] is True
    assert data_flow({"compression_model": "claude-haiku", "anthropic_api_key": KEY})["compression_on_device"] is False
    assert data_flow({"compression_model": "local", "anthropic_api_key": KEY})["compression_on_device"] is True
    assert data_flow({"supabase_url": "u", "supabase_key": "k"})["cloud_backup"] is False
    assert data_flow({"supabase_url": "u", "supabase_key": "k", "encrypt_key": "e"})["cloud_backup"] is True
