"""
tests/phase3/test_doctor.py
`caretaker doctor` must describe what leaves the device the same way the code
that sends data behaves (compressor and cloud sync).
"""

import json

import pytest

import caretaker.storage.local_db as db

KEY = "sk-ant-" + "api03-" + "Qw1Er2Ty3Ui4Op5As6Df7Gh8Jk9Lz0Xc1"


@pytest.fixture
def checks(tmp_path, monkeypatch):
    monkeypatch.setenv("CARETAKER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CARETAKER_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "doctor.db"))
    db.run_migrations()
    cfg = tmp_path / "config.json"
    monkeypatch.setenv("CARETAKER_CONFIG", str(cfg))

    def run(config):
        cfg.write_text(json.dumps(config), encoding="utf-8")
        from caretaker.doctor import run_checks
        return {c["id"]: c for c in run_checks()}
    return run


def test_defaults_keep_everything_local(checks):
    c = checks({})
    assert c["compression"]["status"] == "ok"
    assert c["cloud"]["status"] == "ok" and c["cloud"]["detail"].startswith("Off")


def test_cloud_model_without_key_is_still_local(checks):
    c = checks({"compression_model": "claude-haiku-4-5"})
    assert c["compression"]["status"] == "ok"
    assert "no Anthropic API key" in c["compression"]["detail"]


def test_cloud_model_with_key_warns(checks):
    c = checks({"compression_model": "claude-haiku-4-5", "anthropic_api_key": KEY})
    assert c["compression"]["status"] == "warn"


def test_supabase_without_encryption_key_is_off(checks):
    c = checks({"supabase_url": "https://x.supabase.co", "supabase_key": "k"})
    assert c["cloud"]["status"] == "warn"
    assert "encrypt_key is missing" in c["cloud"]["detail"]
    c = checks({"supabase_url": "https://x.supabase.co", "supabase_key": "k", "encrypt_key": "e"})
    assert c["cloud"]["status"] == "ok" and "configured" in c["cloud"]["detail"]
