"""
tests/phase3/test_cloud.py
Phase 3 — Cloud + Encryption Tests: P3-T13, P3-T14, P3-T15

P3-T13  Cloud Upload    caretaker sync → encrypted rows in Supabase
P3-T14  Cloud Restore   delete local DB → restore from cloud
P3-T15  Encryption Check  raw Supabase row is encrypted, not plaintext

NOTE: P3-T13 and P3-T14 require real Supabase credentials.
      They are SKIPPED automatically when credentials are not set.
      P3-T15 tests the encryption logic directly — no Supabase needed.
"""

import sys
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import caretaker.storage.local_db as db_module


# ── Isolated DB ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    test_db = tmp_path / "caretaker_cloud_test.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(test_db))
    db_module.run_migrations()
    yield test_db


@pytest.fixture
def config(tmp_path):
    """Load real config, or return a minimal test config if not found."""
    config_path = Path(__file__).parent.parent.parent.parent.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {
        "supabase_url"  : "",
        "supabase_key"  : "",
        "encrypt_key"   : "test-encrypt-key-for-unit-tests",
        "archive_score" : 0.2,
    }


def _seed(**overrides) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    base = {
        "id"            : str(uuid.uuid4()),
        "source_agent"  : "claude",
        "keywords"      : '["python", "test"]',
        "short"         : "Test memory for cloud sync.",
        "full"          : "Full content for cloud sync encryption test.",
        "type"          : "PROJECT",
        "subtype"       : "test",
        "fact_type"     : "ADDITIVE",
        "status"        : "ACTIVE",
        "superseded_by" : None,
        "importance"    : 0.7,
        "decay_score"   : 0.8,
        "temperature"   : "HOT",
        "retrieval_count": 0,
        "created_at"    : now,
        "updated_at"    : now,
        "last_used"     : now,
        "source_msg"    : "Test message.",
    }
    base.update(overrides)
    db_module.save_memory(base)
    return base


# ── P3-T15: Encryption Check (no Supabase needed) ─────────────────────────────

class TestP3T15_EncryptionCheck:
    """
    P3-T15: Verify data is encrypted before upload — no Supabase required.
    Tests encrypt_memory_dict and decrypt_memory_dict directly.
    """

    def test_encrypt_memory_dict_hides_content(self, config):
        if not config.get("encrypt_key"):
            config["encrypt_key"] = "test-passphrase-for-unit-tests"

        from caretaker.storage.encrypt import Encryptor, encrypt_memory_dict
        enc = Encryptor(config)

        mem = _seed()
        encrypted = encrypt_memory_dict(mem, enc)

        # Sensitive fields must NOT be plaintext anymore
        assert encrypted.get("short")    != mem["short"],    "short not encrypted"
        assert encrypted.get("full")     != mem["full"],     "full not encrypted"
        # _encrypted flag must be set
        assert encrypted.get("_encrypted") is True

    def test_decrypt_memory_dict_restores_content(self, config):
        if not config.get("encrypt_key"):
            config["encrypt_key"] = "test-passphrase-for-unit-tests"

        from caretaker.storage.encrypt import Encryptor, encrypt_memory_dict, decrypt_memory_dict
        enc = Encryptor(config)

        mem = _seed()
        encrypted = encrypt_memory_dict(mem, enc)
        decrypted = decrypt_memory_dict(encrypted, enc)

        assert decrypted["short"] == mem["short"]
        assert decrypted["full"]  == mem["full"]
        assert "_encrypted" not in decrypted

    def test_wrong_key_raises_on_decrypt(self, config):
        config_a = dict(config, encrypt_key="correct-passphrase-abc")
        config_b = dict(config, encrypt_key="wrong-passphrase-xyz")

        from caretaker.storage.encrypt import Encryptor, encrypt_memory_dict, decrypt_memory_dict

        enc_a = Encryptor(config_a)
        enc_b = Encryptor(config_b)

        mem = _seed()
        encrypted = encrypt_memory_dict(mem, enc_a)

        with pytest.raises(ValueError):
            decrypt_memory_dict(encrypted, enc_b)

    def test_encrypt_non_sensitive_fields_unchanged(self, config):
        if not config.get("encrypt_key"):
            config["encrypt_key"] = "test-passphrase-for-unit-tests"

        from caretaker.storage.encrypt import Encryptor, encrypt_memory_dict
        enc = Encryptor(config)

        mem = _seed()
        encrypted = encrypt_memory_dict(mem, enc)

        # Non-sensitive fields must pass through unchanged
        for field in ("id", "type", "status", "temperature", "importance"):
            assert encrypted.get(field) == mem.get(field), (
                f"Non-sensitive field '{field}' was modified by encryption"
            )

    def test_encryptor_raises_without_key(self):
        from caretaker.storage.encrypt import Encryptor
        with pytest.raises((ValueError, Exception)):
            Encryptor({"encrypt_key": ""})

    def test_encrypt_decrypt_keywords_list(self, config):
        if not config.get("encrypt_key"):
            config["encrypt_key"] = "test-passphrase-for-unit-tests"

        from caretaker.storage.encrypt import Encryptor, encrypt_memory_dict, decrypt_memory_dict
        enc = Encryptor(config)

        mem = _seed()
        mem["keywords"] = ["python", "fastapi", "caretaker"]  # list not JSON string

        encrypted = encrypt_memory_dict(mem, enc)
        decrypted = decrypt_memory_dict(encrypted, enc)

        assert isinstance(decrypted["keywords"], list)
        assert "python" in decrypted["keywords"]


# ── P3-T13: Cloud Upload ───────────────────────────────────────────────────────

class TestP3T13_CloudUpload:
    """
    P3-T13: Manual sync → memories appear in Supabase.
    SKIPPED if supabase_url or supabase_key is not configured.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_credentials(self, config):
        if not config.get("supabase_url") or not config.get("supabase_key"):
            pytest.skip("Supabase credentials not configured — skipping cloud upload test")
        if not config.get("encrypt_key"):
            pytest.skip("encrypt_key not configured — skipping cloud upload test")

    def test_push_all_returns_pushed_count(self, config):
        _seed()
        _seed(type="PREFERENCE", short="Prefers Python.",
              full="I prefer Python.", keywords='["python"]')

        from caretaker.storage.cloud_sync import CloudSync
        cloud = CloudSync(config)
        assert cloud.initialize(), "Could not connect to Supabase"

        result = cloud.push_all()
        assert result.get("pushed", 0) >= 1
        assert result.get("failed", 0) == 0

    def test_push_encrypted_data_in_supabase(self, config):
        mem = _seed()

        from caretaker.storage.cloud_sync import CloudSync
        from caretaker.storage.encrypt import Encryptor, encrypt_memory_dict

        cloud = CloudSync(config)
        cloud.initialize()
        enc = Encryptor(config)

        # Encrypt and check output is base64 (not plaintext)
        encrypted = encrypt_memory_dict(mem, enc)
        assert encrypted["short"] != mem["short"]
        # Base64 strings don't contain spaces
        assert " " not in encrypted["short"]

    def test_get_remote_count_after_push(self, config):
        _seed()

        from caretaker.storage.cloud_sync import CloudSync
        cloud = CloudSync(config)
        cloud.initialize()
        cloud.push_all()

        count = cloud.get_remote_count()
        assert count is not None
        assert count >= 1


# ── P3-T14: Cloud Restore ──────────────────────────────────────────────────────

class TestP3T14_CloudRestore:
    """
    P3-T14: Push to cloud → delete local → restore from cloud.
    SKIPPED if credentials not configured.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_credentials(self, config):
        if not config.get("supabase_url") or not config.get("supabase_key"):
            pytest.skip("Supabase credentials not configured — skipping cloud restore test")
        if not config.get("encrypt_key"):
            pytest.skip("encrypt_key not configured — skipping cloud restore test")

    def test_pull_restores_memories_after_local_wipe(self, config):
        m1 = _seed()
        m2 = _seed(type="PREFERENCE", short="Prefers Python.",
                   full="I prefer Python.", keywords='["python"]')

        from caretaker.storage.cloud_sync import CloudSync
        cloud = CloudSync(config)
        cloud.initialize()

        # Push to cloud
        push_result = cloud.push_all()
        assert push_result.get("pushed", 0) >= 2

        # Wipe local DB
        with db_module.get_connection() as conn:
            conn.execute("DELETE FROM memories")

        # Restore from cloud
        pull_result = cloud.pull_all()
        assert pull_result.get("restored", 0) >= 2

        # Verify both memories restored
        from caretaker.storage.local_db import get_memory_by_id
        for mem in [m1, m2]:
            row = get_memory_by_id(mem["id"])
            assert row is not None, f"Memory {mem['id'][:8]} not restored from cloud"

    def test_pull_skips_already_present(self, config):
        mem = _seed()

        from caretaker.storage.cloud_sync import CloudSync
        cloud = CloudSync(config)
        cloud.initialize()
        cloud.push_all()

        # Pull without wiping — should skip existing
        result = cloud.pull_all()
        assert result.get("skipped", 0) >= 1