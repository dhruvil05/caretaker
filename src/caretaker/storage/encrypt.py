"""
storage/encrypt.py
Phase 3 — AES-256-GCM encryption wrapper for Caretaker cloud data.

Every memory that leaves the local machine gets encrypted here.
Every memory that comes back from Supabase gets decrypted here.

Uses AES-256-GCM (authenticated encryption):
  - 256-bit key
  - Random 96-bit nonce per encrypt call
  - Authentication tag prevents tampering
  - Associated data binds each ciphertext to its memory id + field name,
    so ciphertext can't be moved to another record or field unnoticed

Blob formats:
  - v2 (current):  "v2:" + base64(nonce + ciphertext + tag)
                   key = PBKDF2-HMAC-SHA256, 600,000 iterations,
                   associated data = "caretaker|<memory id>|<field>"
  - v1 (legacy):   base64(nonce + ciphertext + tag), no prefix
                   key = PBKDF2-HMAC-SHA256, 100,000 iterations, no associated data
                   Still decrypted so existing cloud backups keep working;
                   every push re-encrypts in v2.

Key derivation:
  - User sets encrypt_key in config.json as a plain passphrase
  - We derive a 32-byte key from it using PBKDF2-HMAC-SHA256
  - Salt = SHA-256 of supabase_url: unique per Supabase project, and
    recoverable on a new machine from config alone (salts need not be secret)
"""

import base64
import hashlib
import json
import logging
import os
from typing import Optional, Union

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

NONCE_SIZE  = 12   # GCM standard 96-bit nonce
KEY_SIZE    = 32   # AES-256 = 32 bytes
TAG_SIZE    = 16   # GCM auth tag

PBKDF2_ITER        = 600_000   # v2 — OWASP recommendation for PBKDF2-HMAC-SHA256
PBKDF2_ITER_LEGACY = 100_000   # v1 — only used to read old blobs
BLOB_PREFIX_V2     = "v2:"

MIN_PASSPHRASE_LEN = 12

ENCRYPTED_FIELDS = ("short", "full", "keywords", "subtype", "source_msg")


# ── Key derivation ─────────────────────────────────────────────────────────────

def _derive_key(passphrase: str, salt: bytes, iterations: int = PBKDF2_ITER) -> bytes:
    """Derive a 32-byte AES key from a user passphrase using PBKDF2-HMAC-SHA256."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        iterations,
        dklen=KEY_SIZE,
    )


def _get_salt(config: dict) -> bytes:
    """
    Consistent salt for this installation.
    Uses supabase_url (unique per Supabase project) if available, else a fixed salt.
    This is NOT a secret — it just makes keys differ between installations.
    """
    salt_source = config.get("supabase_url") or "caretaker-default-salt-v1"
    return hashlib.sha256(salt_source.encode("utf-8")).digest()


def field_aad(memory_id: str, field: str) -> bytes:
    """Associated data binding a ciphertext to one memory id + field."""
    return f"caretaker|{memory_id}|{field}".encode("utf-8")


def _load_aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise ImportError(
            "[Encryptor] 'cryptography' package not installed. "
            "Run: uv add cryptography"
        )
    return AESGCM


# ── Public API ─────────────────────────────────────────────────────────────────

class Encryptor:
    """
    AES-256-GCM encrypt/decrypt wrapper.

    Usage:
        enc = Encryptor(config)
        blob  = enc.encrypt("some text", aad=b"...")   # → "v2:..." string
        plain = enc.decrypt(blob, aad=b"...")           # → "some text"
    """

    def __init__(self, config: dict):
        passphrase = config.get("encrypt_key", "")
        if not passphrase:
            raise ValueError(
                "[Encryptor] encrypt_key is empty in config.json. "
                "Set a passphrase to enable cloud sync encryption."
            )
        if len(passphrase) < MIN_PASSPHRASE_LEN:
            logger.warning(
                f"[Encryptor] encrypt_key is shorter than {MIN_PASSPHRASE_LEN} characters. "
                "Anyone who can read your Supabase table could try to guess it offline — "
                "use a long passphrase (e.g. 4+ random words)."
            )
        salt = _get_salt(config)
        self._key        = _derive_key(passphrase, salt, PBKDF2_ITER)
        self._legacy_key = _derive_key(passphrase, salt, PBKDF2_ITER_LEGACY)

    def encrypt(self, plaintext: Union[str, bytes], aad: Optional[bytes] = None) -> str:
        """
        Encrypt plaintext with AES-256-GCM (v2 format).
        Returns "v2:" + base64(nonce(12) + ciphertext + tag(16)).
        """
        AESGCM = _load_aesgcm()
        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")

        nonce = os.urandom(NONCE_SIZE)
        ciphertext_with_tag = AESGCM(self._key).encrypt(nonce, plaintext, aad)
        return BLOB_PREFIX_V2 + base64.b64encode(nonce + ciphertext_with_tag).decode("utf-8")

    def decrypt(self, blob: str, aad: Optional[bytes] = None) -> str:
        """
        Decrypt a blob produced by encrypt() (v2) or by older versions (v1).
        `aad` must match what was used to encrypt a v2 blob; it is ignored for
        legacy v1 blobs, which were written without associated data.
        Raises ValueError if the blob is malformed, tampered, or the key is wrong.
        """
        AESGCM = _load_aesgcm()
        if not isinstance(blob, str):
            raise ValueError("[Encryptor] Encrypted value is not a string.")

        if blob.startswith(BLOB_PREFIX_V2):
            key, body, used_aad = self._key, blob[len(BLOB_PREFIX_V2):], aad
        else:
            key, body, used_aad = self._legacy_key, blob, None

        try:
            packed = base64.b64decode(body.encode("utf-8"), validate=True)
        except Exception as e:
            raise ValueError(f"[Encryptor] Invalid base64 blob: {e}")

        if len(packed) < NONCE_SIZE + TAG_SIZE:
            raise ValueError(
                f"[Encryptor] Blob too short ({len(packed)} bytes). "
                "Expected at least nonce(12) + tag(16) = 28 bytes."
            )

        nonce, ciphertext_with_tag = packed[:NONCE_SIZE], packed[NONCE_SIZE:]
        try:
            plaintext_bytes = AESGCM(key).decrypt(nonce, ciphertext_with_tag, used_aad)
        except Exception:
            raise ValueError(
                "[Encryptor] Decryption failed. "
                "Key may be wrong or data may be tampered."
            )

        try:
            return plaintext_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"[Encryptor] Decrypted data is not valid UTF-8: {e}")

    def is_available(self) -> bool:
        """Check if cryptography package is installed."""
        try:
            _load_aesgcm()
            return True
        except ImportError:
            return False


# ── Standalone helpers (used by cloud_sync without instantiating class) ────────

def encrypt_memory_dict(memory: dict, encryptor: "Encryptor") -> dict:
    """
    Encrypt sensitive fields of a memory dict before upload.
    Non-sensitive fields (id, type, status, temperature, timestamps) left as-is
    so Supabase can index/filter on them without decryption.

    Encrypted fields: short, full, keywords, subtype, source_msg

    Raises ValueError if any field cannot be encrypted: a memory is never
    uploaded with some fields left in plaintext.
    """
    memory_id = memory.get("id")
    if not memory_id:
        raise ValueError("[Encryptor] Memory has no id; refusing to encrypt.")

    encrypted = dict(memory)  # shallow copy

    for field in ENCRYPTED_FIELDS:
        raw = memory.get(field)
        if raw is None:
            continue
        # keywords is a list stored as JSON string — serialise first
        if field == "keywords" and isinstance(raw, list):
            raw = json.dumps(raw)
        elif not isinstance(raw, str):
            raw = str(raw)

        try:
            encrypted[field] = encryptor.encrypt(raw, aad=field_aad(memory_id, field))
        except Exception as e:
            raise ValueError(
                f"[Encryptor] Could not encrypt field '{field}' of memory "
                f"{str(memory_id)[:8]}: {e}. Memory will not be uploaded."
            ) from e

    encrypted["_encrypted"] = True
    return encrypted


def decrypt_memory_dict(memory: dict, encryptor: "Encryptor") -> dict:
    """
    Decrypt sensitive fields of a memory dict after download from Supabase.
    Returns the dict unchanged if `_encrypted` is not set — callers that
    receive data from the cloud must check `_encrypted` themselves first
    (cloud_sync.pull_all does) and refuse unencrypted records.

    Raises ValueError if any encrypted field fails to decrypt.
    """
    if not memory.get("_encrypted"):
        return memory

    memory_id = memory.get("id")
    if not memory_id:
        raise ValueError("[Encryptor] Encrypted memory has no id.")

    decrypted = dict(memory)

    for field in ENCRYPTED_FIELDS:
        raw = memory.get(field)
        if not raw:
            continue
        plain = encryptor.decrypt(raw, aad=field_aad(memory_id, field))
        # keywords comes back as JSON string — deserialise
        if field == "keywords":
            try:
                plain = json.loads(plain)
            except Exception:
                pass  # keep as string if parse fails
        decrypted[field] = plain

    decrypted.pop("_encrypted", None)
    return decrypted
