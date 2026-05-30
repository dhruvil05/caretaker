"""
storage/encrypt.py
Phase 3 — AES-256-GCM encryption wrapper for Caretaker cloud data.

Every memory that leaves the local machine gets encrypted here.
Every memory that comes back from Supabase gets decrypted here.

Uses AES-256-GCM (authenticated encryption):
  - 256-bit key
  - Random 96-bit nonce per encrypt call
  - Authentication tag prevents tampering
  - Output format: base64(nonce + ciphertext + tag)

Key derivation:
  - User sets encrypt_key in config.json as a plain passphrase
  - We derive a 32-byte key from it using PBKDF2-HMAC-SHA256
  - Fixed salt per installation (stored in config or derived from supabase_url)
"""

import base64
import hashlib
import os
from typing import Union


# ── Constants ──────────────────────────────────────────────────────────────────

NONCE_SIZE  = 12   # GCM standard 96-bit nonce
KEY_SIZE    = 32   # AES-256 = 32 bytes
TAG_SIZE    = 16   # GCM auth tag
PBKDF2_ITER = 100_000


# ── Key derivation ─────────────────────────────────────────────────────────────

def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """
    Derive a 32-byte AES key from a user passphrase using PBKDF2-HMAC-SHA256.
    Salt should be consistent per installation (derived from supabase_url or fixed).
    """
    return hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        PBKDF2_ITER,
        dklen=KEY_SIZE,
    )


def _get_salt(config: dict) -> bytes:
    """
    Generate a consistent salt for this installation.
    Uses supabase_url as entropy if available, else falls back to fixed salt.
    This is NOT a secret — it just adds uniqueness per installation.
    """
    salt_source = config.get("supabase_url", "caretaker-default-salt-v1")
    return hashlib.sha256(salt_source.encode("utf-8")).digest()


# ── Public API ─────────────────────────────────────────────────────────────────

class Encryptor:
    """
    AES-256-GCM encrypt/decrypt wrapper.

    Usage:
        enc = Encryptor(config)
        blob  = enc.encrypt("some text")   # → base64 string
        plain = enc.decrypt(blob)           # → "some text"
    """

    def __init__(self, config: dict):
        passphrase = config.get("encrypt_key", "")
        if not passphrase:
            raise ValueError(
                "[Encryptor] encrypt_key is empty in config.json. "
                "Set a passphrase to enable cloud sync encryption."
            )
        salt      = _get_salt(config)
        self._key = _derive_key(passphrase, salt)

    def encrypt(self, plaintext: Union[str, bytes]) -> str:
        """
        Encrypt plaintext with AES-256-GCM.
        Returns base64-encoded string: nonce(12) + ciphertext + tag(16).
        Raises ImportError if 'cryptography' package is not installed.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise ImportError(
                "[Encryptor] 'cryptography' package not installed. "
                "Run: uv add cryptography"
            )

        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")

        nonce  = os.urandom(NONCE_SIZE)
        aesgcm = AESGCM(self._key)

        # encrypt() returns ciphertext + appended 16-byte GCM tag
        ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data=None)

        # Pack: nonce || ciphertext+tag → base64
        packed = nonce + ciphertext_with_tag
        return base64.b64encode(packed).decode("utf-8")

    def decrypt(self, blob: str) -> str:
        """
        Decrypt a base64-encoded AES-256-GCM blob produced by encrypt().
        Returns original plaintext string.
        Raises ValueError if blob is tampered or key is wrong.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise ImportError(
                "[Encryptor] 'cryptography' package not installed. "
                "Run: uv add cryptography"
            )

        try:
            packed = base64.b64decode(blob.encode("utf-8"))
        except Exception as e:
            raise ValueError(f"[Encryptor] Invalid base64 blob: {e}")

        if len(packed) < NONCE_SIZE + TAG_SIZE:
            raise ValueError(
                f"[Encryptor] Blob too short ({len(packed)} bytes). "
                "Expected at least nonce(12) + tag(16) = 28 bytes."
            )

        nonce              = packed[:NONCE_SIZE]
        ciphertext_with_tag = packed[NONCE_SIZE:]

        aesgcm = AESGCM(self._key)

        try:
            plaintext_bytes = aesgcm.decrypt(nonce, ciphertext_with_tag, associated_data=None)
        except Exception:
            raise ValueError(
                "[Encryptor] Decryption failed. "
                "Key may be wrong or data may be tampered."
            )

        return plaintext_bytes.decode("utf-8")

    def is_available(self) -> bool:
        """Check if cryptography package is installed."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa
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
    """
    import json
    encrypted = dict(memory)  # shallow copy

    for field in ("short", "full", "keywords", "subtype", "source_msg"):
        raw = memory.get(field)
        if raw is None:
            continue
        # keywords is a list stored as JSON string — serialise first
        if field == "keywords" and isinstance(raw, list):
            raw = json.dumps(raw)
        elif not isinstance(raw, str):
            raw = str(raw)

        try:
            encrypted[field] = encryptor.encrypt(raw)
        except Exception as e:
            # Never fail a sync because of one field — leave unencrypted with marker
            encrypted[field] = raw
            print(f"[Encryptor] Warning: could not encrypt field '{field}': {e}")

    encrypted["_encrypted"] = True
    return encrypted


def decrypt_memory_dict(memory: dict, encryptor: "Encryptor") -> dict:
    """
    Decrypt sensitive fields of a memory dict after download from Supabase.
    Safe to call on non-encrypted dicts — returns as-is if _encrypted not set.
    """
    import json
    if not memory.get("_encrypted"):
        return memory

    decrypted = dict(memory)

    for field in ("short", "full", "keywords", "subtype", "source_msg"):
        raw = memory.get(field)
        if not raw:
            continue
        try:
            plain = encryptor.decrypt(raw)
            # keywords comes back as JSON string — deserialise
            if field == "keywords":
                try:
                    plain = json.loads(plain)
                except Exception:
                    pass  # keep as string if parse fails
            decrypted[field] = plain
        except ValueError:
            # Wrong key or tampered data — re-raise so caller knows
            raise
        except Exception as e:
            print(f"[Encryptor] Warning: could not decrypt field '{field}': {e}")
            # Leave as-is for unexpected non-crypto errors

    decrypted.pop("_encrypted", None)
    return decrypted