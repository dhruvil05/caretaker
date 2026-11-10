"""
caretaker/paths.py
Single source of truth for where Caretaker keeps its config and files.

Everything stays on the user's own device. Caretaker writes into ONE data
directory:

    <data_dir>/
        data/caretaker.db        SQLite memory store (source of truth)
        data/chromadb/           ChromaDB vector index
        logs/                    caretaker.log (MCP server), maintenance.log
        exports/                 default location for `caretaker export`
        cache/                   downloaded models (Hugging Face / sentence-transformers / torch)

Default locations
-----------------
Installed package (pip / uv / uvx):
    config file  ~/.caretaker/config.json
    data_dir     ~/.caretaker
Source checkout (developing Caretaker itself — pyproject.toml + src/ present):
    config file  <project>/config.json
    data_dir     <project>/public

Overrides (first match wins)
----------------------------
config file:  CARETAKER_CONFIG env var → default above
data_dir:     CARETAKER_DATA_DIR env var → "data_dir" in config → default above
Relative paths in the config are resolved against the config file's folder.

Settings from the environment
-----------------------------
MCP clients pass settings through the `env` block of their server config, so
every setting can also come from an environment variable (overrides the file):
    CARETAKER_USER_HANDLE, CARETAKER_COMPRESSION_MODEL, CARETAKER_ANTHROPIC_API_KEY,
    CARETAKER_SUPABASE_URL, CARETAKER_SUPABASE_KEY, CARETAKER_ENCRYPT_KEY,
    CARETAKER_MAINTENANCE_TIME, CARETAKER_SCHEDULER_ENABLED
ANTHROPIC_API_KEY is used as a fallback for anthropic_api_key.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# src/caretaker/paths.py → parents[0]=caretaker, [1]=src, [2]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IS_SOURCE_CHECKOUT = (
    (PROJECT_ROOT / "pyproject.toml").is_file()
    and (PROJECT_ROOT / "src" / "caretaker" / "paths.py").is_file()
)
USER_HOME_DIR = Path.home() / ".caretaker"

CONFIG_ENV   = "CARETAKER_CONFIG"
DATA_DIR_ENV = "CARETAKER_DATA_DIR"

# env var → config key. Values from the environment override config.json.
ENV_SETTINGS = {
    "CARETAKER_USER_HANDLE": "user_handle",
    "CARETAKER_COMPRESSION_MODEL": "compression_model",
    "CARETAKER_ANTHROPIC_API_KEY": "anthropic_api_key",
    "CARETAKER_SUPABASE_URL": "supabase_url",
    "CARETAKER_SUPABASE_KEY": "supabase_key",
    "CARETAKER_ENCRYPT_KEY": "encrypt_key",
    "CARETAKER_MAINTENANCE_TIME": "maintenance_time",
    "CARETAKER_SCHEDULER_ENABLED": "scheduler_enabled",
}
_BOOL_SETTINGS = {"scheduler_enabled"}


def _expand(path_str: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path_str)))


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Config ─────────────────────────────────────────────────────────────────────

def config_path() -> Path:
    """Location of config.json (may not exist yet)."""
    env_value = os.environ.get(CONFIG_ENV, "").strip()
    if env_value:
        return _expand(env_value).resolve()
    if IS_SOURCE_CHECKOUT:
        return PROJECT_ROOT / "config.json"
    return USER_HOME_DIR / "config.json"


def read_config_file() -> dict:
    """Only what is in config.json (no environment overrides). {} if missing."""
    path = config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"[paths] Could not read {path}: {e}")
        return {}


def write_config_file(config: dict) -> Path:
    """Write config.json, creating its folder if needed."""
    path = config_path()
    _ensure(path.parent)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_config() -> dict:
    """Effective configuration: config.json, then environment overrides."""
    config = read_config_file()
    for env_name, key in ENV_SETTINGS.items():
        value = os.environ.get(env_name, "").strip()
        if not value:
            continue
        if key in _BOOL_SETTINGS:
            config[key] = value.lower() in ("1", "true", "yes", "on")
        else:
            config[key] = value
    if not config.get("anthropic_api_key") and os.environ.get("ANTHROPIC_API_KEY"):
        config["anthropic_api_key"] = os.environ["ANTHROPIC_API_KEY"]
    return config


# ── Data directory ─────────────────────────────────────────────────────────────

def default_data_dir() -> Path:
    return PROJECT_ROOT / "public" if IS_SOURCE_CHECKOUT else USER_HOME_DIR


def data_dir(config: Optional[dict] = None) -> Path:
    """Root folder for everything Caretaker stores. Created if missing."""
    env_value = os.environ.get(DATA_DIR_ENV, "").strip()
    if env_value:
        return _ensure(_expand(env_value).resolve())

    if config is None:
        config = read_config_file()
    configured = str(config.get("data_dir") or "").strip()
    if configured:
        path = _expand(configured)
        if not path.is_absolute():
            path = config_path().parent / path
        return _ensure(path.resolve())

    return _ensure(default_data_dir())


def db_path(config: Optional[dict] = None) -> Path:
    return _ensure(data_dir(config) / "data") / "caretaker.db"


def chromadb_dir(config: Optional[dict] = None) -> Path:
    if config is None:
        config = read_config_file()
    if (config.get("database") or {}).get("chromadb_path"):
        logger.warning(
            "[paths] config.json 'database.chromadb_path' is no longer used — "
            "ChromaDB lives in <data_dir>/data/chromadb. Set 'data_dir' instead."
        )
    return _ensure(data_dir(config) / "data" / "chromadb")


def logs_dir(config: Optional[dict] = None) -> Path:
    return _ensure(data_dir(config) / "logs")


def exports_dir(config: Optional[dict] = None) -> Path:
    return _ensure(data_dir(config) / "exports")


def cache_dir(config: Optional[dict] = None) -> Path:
    return _ensure(data_dir(config) / "cache")


def configure_cache_env(config: Optional[dict] = None) -> None:
    """
    Point model caches at <data_dir>/cache. Must run BEFORE huggingface_hub /
    sentence_transformers / torch are imported. Values the user already set
    in their environment are left alone.
    """
    cache = cache_dir(config)
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache / "sentence_transformers"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
