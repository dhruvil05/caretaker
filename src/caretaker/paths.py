"""
caretaker/paths.py
Single source of truth for where Caretaker keeps its files.

Everything Caretaker writes lives under ONE data directory:

    <data_dir>/
        data/caretaker.db        SQLite memory store (source of truth)
        data/chromadb/           ChromaDB vector index
        logs/                    caretaker.log (MCP server), maintenance.log
        exports/                 default location for `caretaker export`
        cache/                   downloaded models (Hugging Face / sentence-transformers / torch)

The data directory is chosen in this order (first match wins):
    1. CARETAKER_DATA_DIR environment variable
    2. "data_dir" in config.json
    3. <project root>/public                        (default)

Relative paths are resolved against the project root (the folder that holds
config.json), never against the current working directory, so the CLI and
the MCP server always agree no matter where they are started from.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# src/caretaker/paths.py → parents[0]=caretaker, [1]=src, [2]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "public"
DATA_DIR_ENV = "CARETAKER_DATA_DIR"


def load_config() -> dict:
    """Read config.json. Returns {} if it is missing or invalid."""
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"[paths] Could not read {CONFIG_PATH}: {e}")
        return {}


def _resolve(path_str: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(path_str)))
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(config: Optional[dict] = None) -> Path:
    """Root folder for everything Caretaker stores. Created if missing."""
    env_value = os.environ.get(DATA_DIR_ENV, "").strip()
    if env_value:
        return _ensure(_resolve(env_value))

    if config is None:
        config = load_config()
    configured = str(config.get("data_dir") or "").strip()
    if configured:
        return _ensure(_resolve(configured))

    return _ensure(DEFAULT_DATA_DIR)


def db_path(config: Optional[dict] = None) -> Path:
    return _ensure(data_dir(config) / "data") / "caretaker.db"


def chromadb_dir(config: Optional[dict] = None) -> Path:
    if config is None:
        config = load_config()
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
