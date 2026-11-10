"""
tests/phase3/test_paths.py
Config + data-directory resolution:
  - installed package → ~/.caretaker ; source checkout → <project>/public
  - CARETAKER_CONFIG / CARETAKER_DATA_DIR / data_dir overrides
  - settings from environment variables (how MCP clients pass config)
  - every file location lives under the chosen folder
"""

import json
import os

import pytest

import caretaker.paths as paths


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated environment: fake project root and fake user home."""
    project = tmp_path / "project"
    home = tmp_path / "home" / ".caretaker"
    project.mkdir()
    monkeypatch.setattr(paths, "PROJECT_ROOT", project)
    monkeypatch.setattr(paths, "USER_HOME_DIR", home)
    for var in [paths.CONFIG_ENV, paths.DATA_DIR_ENV, "ANTHROPIC_API_KEY", *paths.ENV_SETTINGS]:
        monkeypatch.delenv(var, raising=False)
    return {"project": project, "home": home, "mp": monkeypatch}


def _installed(env):
    env["mp"].setattr(paths, "IS_SOURCE_CHECKOUT", False)


def _checkout(env):
    env["mp"].setattr(paths, "IS_SOURCE_CHECKOUT", True)


# ── Defaults ───────────────────────────────────────────────────────────────────

def test_installed_defaults_to_user_home(env):
    _installed(env)
    assert paths.config_path() == env["home"] / "config.json"
    assert paths.data_dir() == env["home"]


def test_source_checkout_defaults_to_public(env):
    _checkout(env)
    assert paths.config_path() == env["project"] / "config.json"
    assert paths.data_dir() == env["project"] / "public"


def test_missing_config_is_empty_not_an_error(env):
    _installed(env)
    assert paths.load_config() == {}


# ── Overrides ──────────────────────────────────────────────────────────────────

def test_config_env_var_picks_config_file(env, tmp_path):
    _installed(env)
    cfg = tmp_path / "elsewhere" / "my-config.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"user_handle": "Asha"}), encoding="utf-8")
    env["mp"].setenv(paths.CONFIG_ENV, str(cfg))
    assert paths.config_path() == cfg.resolve()
    assert paths.load_config()["user_handle"] == "Asha"


def test_relative_data_dir_is_relative_to_config_file(env, tmp_path_factory, monkeypatch):
    _installed(env)
    env["home"].mkdir(parents=True)
    (env["home"] / "config.json").write_text(json.dumps({"data_dir": "store"}), encoding="utf-8")
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))  # cwd must not matter
    assert paths.data_dir() == (env["home"] / "store").resolve()


def test_absolute_data_dir(env, tmp_path_factory):
    _checkout(env)
    target = tmp_path_factory.mktemp("abs") / "store"
    (env["project"] / "config.json").write_text(json.dumps({"data_dir": str(target)}), encoding="utf-8")
    assert paths.data_dir() == target.resolve()


def test_data_dir_env_var_wins(env, tmp_path_factory):
    _checkout(env)
    target = tmp_path_factory.mktemp("env") / "store"
    (env["project"] / "config.json").write_text(json.dumps({"data_dir": "from_config"}), encoding="utf-8")
    env["mp"].setenv(paths.DATA_DIR_ENV, str(target))
    assert paths.data_dir() == target.resolve()


# ── Settings from environment ──────────────────────────────────────────────────

def test_env_settings_override_config_file(env):
    _checkout(env)
    (env["project"] / "config.json").write_text(
        json.dumps({"user_handle": "FromFile", "compression_model": "local", "scheduler_enabled": True}),
        encoding="utf-8",
    )
    env["mp"].setenv("CARETAKER_USER_HANDLE", "FromEnv")
    env["mp"].setenv("CARETAKER_SCHEDULER_ENABLED", "false")
    config = paths.load_config()
    assert config["user_handle"] == "FromEnv"
    assert config["compression_model"] == "local"
    assert config["scheduler_enabled"] is False
    # the file itself is untouched
    assert paths.read_config_file()["user_handle"] == "FromFile"


def test_anthropic_api_key_fallback(env):
    _installed(env)
    env["mp"].setenv("ANTHROPIC_API_KEY", "sk-test-fallback")
    assert paths.load_config()["anthropic_api_key"] == "sk-test-fallback"
    env["mp"].setenv("CARETAKER_ANTHROPIC_API_KEY", "sk-test-specific")
    assert paths.load_config()["anthropic_api_key"] == "sk-test-specific"


def test_write_config_file_creates_folder(env):
    _installed(env)
    path = paths.write_config_file({"user_handle": "New"})
    assert path == env["home"] / "config.json"
    assert paths.read_config_file() == {"user_handle": "New"}


# ── Locations ──────────────────────────────────────────────────────────────────

def test_all_locations_live_under_data_dir(env):
    _installed(env)
    root = env["home"]
    assert paths.db_path() == root / "data" / "caretaker.db"
    assert paths.chromadb_dir() == root / "data" / "chromadb"
    assert paths.logs_dir() == root / "logs"
    assert paths.exports_dir() == root / "exports"
    assert paths.cache_dir() == root / "cache"
    for d in (root / "data", root / "data" / "chromadb", root / "logs", root / "exports", root / "cache"):
        assert d.is_dir()


def test_old_chromadb_path_setting_is_ignored(env):
    _checkout(env)
    (env["project"] / "config.json").write_text(
        json.dumps({"database": {"chromadb_path": "C:/somewhere/else"}}), encoding="utf-8"
    )
    assert paths.chromadb_dir() == env["project"] / "public" / "data" / "chromadb"


def test_cache_env_points_into_data_dir(env):
    _installed(env)
    for var in ("HF_HOME", "SENTENCE_TRANSFORMERS_HOME", "TORCH_HOME"):
        env["mp"].delenv(var, raising=False)
    paths.configure_cache_env()
    cache = env["home"] / "cache"
    assert os.environ["HF_HOME"] == str(cache / "huggingface")
    assert os.environ["SENTENCE_TRANSFORMERS_HOME"] == str(cache / "sentence_transformers")
    assert os.environ["TORCH_HOME"] == str(cache / "torch")


# ── CLI uses the same locations ────────────────────────────────────────────────

def test_export_defaults_to_exports_folder(env, tmp_path_factory):
    from click.testing import CliRunner
    import caretaker.storage.local_db as db_module
    from caretaker.cli.main import cli

    _installed(env)
    env["mp"].setattr(db_module, "DB_PATH", str(tmp_path_factory.mktemp("db") / "t.db"))
    db_module.run_migrations()
    now = "2026-01-01T00:00:00+00:00"
    db_module.save_memory({
        "id": "m1", "source_agent": "claude", "keywords": None, "short": "s",
        "full": "Builds Caretaker.", "type": "PROJECT", "subtype": None,
        "fact_type": "REPLACEABLE", "status": "ACTIVE", "superseded_by": None,
        "importance": 0.5, "decay_score": 1.0, "temperature": "HOT",
        "retrieval_count": 0, "created_at": now, "updated_at": now, "last_used": None,
    })
    result = CliRunner().invoke(cli, ["export"])
    assert result.exit_code == 0, result.output
    assert len(list((env["home"] / "exports").glob("caretaker_export_*.json"))) == 1


def test_config_set_creates_user_config(env):
    from click.testing import CliRunner
    from caretaker.cli.main import cli

    _installed(env)
    result = CliRunner().invoke(cli, ["config", "set", "user_handle", "Asha"])
    assert result.exit_code == 0, result.output
    assert json.loads((env["home"] / "config.json").read_text(encoding="utf-8"))["user_handle"] == "Asha"
