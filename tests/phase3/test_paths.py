"""
tests/phase3/test_paths.py
Data-directory resolution: default public/, config data_dir, env override,
and every file location living under the chosen folder.
"""

import json

import pytest

import caretaker.paths as paths


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    """Point paths at a throwaway project root with its own config.json."""
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paths, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(paths, "DEFAULT_DATA_DIR", tmp_path / "public")
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    return tmp_path


def _write_config(root, **values):
    (root / "config.json").write_text(json.dumps(values), encoding="utf-8")


def test_default_is_public_folder(fake_project):
    assert paths.data_dir() == fake_project / "public"


def test_empty_data_dir_falls_back_to_public(fake_project):
    _write_config(fake_project, data_dir="")
    assert paths.data_dir() == fake_project / "public"


def test_relative_data_dir_is_relative_to_project_not_cwd(fake_project, tmp_path_factory, monkeypatch):
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    _write_config(fake_project, data_dir="my_data")
    assert paths.data_dir() == fake_project / "my_data"
    assert paths.data_dir().is_dir()


def test_absolute_data_dir(fake_project, tmp_path_factory):
    target = tmp_path_factory.mktemp("abs") / "store"
    _write_config(fake_project, data_dir=str(target))
    assert paths.data_dir() == target


def test_env_var_overrides_config(fake_project, tmp_path_factory, monkeypatch):
    target = tmp_path_factory.mktemp("env") / "store"
    _write_config(fake_project, data_dir="from_config")
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(target))
    assert paths.data_dir() == target


def test_all_locations_live_under_data_dir(fake_project):
    _write_config(fake_project, data_dir="store")
    root = fake_project / "store"
    assert paths.db_path() == root / "data" / "caretaker.db"
    assert paths.chromadb_dir() == root / "data" / "chromadb"
    assert paths.logs_dir() == root / "logs"
    assert paths.exports_dir() == root / "exports"
    assert paths.cache_dir() == root / "cache"
    for d in (root / "data", root / "data" / "chromadb", root / "logs", root / "exports", root / "cache"):
        assert d.is_dir()


def test_old_chromadb_path_setting_is_ignored(fake_project):
    _write_config(fake_project, database={"chromadb_path": "C:/somewhere/else"})
    assert paths.chromadb_dir() == fake_project / "public" / "data" / "chromadb"


def test_cache_env_points_into_data_dir(fake_project, monkeypatch):
    for var in ("HF_HOME", "SENTENCE_TRANSFORMERS_HOME", "TORCH_HOME"):
        monkeypatch.delenv(var, raising=False)
    paths.configure_cache_env()
    import os
    cache = fake_project / "public" / "cache"
    assert os.environ["HF_HOME"] == str(cache / "huggingface")
    assert os.environ["SENTENCE_TRANSFORMERS_HOME"] == str(cache / "sentence_transformers")
    assert os.environ["TORCH_HOME"] == str(cache / "torch")


def test_export_defaults_to_exports_folder(fake_project, monkeypatch, tmp_path_factory):
    from click.testing import CliRunner
    import caretaker.storage.local_db as db_module
    from caretaker.cli.main import cli

    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path_factory.mktemp("db") / "t.db"))
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
    exported = list((fake_project / "public" / "exports").glob("caretaker_export_*.json"))
    assert len(exported) == 1
