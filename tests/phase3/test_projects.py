"""
tests/phase3/test_projects.py
Project scopes: memories from different projects stay apart, global memories
are shared, personal facts are always global, CARETAKER_PROJECT works.
"""

import pytest

import caretaker.storage.local_db as db
from caretaker.projects import normalise_project, resolve_project


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "projects.db"))
    monkeypatch.delenv("CARETAKER_PROJECT", raising=False)
    db._FTS_CACHE.clear()
    db.run_migrations()


def rows():
    with db.get_connection() as conn:
        return [dict(r) for r in conn.execute("SELECT full, type, project FROM memories")]


def test_normalise_and_resolve(monkeypatch):
    assert normalise_project("  My Project / v2 ") == "my-project-v2"
    assert normalise_project("") == "" and normalise_project("///") == ""
    assert resolve_project("Orbit") == "orbit"
    monkeypatch.setenv("CARETAKER_PROJECT", "CRM App")
    assert resolve_project("") == "crm-app" and resolve_project("orbit") == "orbit"


def test_save_tags_project_but_personal_facts_stay_global():
    from caretaker.mcp_server.tools import save_message
    save_message("I am building Orbit with FastAPI and PostgreSQL", project="Orbit")
    save_message("My name is Dhruvil and I live in Ahmedabad", project="Orbit")
    by_text = {r["full"]: r for r in rows()}
    assert by_text["I am building Orbit with FastAPI and PostgreSQL"]["project"] == "orbit"
    personal = [r for r in rows() if "Dhruvil" in r["full"]][0]
    assert personal["type"] == "PERSONAL" and personal["project"] == ""


def test_env_var_sets_project_for_saves(monkeypatch):
    from caretaker.mcp_server.tools import save_message
    monkeypatch.setenv("CARETAKER_PROJECT", "crm")
    save_message("I am building the lead export feature for the CRM")
    assert {r["project"] for r in rows()} == {"crm"}


def test_recall_in_a_project_sees_own_and_global_only():
    from caretaker.mcp_server.tools import get_context, save_message
    save_message("I am building Orbit, the deploy uses Docker", project="orbit")
    save_message("I am building the CRM, the deploy uses Kubernetes", project="crm")
    save_message("I prefer short answers about deploy steps")          # global
    out = get_context("how do we deploy?", project="orbit")
    assert "Docker" in out and "Kubernetes" not in out and "short answers" in out
    everything = get_context("how do we deploy?")                       # no project: all
    assert "Docker" in everything and "Kubernetes" in everything


def test_search_and_list_respect_project():
    from caretaker.mcp_server.tools import list_memories, save_message, search_memories
    save_message("I am building Orbit with Postgres", project="orbit")
    save_message("I am building the CRM with Postgres", project="crm")
    found = search_memories("Postgres", project="orbit")
    assert "Orbit" in found and "CRM" not in found and "project=orbit" in found
    listed = list_memories(project="crm")
    assert "CRM" in listed and "Orbit" not in listed


def test_cli_list_project_filter():
    from click.testing import CliRunner
    from caretaker.cli.main import cli
    from caretaker.mcp_server.tools import save_message
    save_message("I am building Orbit with Postgres", project="orbit")
    save_message("I am building the CRM with Postgres", project="crm")
    out = CliRunner().invoke(cli, ["list", "--project", "orbit"]).output
    assert "Orbit" in out and "CRM" not in out
