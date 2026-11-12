"""
caretaker/doctor.py
Setup checks shared by `caretaker doctor` and the dashboard's Setup page, plus
ready-to-paste MCP config for common AI apps.

Every check returns {id, title, status, detail, fix}; status is ok | warn | error.
AI-app detection only reads whether a "caretaker" server entry exists — it
never returns other servers' settings, env values or keys.
"""

import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from caretaker import paths


def _check(id_, title, status, detail, fix=""):
    return {"id": id_, "title": title, "status": status, "detail": detail, "fix": fix}


# ── How to launch the server on this machine ───────────────────────────────────

def server_command() -> dict:
    """{"command": ..., "args": [...]} that starts this Caretaker install."""
    if paths.IS_SOURCE_CHECKOUT:
        uv = shutil.which("uv") or "uv"
        return {"command": uv, "args": ["run", "--project", str(paths.PROJECT_ROOT), "caretaker-mcp"]}
    exe = shutil.which("caretaker-mcp")
    if exe:
        return {"command": exe, "args": []}
    return {"command": "uvx", "args": ["caretaker-mcp"]}


def _server_entry() -> dict:
    entry = server_command()
    env_dir = os.environ.get(paths.DATA_DIR_ENV)
    if env_dir:
        entry["env"] = {paths.DATA_DIR_ENV: env_dir}
    return entry


def client_snippets() -> list:
    """Config to paste into each AI app (same server entry, app-specific wrapper)."""
    entry = _server_entry()
    std = {"mcpServers": {"caretaker": {k: v for k, v in entry.items() if v != []}}}
    vscode = {"servers": {"caretaker": {"type": "stdio", **{k: v for k, v in entry.items() if v != []}}}}
    cli_args = " ".join(f'"{a}"' if " " in a else a for a in [entry["command"], *entry["args"]])
    return [
        {"app": "Claude Desktop", "where": "Settings → Developer → Edit Config", "json": std},
        {"app": "Cursor", "where": "~/.cursor/mcp.json (or .cursor/mcp.json in a project)", "json": std},
        {"app": "Windsurf", "where": "~/.codeium/windsurf/mcp_config.json", "json": std},
        {"app": "VS Code", "where": "MCP: Open User Configuration (mcp.json)", "json": vscode},
        {"app": "Claude Code", "where": "Run in a terminal", "command": f"claude mcp add caretaker -- {cli_args}"},
    ]


# ── AI app detection ───────────────────────────────────────────────────────────

def _candidate_configs() -> list:
    home = Path.home()
    appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    out = []
    if sys.platform == "win32":
        out.append(("Claude Desktop", appdata / "Claude" / "claude_desktop_config.json", "mcpServers"))
        for p in sorted(local.glob("Claude-Data/claude_*/claude_desktop_config.json")):
            out.append(("Claude Desktop (Microsoft Store)", p, "mcpServers"))
        for p in sorted(local.glob("Packages/Claude_*/LocalCache/Roaming/Claude/claude_desktop_config.json")):
            out.append(("Claude Desktop (Microsoft Store)", p, "mcpServers"))
        out.append(("VS Code", appdata / "Code" / "User" / "mcp.json", "servers"))
    elif sys.platform == "darwin":
        support = home / "Library" / "Application Support"
        out.append(("Claude Desktop", support / "Claude" / "claude_desktop_config.json", "mcpServers"))
        out.append(("VS Code", support / "Code" / "User" / "mcp.json", "servers"))
    else:
        out.append(("Claude Desktop", home / ".config" / "Claude" / "claude_desktop_config.json", "mcpServers"))
        out.append(("VS Code", home / ".config" / "Code" / "User" / "mcp.json", "servers"))
    out.append(("Cursor", home / ".cursor" / "mcp.json", "mcpServers"))
    out.append(("Windsurf", home / ".codeium" / "windsurf" / "mcp_config.json", "mcpServers"))
    out.append(("Claude Code", home / ".claude.json", "mcpServers"))
    return out


def _mentions_caretaker(server: dict) -> bool:
    if not isinstance(server, dict):
        return False
    parts = [str(server.get("command", ""))] + [str(a) for a in server.get("args", []) if isinstance(a, str)]
    return any("caretaker" in p.lower() for p in parts)


def detect_clients() -> list:
    found = []
    for app, path, key in _candidate_configs():
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            found.append({"app": app, "path": str(path), "configured": None, "note": "config file is not valid JSON"})
            continue
        servers = data.get(key) or {}
        configured = isinstance(servers, dict) and any(
            name.lower() == "caretaker" or _mentions_caretaker(s) for name, s in servers.items()
        )
        found.append({"app": app, "path": str(path), "configured": configured, "note": ""})
    return found


# ── Checks ─────────────────────────────────────────────────────────────────────

def run_checks() -> list:
    from caretaker.storage import local_db
    from caretaker.storage.schema import LATEST_VERSION, current_version, has_fulltext

    checks = []
    config = paths.load_config()

    # Data folder
    data_dir = paths.data_dir(config)
    writable = os.access(data_dir, os.W_OK)
    checks.append(_check("data_dir", "Data folder", "ok" if writable else "error",
                         f"{data_dir}" + ("" if writable else " is not writable"),
                         "" if writable else "Set data_dir in config.json or CARETAKER_DATA_DIR to a writable folder."))

    # Config
    cfg_path = paths.config_path()
    if not cfg_path.exists():
        checks.append(_check("config", "Config file", "ok", f"Not created yet ({cfg_path}); using defaults.",
                             "Optional: `caretaker config set user_handle \"Your Name\"`."))
    else:
        try:
            json.loads(cfg_path.read_text(encoding="utf-8"))
            checks.append(_check("config", "Config file", "ok", str(cfg_path)))
        except Exception as e:
            checks.append(_check("config", "Config file", "error", f"{cfg_path} is not valid JSON ({e}).",
                                 "Fix the JSON syntax (a missing comma or quote is the usual cause)."))

    # Database
    try:
        local_db.run_migrations()
        with local_db.get_connection() as conn:
            version = current_version(conn)
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            active = conn.execute("SELECT COUNT(*) FROM memories WHERE status='ACTIVE'").fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE status='ACTIVE' AND (short IS NULL OR short='')"
            ).fetchone()[0]
            flagged = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE sensitivity='flagged' AND privacy_reviewed=0"
            ).fetchone()[0]
            fts = has_fulltext(conn)
        checks.append(_check("database", "Memory database",
                             "ok" if version == LATEST_VERSION else "warn",
                             f"{active} active of {total} memories · schema v{version}"))
        checks.append(_check("fulltext", "Keyword search index", "ok" if fts else "warn",
                             "Full-text index ready" if fts else "This SQLite build has no FTS5; using slower search."))
        checks.append(_check("summaries", "Summaries", "ok" if pending < 25 else "warn",
                             f"{pending} memories waiting for a short summary",
                             "" if pending < 25 else "Summaries are made while the MCP server runs; keep it running."))
        checks.append(_check("privacy_review", "Privacy review", "ok" if not flagged else "warn",
                             f"{flagged} memories flagged for review" if flagged else "Nothing waiting for review",
                             "Open the dashboard's Privacy page to review them." if flagged else ""))
    except Exception as e:
        checks.append(_check("database", "Memory database", "error", f"Could not open the database: {e}",
                             "Check the data folder above, then run `caretaker doctor` again."))

    # Semantic search
    has_semantic = all(importlib.util.find_spec(m) for m in ("chromadb", "sentence_transformers"))
    if has_semantic:
        cache = paths.cache_dir(config)
        model_cached = any(cache.rglob("models--sentence-transformers--all-MiniLM-L6-v2"))
        checks.append(_check("semantic", "Semantic search", "ok",
                             "Installed" + ("; embedding model downloaded" if model_cached
                                            else "; the embedding model (~90 MB) downloads on first start")))
    else:
        checks.append(_check("semantic", "Semantic search", "warn", "Not installed — using keyword search.",
                             "For better recall: pip install 'caretaker-mcp[semantic]'"))

    # Privacy: what leaves the device
    from caretaker.privacy import data_flow
    flow = data_flow(config)
    if flow["compression_on_device"]:
        note = "Local summaries"
        if flow["compression"] != "local":
            note += f" ('{flow['compression']}' is set but there is no Anthropic API key)"
        checks.append(_check("compression", "Summaries stay on device", "ok", note))
    else:
        checks.append(_check("compression", "Summaries stay on device", "warn",
                             f"'{flow['compression']}' sends memory text to Anthropic's API",
                             "To keep everything local: `caretaker config set compression_model local`"))
    if flow["cloud_backup"]:
        checks.append(_check("cloud", "Cloud backup", "ok", "Encrypted backup to Supabase is configured",
                             "Make sure Row Level Security is on for the memories table."))
    elif flow["cloud_missing_key"]:
        checks.append(_check("cloud", "Cloud backup", "warn", "Supabase is set but encrypt_key is missing, so backup is off",
                             "Set encrypt_key to turn on encrypted backup, or remove the Supabase settings."))
    else:
        checks.append(_check("cloud", "Cloud backup", "ok", "Off (memories only on this device)"))

    # Server activity
    log = paths.logs_dir(config) / "caretaker.log"
    if log.exists():
        when = datetime.fromtimestamp(log.stat().st_mtime, tz=timezone.utc)
        checks.append(_check("server", "MCP server", "ok", f"Last activity {when.strftime('%Y-%m-%d %H:%M UTC')}"))
    else:
        checks.append(_check("server", "MCP server", "warn", "No server activity yet.",
                             "Add Caretaker to an AI app (see below), then restart that app."))

    # AI apps
    clients = detect_clients()
    configured = [c["app"] for c in clients if c["configured"]]
    if configured:
        checks.append(_check("clients", "Connected AI apps", "ok", ", ".join(sorted(set(configured)))))
    else:
        checks.append(_check("clients", "Connected AI apps", "warn",
                             "No AI app has Caretaker configured yet." if clients else "No supported AI app config found.",
                             "Use the snippets from `caretaker doctor --setup` or the dashboard's Setup page."))
    return checks
