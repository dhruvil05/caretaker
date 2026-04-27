import json
from storage.local_db import get_connection


def _load_config() -> dict:
    import os
    from pathlib import Path
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def _get_core_identity(config: dict) -> str:
    user_handle = config.get("user_handle", "User")

    with get_connection() as conn:
        personal = conn.execute(
            "SELECT full FROM memories WHERE type='PERSONAL' AND status='ACTIVE' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        project = conn.execute(
            "SELECT full FROM memories WHERE type='PROJECT' AND status='ACTIVE' ORDER BY importance DESC LIMIT 1"
        ).fetchone()
        prefs = conn.execute(
            "SELECT full FROM memories WHERE type='PREFERENCE' AND status='ACTIVE' ORDER BY importance DESC LIMIT 3"
        ).fetchall()

    lines = [f"Name/Handle: {user_handle}"]

    if personal:
        lines.append(f"About: {personal['full'][:80]}")

    if project:
        lines.append(f"Active Project: {project['full'][:100]}")

    if prefs:
        pref_list = " · ".join(p["full"][:40] for p in prefs)
        lines.append(f"Key Preferences: {pref_list}")

    return "\n".join(lines)


def _get_recent_sessions(recent: list) -> str:
    if not recent:
        return "No recent sessions yet."

    lines = []
    for mem in recent:
        date    = mem.get("created_at", "")[:10]
        agent   = mem.get("source_agent", "claude")
        content = mem.get("short") or mem.get("full", "")[:80]
        lines.append(f"[{date}] via {agent}: {content}")

    return "\n".join(lines)


def _get_relevant_memories(relevant: list, use_full: bool) -> str:
    if not relevant:
        return "No relevant memories found."

    lines = []
    for mem in relevant[:8]:
        temp    = mem.get("temperature", "HOT")
        mtype   = mem.get("type", "MEMORY")
        status  = mem.get("status", "ACTIVE")
        content = mem.get("full") if use_full else (mem.get("short") or mem.get("full", ""))

        if not content:
            continue

        content = content[:300] if use_full else content[:80]

        if status == "OUTDATED":
            label = f"[OUTDATED][{mtype}]"
        else:
            label = f"[{temp}][{mtype}]"

        lines.append(f"{label} {content}")

    return "\n".join(lines)


def build_whisper(context: dict) -> str:
    config   = _load_config()
    relevant = context.get("relevant", [])
    recent   = context.get("recent", [])
    use_full = context.get("use_full", False)

    core     = _get_core_identity(config)
    sessions = _get_recent_sessions(recent)
    memories = _get_relevant_memories(relevant, use_full)

    whisper = f"""[CARETAKER CONTEXT]

=== CORE IDENTITY (always present) ===
{core}

=== RECENT SESSIONS (last 3) ===
{sessions}

=== RELEVANT MEMORY ===
{memories}

[END CARETAKER CONTEXT]"""

    return whisper