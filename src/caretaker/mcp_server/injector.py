import json
import re
from caretaker.storage.local_db import get_connection


# ── Memory text sanitising ─────────────────────────────────────────────────────
# Memory text is data about the user, and it is placed inside a block in the
# agent's prompt. These helpers stop stored text from changing the structure
# of that block: no extra lines, no closing tags / comment terminators, no
# fake section headers. The words themselves are kept.

_WHITESPACE_RE = re.compile(r"[\s\x00-\x1f\x7f]+")
_TAG_RE = re.compile(
    r"<\s*/?\s*(context|memory|caretaker[^>]*|user[\s_-]*context)\s*>",
    re.IGNORECASE,
)
_HEADER_RE = re.compile(
    r"\[\s*((?:END\b|CARETAKER|USER CONTEXT|MEMORY CONTEXT)[^\]]*)\]",
    re.IGNORECASE,
)


def _safe_text(text, max_len: int = None) -> str:
    """Flatten one memory's text so it can't break out of its line/block."""
    if not text:
        return ""
    text = _WHITESPACE_RE.sub(" ", str(text))
    text = _TAG_RE.sub("(tag removed)", text)
    text = _HEADER_RE.sub(r"(\1)", text)
    text = text.replace("*/", "* /").replace("/*", "/ *").replace("===", "==")
    text = text.strip()
    return text[:max_len] if max_len else text


def _safe_label(value, default: str) -> str:
    """Labels such as type/temperature: uppercase letters and _ only."""
    label = re.sub(r"[^A-Z_]", "", str(value or "").upper())[:24]
    return label or default


def _safe_agent(value) -> str:
    """Agent names: short identifier characters only."""
    agent = re.sub(r"[^A-Za-z0-9_.-]", "", str(value or ""))[:32]
    return agent or "agent"


def _load_config() -> dict:
    from caretaker.paths import load_config
    return load_config()


def _get_core_identity(config: dict) -> str:
    user_handle = _safe_text(config.get("user_handle"), 40) or "User"

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
        lines.append(f"About: {_safe_text(personal['full'], 80)}")

    if project:
        lines.append(f"Active Project: {_safe_text(project['full'], 100)}")

    if prefs:
        pref_list = " · ".join(_safe_text(p["full"], 40) for p in prefs)
        lines.append(f"Key Preferences: {pref_list}")

    return "\n".join(lines)


def _get_recent_sessions(recent: list) -> str:
    if not recent:
        return "No recent sessions yet."

    lines = []
    for mem in recent:
        date    = re.sub(r"[^0-9-]", "", str(mem.get("created_at") or ""))[:10]
        agent   = _safe_agent(mem.get("source_agent", "claude"))
        content = _safe_text(mem.get("short") or mem.get("full", ""), 80)
        if not content:
            continue
        lines.append(f"[{date}] via {agent}: {content}")

    return "\n".join(lines) if lines else "No recent sessions yet."


def _get_relevant_memories(relevant: list, use_full: bool) -> str:
    if not relevant:
        return "No relevant memories found."

    lines = []
    for mem in relevant[:8]:
        temp   = _safe_label(mem.get("temperature"), "HOT")
        # Phase 2 fix: support both "type" and "memory_type"
        mtype  = _safe_label(mem.get("type") or mem.get("memory_type"), "MEMORY")
        status = mem.get("status", "ACTIVE")

        # Phase 3 fix: honour the per-memory form chosen by the budget engine
        # (selected_form), falling back to the batch use_full flag. This keeps a
        # memory selected as FULL from being collapsed to an 80-char SHORT snippet.
        form = mem.get("selected_form")
        if form == "FULL":
            content = mem.get("full") or mem.get("full_text", "")
            max_len = 300
        elif form == "SHORT":
            content = mem.get("short") or mem.get("full") or mem.get("full_text", "")
            max_len = 80
        elif use_full:
            content = mem.get("full") or mem.get("full_text", "")
            max_len = 300
        else:
            content = mem.get("short") or mem.get("full") or mem.get("full_text", "")
            max_len = 80

        if not content:
            continue

        content = _safe_text(content, max_len)
        if not content:
            continue

        if status == "OUTDATED":
            label = f"[OUTDATED][{mtype}]"
        else:
            label = f"[{temp}][{mtype}]"

        lines.append(f"{label} {content}")

    return "\n".join(lines) if lines else "No relevant memories found."


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