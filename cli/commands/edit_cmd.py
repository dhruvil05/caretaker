"""
cli/commands/edit_cmd.py
Phase 3 — caretaker edit command.

Opens memory short/full fields in a text editor.
Updates SQLite + re-embeds in ChromaDB after save.

Usage:
    caretaker edit <id>
    caretaker edit <id> --field short
    caretaker edit <id> --field full
"""

import click
import tempfile
import subprocess
import os
from cli.formatters import (
    print_success, print_error, print_info, print_warning, confirm,
    GREY, RESET, BOLD
)


@click.command("edit")
@click.argument("memory_id")
@click.option(
    "--field",
    type=click.Choice(["short", "full", "both"], case_sensitive=False),
    default="both",
    help="Which field to edit: short, full, or both (default: both)",
)
def edit_cmd(memory_id, field):
    """Edit memory content in your default text editor."""
    from storage.local_db import get_memory_by_id, get_all_memories, update_memory_fields

    # Resolve ID
    mem = get_memory_by_id(memory_id)
    if not mem:
        all_mems = get_all_memories(status=None)
        matches = [m for m in all_mems if m["id"].startswith(memory_id)]
        if len(matches) == 1:
            mem = matches[0]
        elif len(matches) > 1:
            print_error(f"Ambiguous prefix '{memory_id}' — {len(matches)} matches.")
            return
        else:
            print_error(f"Memory not found: '{memory_id}'")
            return

    mem_id   = mem["id"]
    mem_type = mem.get("type", "?")

    print_info(f"Editing memory {GREY}{mem_id[:8]}…{RESET} [{mem_type}]")

    # Build editor content
    edit_lines = [
        f"# Caretaker Memory Editor",
        f"# ID: {mem_id}",
        f"# Type: {mem_type}",
        f"# Lines starting with # are ignored.",
        f"# Save and close the editor to apply changes.",
        f"#",
    ]

    if field in ("short", "both"):
        edit_lines += [
            f"# ── SHORT (max ~60 tokens) ────────────────────────────────",
            f"[SHORT]",
            mem.get("short") or "",
            f"[/SHORT]",
            f"#",
        ]

    if field in ("full", "both"):
        edit_lines += [
            f"# ── FULL (max ~300 tokens) ────────────────────────────────",
            f"[FULL]",
            mem.get("full") or "",
            f"[/FULL]",
        ]

    original_content = "\n".join(edit_lines)

    # Write to temp file and open editor
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(original_content)
        tmp_path = tmp.name

    editor = (
        os.environ.get("VISUAL")
        or os.environ.get("EDITOR")
        or ("notepad" if os.name == "nt" else "nano")
    )

    try:
        subprocess.run([editor, tmp_path], check=True)
    except FileNotFoundError:
        # Try fallback editors
        for fallback in ("nano", "vim", "vi", "notepad"):
            try:
                subprocess.run([fallback, tmp_path], check=True)
                break
            except FileNotFoundError:
                continue
        else:
            print_error(
                f"No editor found. Set EDITOR environment variable. "
                f"Temp file at: {tmp_path}"
            )
            return
    except subprocess.CalledProcessError:
        print_warning("Editor exited with error. Changes not saved.")
        os.unlink(tmp_path)
        return

    # Read edited content
    with open(tmp_path, "r", encoding="utf-8") as f:
        edited = f.read()
    os.unlink(tmp_path)

    if edited.strip() == original_content.strip():
        print_info("No changes detected. Memory unchanged.")
        return

    # Parse edited fields
    updates = {}

    def _extract_block(text: str, tag: str) -> str:
        """Extract content between [TAG] and [/TAG]."""
        start_marker = f"[{tag}]"
        end_marker   = f"[/{tag}]"
        start = text.find(start_marker)
        end   = text.find(end_marker)
        if start == -1 or end == -1:
            return None
        content = text[start + len(start_marker):end].strip()
        return content

    if field in ("short", "both"):
        new_short = _extract_block(edited, "SHORT")
        if new_short is not None and new_short != (mem.get("short") or ""):
            updates["short"] = new_short

    if field in ("full", "both"):
        new_full = _extract_block(edited, "FULL")
        if new_full is not None and new_full != (mem.get("full") or ""):
            updates["full"] = new_full

    if not updates:
        print_info("No changes detected in content blocks. Memory unchanged.")
        return

    # Apply updates to SQLite
    ok = update_memory_fields(mem_id, updates)
    if not ok:
        print_error("Failed to update memory in database.")
        return

    # Re-embed in ChromaDB if short was changed
    if "short" in updates:
        try:
            from storage.vector_db import VectorDB
            from pathlib import Path
            import json

            config_path = Path(__file__).parent.parent.parent / "config.json"
            with open(config_path) as f:
                config = json.load(f)

            chromadb_path = config.get("database", {}).get("chromadb_path", "data/chromadb")
            vdb = VectorDB(persist_directory=chromadb_path)
            vdb.initialize()
            vdb.update(
                memory_id=mem_id,
                text=updates["short"],
                metadata={"type": mem_type, "temperature": mem.get("temperature", "WARM")},
            )
            print_success("SHORT updated and re-embedded in ChromaDB.")
        except Exception as e:
            print_warning(f"SQLite updated but ChromaDB re-embed failed: {e}")
    else:
        print_success("Memory updated in database.")

    for field_name, value in updates.items():
        preview = value[:60] + "…" if len(value) > 60 else value
        print_info(f"  {field_name}: {preview}")