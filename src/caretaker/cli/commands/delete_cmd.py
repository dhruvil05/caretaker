"""
cli/commands/delete_cmd.py
Phase 3 — caretaker delete command.

Soft-deletes a memory (sets status = ARCHIVED).
Never hard deletes. Reversible via: caretaker restore <id>

Usage:
    caretaker delete <id>
    caretaker delete <id> --force     (skip confirmation)
"""

import click
from src.caretaker.cli.formatters import (
    print_success, print_error, print_info, print_warning,
    confirm, format_memory_row, GREY, RESET
)


@click.command("delete")
@click.argument("memory_id")
@click.option("--force", is_flag=True, default=False, help="Skip confirmation prompt")
def delete_cmd(memory_id, force):
    """Archive a memory (soft delete — reversible with restore)."""
    from src.caretaker.storage.local_db import get_memory_by_id, get_all_memories, archive_memory

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

    if mem.get("status") == "ARCHIVED":
        print_warning("Memory is already ARCHIVED.")
        print_info("Use: caretaker restore <id>  to bring it back.")
        return

    # Show what will be deleted
    print_info("Memory to archive:")
    print(f"  {format_memory_row(mem)}\n")

    if not force and not confirm("Archive this memory?"):
        print_info("Cancelled.")
        return

    ok = archive_memory(mem["id"])
    if not ok:
        print_error("Failed to archive memory.")
        return

    # Remove from ChromaDB too
    try:
        from src.caretaker.storage.vector_db import VectorDB
        from pathlib import Path
        import json

        config_path = Path(__file__).parent.parent.parent.parent.parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        chromadb_path = config.get("database", {}).get("chromadb_path", "data/chromadb")
        vdb = VectorDB(persist_directory=chromadb_path)
        vdb.initialize()
        vdb.delete(mem["id"])
    except Exception as e:
        print_warning(f"Archived in SQLite but ChromaDB removal failed: {e}")

    print_success(f"Memory archived.  {GREY}{mem['id'][:8]}…{RESET}")
    print_info("Restore with: caretaker restore " + mem["id"][:8])