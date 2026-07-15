"""
cli/commands/restore_cmd.py
Phase 3 — caretaker restore command.

Restores an ARCHIVED or OUTDATED memory back to ACTIVE.
Also re-embeds the SHORT summary into ChromaDB.

Usage:
    caretaker restore <id>
"""

import click
from caretaker.cli.formatters import (
    print_success, print_error, print_info, print_warning,
    format_memory_row
)


@click.command("restore")
@click.argument("memory_id")
def restore_cmd(memory_id):
    """Restore an ARCHIVED or OUTDATED memory back to ACTIVE."""
    from caretaker.storage.local_db import get_memory_by_id, get_all_memories, restore_memory

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

    status = mem.get("status")
    if status == "ACTIVE":
        print_warning("Memory is already ACTIVE. Nothing to restore.")
        return

    print_info(f"Restoring memory [{status}]:")
    print(f"  {format_memory_row(mem)}\n")

    ok = restore_memory(mem["id"])
    if not ok:
        print_error("Failed to restore memory.")
        return

    # Re-embed in ChromaDB if SHORT exists
    if mem.get("short"):
        try:
            from caretaker.storage.vector_db import VectorDB
            from pathlib import Path
            import json

            config_path = Path(__file__).parent.parent.parent.parent.parent / "config.json"
            with open(config_path) as f:
                config = json.load(f)

            chromadb_path = config.get("database", {}).get("chromadb_path", "data/chromadb")
            vdb = VectorDB(persist_directory=chromadb_path)
            vdb.initialize()
            vdb.add(
                memory_id=mem["id"],
                text=mem["short"],
                metadata={
                    "type"       : mem.get("type", "UNKNOWN"),
                    "temperature": "WARM",   # restored always starts WARM
                }
            )
            print_success("Memory restored to ACTIVE and re-embedded in ChromaDB.")
        except Exception as e:
            print_warning(f"Restored in SQLite but ChromaDB re-embed failed: {e}")
            print_success("Memory restored to ACTIVE.")
    else:
        print_success("Memory restored to ACTIVE.")
        print_info("No SHORT summary — run server to trigger compression.")