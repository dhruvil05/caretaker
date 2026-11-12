"""
cli/commands/restore_cmd.py
Phase 3 — caretaker restore command.

Restores an ARCHIVED or OUTDATED memory back to ACTIVE.
Also re-embeds the SHORT summary into ChromaDB.

Usage:
    caretaker restore <id>
"""

from caretaker.paths import load_config
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

    print_success("Memory restored to ACTIVE.")
    if mem.get("short"):
        print_info("The search index catches up at the next maintenance run (nightly, or `caretaker maintenance`).")
    else:
        print_info("No SHORT summary — run server to trigger compression.")