"""
cli/commands/view_cmd.py
Phase 3 — caretaker view command.

Usage:
    caretaker view <id>
    caretaker view abc12345        (partial ID prefix match)
"""

import click
from cli.formatters import format_memory_card, print_error, print_info


@click.command("view")
@click.argument("memory_id")
def view_cmd(memory_id):
    """Show full details of one memory unit."""
    from storage.local_db import get_memory_by_id, get_all_memories

    # Try exact match first
    mem = get_memory_by_id(memory_id)

    # Try prefix match (user typed partial ID)
    if not mem:
        all_mems = get_all_memories(status=None)
        matches = [m for m in all_mems if m["id"].startswith(memory_id)]
        if len(matches) == 1:
            mem = matches[0]
        elif len(matches) > 1:
            print_error(f"Ambiguous ID prefix '{memory_id}' — {len(matches)} matches found.")
            print_info("Use a longer prefix or full ID.")
            for m in matches[:5]:
                print(f"  {m['id']}  [{m.get('type')}]  {(m.get('short') or m.get('full') or '')[:50]}")
            return

    if not mem:
        print_error(f"Memory not found: '{memory_id}'")
        return

    print(format_memory_card(mem))