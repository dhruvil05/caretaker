"""
cli/commands/list_cmd.py
Phase 3 — caretaker list command.

Usage:
    caretaker list
    caretaker list --type PROJECT
    caretaker list --outdated
    caretaker list --cold
    caretaker list --agent chatgpt
    caretaker list --all
"""

import click
from src.caretaker.cli.formatters import (
    format_list_header, format_memory_row,
    print_info, print_warning, GREY, RESET, BOLD
)


@click.command("list")
@click.option("--type",    "mem_type", default=None, help="Filter by memory type (PROJECT, PREFERENCE, …)")
@click.option("--outdated","outdated", is_flag=True, default=False, help="Show OUTDATED memories")
@click.option("--cold",    "cold",     is_flag=True, default=False, help="Show COLD temperature memories")
@click.option("--all",     "show_all", is_flag=True, default=False, help="Show ALL memories (every status)")
@click.option("--agent",   "agent_id", default=None, help="Filter by source agent (claude, chatgpt, …)")
@click.option("--limit",   "limit",    default=50,   help="Max memories to show (default 50)")
def list_cmd(mem_type, outdated, cold, show_all, agent_id, limit):
    """List memories sorted by temperature (HOT first)."""
    from src.caretaker.storage.local_db import get_all_memories

    # Determine which status to pull
    if show_all:
        status_filter = None
    elif outdated:
        status_filter = "OUTDATED"
    else:
        status_filter = "ACTIVE"

    memories = get_all_memories(status=status_filter)

    # Filter by type
    if mem_type:
        memories = [m for m in memories if m.get("type", "").upper() == mem_type.upper()]

    # Filter by agent
    if agent_id:
        memories = [m for m in memories if m.get("source_agent", "").lower() == agent_id.lower()]

    # Filter cold (only if not show_all)
    if cold and not show_all:
        memories = [m for m in memories if m.get("temperature") == "COLD"]
    elif not cold and not show_all and not outdated:
        # Default: hide COLD memories
        memories = [m for m in memories if m.get("temperature") != "COLD"]

    # Sort: PRIORITY_HOT → HOT → WARM → COLD, then by importance desc
    tier_order = {"PRIORITY_HOT": 0, "HOT": 1, "WARM": 2, "COLD": 3, "ARCHIVED": 4}
    memories.sort(key=lambda m: (
        tier_order.get(m.get("temperature", "WARM"), 2),
        -(m.get("importance") or 0)
    ))

    # Apply limit
    total = len(memories)
    memories = memories[:limit]

    if not memories:
        print_info("No memories found. Capture some memories first!")
        return

    # Print header + rows
    print(format_list_header())
    for i, mem in enumerate(memories, start=1):
        print(format_memory_row(mem, index=i))

    print(f"\n{GREY}Showing {len(memories)} of {total} memories.{RESET}")
    if total > limit:
        print_warning(f"Use --limit {total} to see all.")