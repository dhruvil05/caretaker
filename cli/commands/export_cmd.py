"""
cli/commands/export_cmd.py
Phase 3 — caretaker export command.

Exports all memories to a JSON file for backup.

Usage:
    caretaker export
    caretaker export --file my_backup.json
    caretaker export --status ACTIVE
"""

import click
import json
from datetime import datetime, timezone
from pathlib import Path
from cli.formatters import print_success, print_error, print_info, GREY, RESET


@click.command("export")
@click.option(
    "--file", "output_file", default=None,
    help="Output file path (default: caretaker_export_YYYYMMDD.json)"
)
@click.option(
    "--status", "status_filter", default=None,
    help="Export only memories with this status (ACTIVE, OUTDATED, ARCHIVED)"
)
def export_cmd(output_file, status_filter):
    """Export all memories to a JSON file for backup or migration."""
    from storage.local_db import get_all_memories

    memories = get_all_memories(status=status_filter)

    if not memories:
        print_info("No memories to export.")
        return

    # Build export payload
    export_data = {
        "caretaker_export": True,
        "version"         : "3.0",
        "exported_at"     : datetime.now(timezone.utc).isoformat(),
        "total"           : len(memories),
        "status_filter"   : status_filter or "ALL",
        "memories"        : memories,
    }

    # Resolve output path
    if not output_file:
        date_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"caretaker_export_{date_str}.json"

    output_path = Path(output_file)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print_error(f"Export failed: {e}")
        return

    size_kb = output_path.stat().st_size / 1024
    print_success(f"Exported {len(memories)} memories → {output_path}")
    print_info(f"File size: {size_kb:.1f} KB")
    print_info(f"Restore with: caretaker import {output_path}")