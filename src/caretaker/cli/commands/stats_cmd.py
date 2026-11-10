"""
cli/commands/stats_cmd.py
Phase 3 — caretaker stats command.

Usage:
    caretaker stats
"""

from caretaker.paths import load_config
import click
from caretaker.cli.formatters import format_stats, print_error


@click.command("stats")
def stats_cmd():
    """Show memory health stats: counts by type, status, temperature, agent."""
    from caretaker.storage.local_db import get_stats

    try:
        stats = get_stats()
    except Exception as e:
        print_error(f"Could not load stats: {e}")
        return

    # Try to get scheduler status
    scheduler_status = None
    try:
        from pathlib import Path
        import json

        config = load_config()

        maintenance_time = config.get("maintenance_time", "02:00")
        scheduler_status = {
            "running" : False,   # Scheduler is in server process, not CLI process
            "schedule": f"{maintenance_time} UTC",
            "next_run": None,
        }
    except Exception:
        pass

    print(format_stats(stats, scheduler_status=scheduler_status))