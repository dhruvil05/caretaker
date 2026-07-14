"""
cli/commands/sync_cmd.py
Phase 3 — caretaker sync command.

Manually triggers cloud sync to Supabase.

Usage:
    caretaker sync
    caretaker sync --pull       (restore from cloud to local)
    caretaker sync --push       (push local to cloud — default)
    caretaker sync --full       (push ALL memories, not just recent)
"""

import click
from src.caretaker.cli.formatters import (
    print_success, print_error, print_info, print_warning, GREY, RESET
)


@click.command("sync")
@click.option("--push", "direction", flag_value="push", default=True, help="Push local memories to Supabase (default)")
@click.option("--pull", "direction", flag_value="pull",               help="Pull memories from Supabase to local")
@click.option("--full", is_flag=True, default=False,                  help="Push ALL memories (not just recent 24h)")
def sync_cmd(direction, full):
    """Manually trigger cloud sync to/from Supabase."""
    from pathlib import Path
    import json

    config_path = Path(__file__).parent.parent.parent.parent.parent / "config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        print_error(f"Could not load config: {e}")
        return

    from src.caretaker.storage.cloud_sync import CloudSync
    cloud = CloudSync(config)

    if not cloud.is_configured():
        print_error(
            "Cloud sync not configured.\n"
            "  Set supabase_url, supabase_key, and encrypt_key in config.json"
        )
        return

    if not cloud.initialize():
        print_error("Could not connect to Supabase. Check your supabase_url and supabase_key.")
        return

    if direction == "push":
        print_info("Pushing memories to Supabase…")

        if full:
            result = cloud.push_all()
        else:
            from datetime import datetime, timedelta, timezone
            since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            result = cloud.push_since(since)

        if "error" in result:
            print_error(f"Sync failed: {result['error']}")
            return

        print_success("Push complete.")
        print_info(f"  Pushed  : {result.get('pushed', 0)}")
        print_info(f"  Failed  : {result.get('failed', 0)}")
        if not full:
            print_info(f"  (Last 24h window — use --full to sync all)")

    elif direction == "pull":
        print_info("Pulling memories from Supabase…")

        remote_count = cloud.get_remote_count()
        if remote_count is not None:
            print_info(f"  Remote memories: {remote_count}")

        result = cloud.pull_all()

        if "error" in result:
            print_error(f"Pull failed: {result['error']}")
            return

        print_success("Pull complete.")
        print_info(f"  Restored : {result.get('restored', 0)}")
        print_info(f"  Skipped  : {result.get('skipped', 0)}  (already up to date)")
        print_info(f"  Failed   : {result.get('failed', 0)}")