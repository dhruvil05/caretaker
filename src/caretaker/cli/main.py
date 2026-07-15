"""
cli/main.py
Phase 3 — Caretaker CLI entry point.

Registered as 'caretaker' terminal command via setup.py entry_points.
All 11 commands wired here.

Usage:
    caretaker --help
    caretaker list
    caretaker list --type PROJECT
    caretaker list --outdated
    caretaker list --cold
    caretaker view <id>
    caretaker search "python project"
    caretaker edit <id>
    caretaker delete <id>
    caretaker restore <id>
    caretaker stats
    caretaker export
    caretaker export --file my_backup.json
    caretaker import backup.json
    caretaker sync
    caretaker sync --pull
    caretaker sync --full
    caretaker config
    caretaker config get <key>
    caretaker config set <key> <value>
    caretaker maintenance
"""

import click

from caretaker.cli.commands.list_cmd    import list_cmd
from caretaker.cli.commands.view_cmd    import view_cmd
from caretaker.cli.commands.search_cmd  import search_cmd
from caretaker.cli.commands.edit_cmd    import edit_cmd
from caretaker.cli.commands.delete_cmd  import delete_cmd
from caretaker.cli.commands.restore_cmd import restore_cmd
from caretaker.cli.commands.stats_cmd   import stats_cmd
from caretaker.cli.commands.export_cmd  import export_cmd
from caretaker.cli.commands.import_cmd  import import_cmd
from caretaker.cli.commands.sync_cmd    import sync_cmd
from caretaker.cli.commands.config_cmd  import config_cmd
from caretaker.cli.formatters           import BOLD, RESET, CYAN, GREEN, YELLOW


# ── CLI group ──────────────────────────────────────────────────────────────────

@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(version="0.3.2", prog_name="caretaker")
@click.pass_context
def cli(ctx):
    """
    \b
    🧠 CARETAKER — Universal Agent Memory Layer
    ─────────────────────────────────────────────
    Manage your AI memory from the terminal.

    \b
    QUICK START:
      caretaker list              List all memories
      caretaker search "python"   Semantic search
      caretaker stats             Memory health report
      caretaker sync              Push to Supabase cloud

    \b
    Run 'caretaker <command> --help' for command details.
    """
    if ctx.invoked_subcommand is None:
        # No subcommand — show a friendly dashboard
        _show_dashboard()


# ── Register all commands ──────────────────────────────────────────────────────

cli.add_command(list_cmd,    name="list")
cli.add_command(view_cmd,    name="view")
cli.add_command(search_cmd,  name="search")
cli.add_command(edit_cmd,    name="edit")
cli.add_command(delete_cmd,  name="delete")
cli.add_command(restore_cmd, name="restore")
cli.add_command(stats_cmd,   name="stats")
cli.add_command(export_cmd,  name="export")
cli.add_command(import_cmd,  name="import")
cli.add_command(sync_cmd,    name="sync")
cli.add_command(config_cmd,  name="config")

# ── Maintenance command (inline — triggers nightly pipeline now) ───────────────

@cli.command("maintenance")
def maintenance_cmd():
    """Manually run the nightly maintenance pipeline right now."""
    import asyncio
    from pathlib import Path
    import json

    config_path = Path(__file__).parent.parent.parent.parent / "config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        click.echo(f"Could not load config: {e}")
        return

    click.echo(f"\n{CYAN}Running nightly maintenance…{RESET}")

    try:
        from caretaker.scheduler.scheduler import run_maintenance_now
        result = asyncio.run(run_maintenance_now(config))

        click.echo(f"\n{BOLD}Maintenance Results:{RESET}")
        click.echo(f"  Decayed          : {result.get('decayed', 0)}")
        click.echo(f"  ChromaDB removed : {result.get('chroma_removed', 0)}")
        click.echo(f"  Archived         : {result.get('archived', 0)}")
        click.echo(f"  Deduped          : {result.get('deduped', 0)}")
        click.echo(f"  Boosted          : {result.get('boosted', 0)}")
        cloud = result.get("cloud", {})
        click.echo(f"  Cloud pushed     : {cloud.get('pushed', 0)}")
        click.echo(f"  Reindexed        : {result.get('reindexed', 0)}")
        click.echo(f"  Elapsed          : {result.get('elapsed_seconds', '?')}s")

        if result.get("error"):
            click.echo(f"\n{YELLOW}Warning: {result['error']}{RESET}")
        else:
            click.echo(f"\n{GREEN}✓ Maintenance complete.{RESET}\n")

    except Exception as e:
        click.echo(f"Maintenance failed: {e}")


# ── Score command (bonus: manually set importance score) ──────────────────────

@cli.command("score")
@click.argument("memory_id")
@click.argument("value", type=float)
def score_cmd(memory_id, value):
    """Manually set importance score for a memory (0.0 – 1.0)."""
    from caretaker.cli.formatters import print_success, print_error
    from caretaker.storage.local_db import get_memory_by_id, get_all_memories, update_memory_fields

    if not 0.0 <= value <= 1.0:
        print_error("Score must be between 0.0 and 1.0")
        return

    mem = get_memory_by_id(memory_id)
    if not mem:
        all_mems = get_all_memories(status=None)
        matches = [m for m in all_mems if m["id"].startswith(memory_id)]
        if len(matches) == 1:
            mem = matches[0]
        else:
            print_error(f"Memory not found: '{memory_id}'")
            return

    old_score = mem.get("importance", 0.5)

    # Recalculate temperature from new score
    from caretaker.memory.temperature_engine import assign_temperature
    new_temp = assign_temperature(value, mem.get("temperature", "WARM"))

    ok = update_memory_fields(mem["id"], {
        "importance" : value,
        "temperature": new_temp,
    })

    if ok:
        print_success(f"Score updated: {old_score:.2f} → {value:.2f}  (temp → {new_temp})")
    else:
        print_error("Failed to update score.")


# ── Dashboard (shown when `caretaker` run with no args) ───────────────────────

def _show_dashboard():
    """Quick summary shown when caretaker is run alone."""
    try:
        from caretaker.storage.local_db import get_stats
        stats = get_stats()

        total    = stats.get("total", 0)
        active   = stats.get("by_status", {}).get("ACTIVE", 0)
        hot      = stats.get("by_temperature", {}).get("HOT", 0)
        phot     = stats.get("by_temperature", {}).get("PRIORITY_HOT", 0)

        click.echo(f"\n{BOLD}🧠 Caretaker Memory Layer{RESET}  v3.0")
        click.echo(f"{'─' * 35}")
        click.echo(f"  Total memories : {WHITE_placeholder(total)}")
        click.echo(f"  Active         : {GREEN}{active}{RESET}")
        click.echo(f"  Hot + P.Hot    : {YELLOW}{hot + phot}{RESET}")
        click.echo(f"{'─' * 35}")
        click.echo(f"  Run {CYAN}caretaker --help{RESET} to see all commands.\n")
    except Exception:
        # If DB not set up yet, just show help hint
        click.echo(f"\n{BOLD}🧠 Caretaker{RESET}  Run: caretaker --help\n")


def WHITE_placeholder(val):
    """Simple white colour helper (avoids import cycle in formatters)."""
    return f"\033[97m{val}\033[0m"


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    """Entry point called by setup.py / pyproject.toml console_scripts."""
    cli()


if __name__ == "__main__":
    main()