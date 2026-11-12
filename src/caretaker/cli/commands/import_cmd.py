"""
cli/commands/import_cmd.py
Phase 3 — caretaker import command.

Restores memories from a JSON export file.

Usage:
    caretaker import backup.json
    caretaker import backup.json --skip-existing
"""

from caretaker.paths import load_config
import click
import json
from pathlib import Path
from caretaker.cli.formatters import (
    print_success, print_error, print_info, print_warning, confirm
)


def _protect_imported(mem: dict) -> dict:
    """Apply the sensitive-data policy to an imported memory's text."""
    import json as _json
    from caretaker.privacy import load_policy, protect, sensitivity_of
    policy = load_policy()
    mem = dict(mem)
    findings = []
    for field in ("full", "short"):
        if isinstance(mem.get(field), str) and mem[field]:
            result = protect(mem[field], policy=policy)
            mem[field] = result.text
            findings.extend(result.findings)
    if findings:
        mem["sensitivity"] = sensitivity_of(findings)
        mem["privacy_findings"] = _json.dumps(findings)
    return mem


@click.command("import")
@click.argument("input_file")
@click.option(
    "--skip-existing/--overwrite", "skip_existing", default=True,
    help="Skip memories that already exist locally (default), or overwrite them"
)
@click.option(
    "--force", is_flag=True, default=False,
    help="Skip confirmation prompt"
)
def import_cmd(input_file, skip_existing, force):
    """Import memories from a JSON export file."""
    from caretaker.storage.local_db import get_memory_by_id, upsert_memory

    input_path = Path(input_file)
    if not input_path.exists():
        print_error(f"File not found: {input_file}")
        return

    # Load export file
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            export_data = json.load(f)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON file: {e}")
        return

    # Validate export format
    if not export_data.get("caretaker_export"):
        print_warning("File does not appear to be a Caretaker export. Proceeding anyway.")

    memories = export_data.get("memories", [])
    if not memories:
        print_info("No memories found in export file.")
        return

    print_info(f"Found {len(memories)} memories in export.")
    print_info(f"Exported at: {export_data.get('exported_at', 'unknown')}")
    print_info(f"Status filter used: {export_data.get('status_filter', 'ALL')}")

    if not force and not confirm(f"Import {len(memories)} memories?"):
        print_info("Cancelled.")
        return

    imported  = 0
    skipped   = 0
    failed    = 0

    for mem in memories:
        mem_id = mem.get("id")
        if not mem_id:
            failed += 1
            continue

        # Skip if already exists and skip_existing is True
        if skip_existing:
            existing = get_memory_by_id(mem_id)
            if existing:
                skipped += 1
                continue

        try:
            mem = _protect_imported(mem)
            ok = upsert_memory(mem)
            if ok:
                imported += 1
            else:
                failed += 1
        except Exception as e:
            print_warning(f"Failed to import {mem_id[:8]}: {e}")
            failed += 1

    if imported > 0:
        print_info("The search index catches up at the next maintenance run (nightly, or `caretaker maintenance`).")

    print_success(f"Import complete.")
    print_info(f"  Imported : {imported}")
    print_info(f"  Skipped  : {skipped}  (already existed)")
    print_info(f"  Failed   : {failed}")