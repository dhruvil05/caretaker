"""
cli/commands/import_cmd.py
Phase 3 — caretaker import command.

Restores memories from a JSON export file.

Usage:
    caretaker import backup.json
    caretaker import backup.json --skip-existing
"""

import click
import json
from pathlib import Path
from cli.formatters import (
    print_success, print_error, print_info, print_warning, confirm
)


@click.command("import")
@click.argument("input_file")
@click.option(
    "--skip-existing", "skip_existing", is_flag=True, default=True,
    help="Skip memories that already exist locally (default: True)"
)
@click.option(
    "--force", is_flag=True, default=False,
    help="Skip confirmation prompt"
)
def import_cmd(input_file, skip_existing, force):
    """Import memories from a JSON export file."""
    from storage.local_db import get_memory_by_id, upsert_memory

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
            ok = upsert_memory(mem)
            if ok:
                imported += 1
            else:
                failed += 1
        except Exception as e:
            print_warning(f"Failed to import {mem_id[:8]}: {e}")
            failed += 1

    # Re-embed imported memories into ChromaDB
    if imported > 0:
        try:
            from storage.vector_db import VectorDB
            import json as _json

            config_path = Path(__file__).parent.parent.parent / "config.json"
            with open(config_path) as f:
                config = _json.load(f)

            chromadb_path = config.get("database", {}).get("chromadb_path", "data/chromadb")
            vdb = VectorDB(persist_directory=chromadb_path)
            vdb.initialize()

            embed_count = 0
            for mem in memories:
                if mem.get("short") and mem.get("status") == "ACTIVE":
                    try:
                        vdb.add(
                            memory_id=mem["id"],
                            text=mem["short"],
                            metadata={
                                "type"       : mem.get("type", "UNKNOWN"),
                                "temperature": mem.get("temperature", "WARM"),
                            }
                        )
                        embed_count += 1
                    except Exception:
                        pass   # Non-fatal — server will re-embed on next maintenance

            print_info(f"Re-embedded {embed_count} memories into ChromaDB.")
        except Exception as e:
            print_warning(f"ChromaDB re-embedding failed: {e} — run server to re-embed.")

    print_success(f"Import complete.")
    print_info(f"  Imported : {imported}")
    print_info(f"  Skipped  : {skipped}  (already existed)")
    print_info(f"  Failed   : {failed}")