"""
cli/commands/privacy_cmd.py
0.5 — `caretaker privacy scan [--apply]`
Check stored memories for secrets and personal data. Report only by default;
--apply redacts/masks in place and records what was found. Never prints the
sensitive text itself.
"""

from collections import Counter

import click

from caretaker.cli.formatters import (
    BOLD, GREY, RESET, confirm, print_info, print_success, print_warning,
)


@click.group("privacy")
def privacy_cmd():
    """Find and handle sensitive data (keys, card numbers, contact details) in memories."""


@privacy_cmd.command("scan")
@click.option("--apply", "apply_changes", is_flag=True, help="Redact/mask what the policy says to, in place")
@click.option("--include-reviewed", is_flag=True, help="Also re-check memories marked 'not sensitive'")
@click.option("--force", is_flag=True, help="Skip the confirmation prompt with --apply")
def privacy_scan(apply_changes, include_reviewed, force):
    """Scan stored memories against your privacy settings."""
    from caretaker import paths
    from caretaker.privacy.review import scan_memories
    from caretaker.storage.local_db import run_migrations

    run_migrations()
    report = scan_memories(apply=False, include_reviewed=include_reviewed)
    items = report["items"]
    print_info(f"Scanned {report['scanned']} memories: {len(items)} with findings.")
    if not items:
        print_success("Nothing sensitive found.")
        return

    by_action = Counter()
    for item in items:
        for f in item["findings"]:
            by_action[(f["action"], f["label"])] += 1
    verbs = {"redact": "will be redacted", "mask": "will be masked", "flag": "flagged for review"}
    for (action, label), n in sorted(by_action.items()):
        print(f"  {BOLD}{n:>4}{RESET}  {label}  {GREY}{verbs.get(action, action)}{RESET}")

    to_change = sum(1 for i in items if i["will_change"])
    if not apply_changes:
        print_info(f"{to_change} memories would be changed. Run with --apply to do it, "
                   "or review them one by one in `caretaker dashboard` (Privacy page).")
        return
    if to_change and not force and not confirm(f"Redact/mask sensitive text in {to_change} memories now?"):
        print_info("Cancelled. Nothing changed.")
        return

    result = scan_memories(apply=True, include_reviewed=include_reviewed)
    print_success(f"Done: {result['changed']} memories changed, findings recorded for {result['with_findings']}.")
    print_info("The search index picks up the cleaned text at the next maintenance run "
               "(or run `caretaker maintenance`).")
    backups = list((paths.db_path().parent / "backups").glob("*.db"))
    exports = list(paths.exports_dir().glob("*.json"))
    if backups or exports:
        print_warning(f"Older copies may still contain the original text: {len(backups)} backup(s) in "
                      f"{paths.db_path().parent / 'backups'} and {len(exports)} export(s) in {paths.exports_dir()}. "
                      "Delete them if you don't need them.")
