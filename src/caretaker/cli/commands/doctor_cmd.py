"""
cli/commands/doctor_cmd.py
0.5 — `caretaker doctor [--setup]`: check the installation and print fixes;
with --setup also print config snippets for AI apps.
"""

import json

import click

from caretaker.cli.formatters import BOLD, GREEN, GREY, RED, RESET, YELLOW


@click.command("doctor")
@click.option("--setup", is_flag=True, help="Also show the config to add Caretaker to your AI apps")
def doctor_cmd(setup):
    """Check that Caretaker is set up correctly, with plain-language fixes."""
    from caretaker.doctor import client_snippets, detect_clients, run_checks

    icons = {"ok": f"{GREEN}✓{RESET}", "warn": f"{YELLOW}!{RESET}", "error": f"{RED}✗{RESET}"}
    print(f"\n{BOLD}Caretaker doctor{RESET}\n")
    checks = run_checks()
    for c in checks:
        print(f"  {icons[c['status']]}  {BOLD}{c['title']}{RESET}  {GREY}{c['detail']}{RESET}")
        if c["fix"]:
            print(f"       → {c['fix']}")

    clients = detect_clients()
    if clients:
        print(f"\n{BOLD}AI apps found{RESET}")
        for c in clients:
            state = {True: f"{GREEN}Caretaker configured{RESET}", False: f"{YELLOW}not configured{RESET}",
                     None: f"{RED}{c['note']}{RESET}"}[c["configured"]]
            print(f"  {c['app']:<34} {state}\n  {GREY}{c['path']}{RESET}")

    if setup:
        print(f"\n{BOLD}Add Caretaker to an AI app{RESET}")
        for s in client_snippets():
            print(f"\n  {BOLD}{s['app']}{RESET}  {GREY}({s['where']}){RESET}")
            body = s.get("command") or json.dumps(s["json"], indent=2)
            print("\n".join("    " + line for line in body.splitlines()))
    elif any(c["id"] == "clients" and c["status"] != "ok" for c in checks):
        print(f"\n  Run {BOLD}caretaker doctor --setup{RESET} for copy-paste config.")

    problems = sum(c["status"] == "error" for c in checks)
    warnings = sum(c["status"] == "warn" for c in checks)
    print(f"\n  {problems} problem(s), {warnings} suggestion(s).\n")
