"""
cli/commands/config_cmd.py
Phase 3 — caretaker config command.

View or edit config.json from the terminal.

Usage:
    caretaker config                        (show all config)
    caretaker config get maintenance_time   (get one value)
    caretaker config set maintenance_time 03:00
    caretaker config set user_handle Dhruvil
"""

import click
import json
from pathlib import Path
from src.caretaker.cli.formatters import (
    print_success, print_error, print_info, print_warning,
    BOLD, GREY, RESET, CYAN, GREEN
)

CONFIG_PATH = Path(__file__).parent.parent.parent.parent.parent / "config.json"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_config(config: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _format_value(key: str, val) -> str:
    """Format a config value with type colouring."""
    if isinstance(val, bool):
        return (GREEN + "true" + RESET) if val else (GREY + "false" + RESET)
    if isinstance(val, (int, float)):
        return CYAN + str(val) + RESET
    if isinstance(val, str):
        if not val:
            return GREY + "(empty)" + RESET
        # Mask sensitive keys
        if any(s in key.lower() for s in ("key", "url", "password", "secret")):
            return GREY + "***hidden***" + RESET
        return GREEN + val + RESET
    if isinstance(val, dict):
        return GREY + "{…}" + RESET
    return str(val)


@click.group("config")
def config_cmd():
    """View or edit Caretaker configuration."""
    pass


@config_cmd.command("show")
def config_show():
    """Show all config values."""
    try:
        config = _load_config()
    except Exception as e:
        print_error(f"Could not load config: {e}")
        return

    print(f"\n{BOLD}{'═' * 50}{RESET}")
    print(f"  {BOLD}CARETAKER CONFIG{RESET}  {GREY}{CONFIG_PATH}{RESET}")
    print(f"{BOLD}{'═' * 50}{RESET}")

    for key, val in config.items():
        if key.startswith("_comment"):
            continue
        if isinstance(val, dict):
            print(f"  {GREY}{key}{RESET}")
            for k2, v2 in val.items():
                print(f"    {GREY}{k2:<30}{RESET}  {_format_value(k2, v2)}")
        else:
            print(f"  {GREY}{key:<32}{RESET}  {_format_value(key, val)}")

    print(f"{BOLD}{'═' * 50}{RESET}\n")


@config_cmd.command("get")
@click.argument("key")
def config_get(key):
    """Get a single config value."""
    try:
        config = _load_config()
    except Exception as e:
        print_error(f"Could not load config: {e}")
        return

    if key not in config:
        print_error(f"Key '{key}' not found in config.")
        return

    val = config[key]
    print(f"{GREY}{key}{RESET}  =  {_format_value(key, val)}")


@config_cmd.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a config value. Automatically casts to correct type."""
    try:
        config = _load_config()
    except Exception as e:
        print_error(f"Could not load config: {e}")
        return

    if key not in config:
        print_warning(f"Key '{key}' not in config — will be added.")

    # Type casting: try to match existing type
    existing = config.get(key)
    try:
        if isinstance(existing, bool):
            typed_value = value.lower() in ("true", "1", "yes")
        elif isinstance(existing, int):
            typed_value = int(value)
        elif isinstance(existing, float):
            typed_value = float(value)
        else:
            typed_value = value   # keep as string
    except ValueError:
        typed_value = value   # fallback to string

    old_val = config.get(key, "(not set)")
    config[key] = typed_value

    try:
        _save_config(config)
    except Exception as e:
        print_error(f"Could not save config: {e}")
        return

    print_success(f"Config updated.")
    print_info(f"  {key}:  {_format_value(key, old_val)}  →  {_format_value(key, typed_value)}")
    print_info("Restart the MCP server for changes to take effect.")


# Make bare `caretaker config` show config (not just help)
@click.pass_context
def config_cmd_default(ctx):
    if ctx.invoked_subcommand is None:
        ctx.invoke(config_show)