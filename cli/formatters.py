"""
cli/formatters.py
Phase 3 — CLI output formatting helpers for Caretaker.

All terminal output goes through these helpers. Keeps formatting
consistent across all 11 commands. Handles:
  - Tables with column alignment
  - Memory cards (full single-memory display)
  - Temperature colour coding (ANSI)
  - Truncation for long fields
  - Status badges
  - Stats display
"""

import json
import textwrap
from datetime import datetime, timezone
from typing import Optional

# ── ANSI colour codes ──────────────────────────────────────────────────────────
# Degrade gracefully on terminals that don't support colour

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
WHITE  = "\033[97m"
GREY   = "\033[90m"


# ── Temperature colours ────────────────────────────────────────────────────────

TEMP_COLOURS = {
    "PRIORITY_HOT": RED    + BOLD,
    "HOT":          YELLOW + BOLD,
    "WARM":         GREEN,
    "COLD":         BLUE + DIM,
    "ARCHIVED":     GREY,
}

STATUS_COLOURS = {
    "ACTIVE":   GREEN,
    "OUTDATED": YELLOW,
    "ARCHIVED": GREY,
}

TYPE_COLOURS = {
    "PROJECT":    CYAN,
    "PREFERENCE": MAGENTA,
    "PROBLEM":    RED,
    "DECISION":   YELLOW,
    "LEARNING":   BLUE,
    "PERSONAL":   GREEN,
    "EMOTION":    MAGENTA,
    "CORRECTION": RED,
}


def _colour_temp(temp: str) -> str:
    c = TEMP_COLOURS.get(temp, "")
    return f"{c}{temp}{RESET}" if c else temp


def _colour_status(status: str) -> str:
    c = STATUS_COLOURS.get(status, "")
    return f"{c}{status}{RESET}" if c else status


def _colour_type(mem_type: str) -> str:
    c = TYPE_COLOURS.get(mem_type, "")
    return f"{c}{mem_type}{RESET}" if c else mem_type


# ── Text helpers ───────────────────────────────────────────────────────────────

def truncate(text: str, max_len: int = 60) -> str:
    """Truncate text to max_len chars, add … if cut."""
    if not text:
        return GREY + "(empty)" + RESET
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"


def format_dt(iso: Optional[str]) -> str:
    """Format ISO datetime to human-readable local string."""
    if not iso:
        return GREY + "—" + RESET
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso[:16]


def format_score(score: Optional[float]) -> str:
    """Format float score as 0.00 with colour."""
    if score is None:
        return GREY + "—" + RESET
    s = f"{score:.2f}"
    if score >= 0.7:
        return GREEN + s + RESET
    elif score >= 0.4:
        return YELLOW + s + RESET
    else:
        return RED + s + RESET


def format_keywords(kw_raw) -> str:
    """Parse and format keywords field (JSON string or list)."""
    if not kw_raw:
        return GREY + "(none)" + RESET
    if isinstance(kw_raw, list):
        kws = kw_raw
    else:
        try:
            kws = json.loads(kw_raw)
        except Exception:
            return str(kw_raw)
    return CYAN + " · ".join(kws) + RESET


# ── Memory list row ────────────────────────────────────────────────────────────

def format_memory_row(mem: dict, index: int = None) -> str:
    """
    Format one memory as a single list row.
    Used by list_cmd and search_cmd.

    Output:
      [1]  PROJECT  HOT    0.85  2026-05-01  User is building Caretaker…
    """
    idx_str  = f"{GREY}[{index}]{RESET} " if index is not None else ""
    mem_type = _colour_type(mem.get("type", "?"))
    temp     = _colour_temp(mem.get("temperature", "?"))
    status   = mem.get("status", "ACTIVE")
    imp      = format_score(mem.get("importance"))
    date     = format_dt(mem.get("created_at"))
    mem_id   = GREY + mem.get("id", "")[:8] + "…" + RESET

    # Show short if available, else truncated full
    preview_text = mem.get("short") or mem.get("full") or ""
    preview = truncate(preview_text, 55)

    # OUTDATED marker
    status_marker = ""
    if status == "OUTDATED":
        status_marker = f" {YELLOW}[OUTDATED]{RESET}"
    elif status == "ARCHIVED":
        status_marker = f" {GREY}[ARCHIVED]{RESET}"

    return (
        f"{idx_str}{mem_id}  "
        f"{mem_type:<22}  "
        f"{temp:<22}  "
        f"{imp}  "
        f"{GREY}{date}{RESET}  "
        f"{preview}{status_marker}"
    )


def format_list_header() -> str:
    """Header row for memory list."""
    return (
        f"\n{BOLD}"
        f"{'ID':<10}  {'TYPE':<14}  {'TEMP':<14}  {'IMP'}  {'CREATED':<16}  PREVIEW"
        f"{RESET}"
        f"\n{GREY}{'─' * 100}{RESET}"
    )


# ── Memory card (single full view) ────────────────────────────────────────────

def format_memory_card(mem: dict) -> str:
    """
    Full detail card for one memory. Used by view_cmd.
    Shows every field, formatted.
    """
    sep  = GREY + "─" * 70 + RESET
    lines = [
        f"\n{BOLD}{'═' * 70}{RESET}",
        f"  {BOLD}MEMORY DETAIL{RESET}",
        f"{BOLD}{'═' * 70}{RESET}",
        f"  {GREY}ID          {RESET} {mem.get('id', '—')}",
        f"  {GREY}Type        {RESET} {_colour_type(mem.get('type', '—'))}  /  {mem.get('subtype') or GREY + '(no subtype)' + RESET}",
        f"  {GREY}Status      {RESET} {_colour_status(mem.get('status', '—'))}",
        f"  {GREY}Fact Type   {RESET} {mem.get('fact_type', '—')}",
        f"  {GREY}Temperature {RESET} {_colour_temp(mem.get('temperature', '—'))}",
        sep,
        f"  {GREY}Importance  {RESET} {format_score(mem.get('importance'))}",
        f"  {GREY}Decay Score {RESET} {format_score(mem.get('decay_score'))}",
        f"  {GREY}Retrievals  {RESET} {mem.get('retrieval_count', 0)}",
        sep,
        f"  {GREY}Agent       {RESET} {mem.get('source_agent', '—')}",
        f"  {GREY}Created     {RESET} {format_dt(mem.get('created_at'))}",
        f"  {GREY}Updated     {RESET} {format_dt(mem.get('updated_at'))}",
        f"  {GREY}Last Used   {RESET} {format_dt(mem.get('last_used'))}",
    ]

    if mem.get("superseded_by"):
        lines.append(
            f"  {GREY}Superseded  {RESET} {YELLOW}{mem['superseded_by']}{RESET}"
        )

    lines += [
        sep,
        f"  {GREY}Keywords    {RESET} {format_keywords(mem.get('keywords'))}",
        sep,
        f"  {BOLD}SHORT{RESET}",
        f"  {mem.get('short') or GREY + '(not yet compressed)' + RESET}",
        sep,
        f"  {BOLD}FULL{RESET}",
    ]

    full_text = mem.get("full") or GREY + "(empty)" + RESET
    # Word-wrap full text at 68 chars
    wrapped = textwrap.fill(full_text, width=68, initial_indent="  ", subsequent_indent="  ")
    lines.append(wrapped)

    if mem.get("source_msg"):
        lines += [
            sep,
            f"  {BOLD}SOURCE MSG{RESET}",
            f"  {GREY}" + truncate(mem.get("source_msg", ""), 200) + RESET,
        ]

    lines.append(f"{BOLD}{'═' * 70}{RESET}\n")
    return "\n".join(lines)


# ── Search results ─────────────────────────────────────────────────────────────

def format_search_result(mem: dict, rank: int, score: float = None) -> str:
    """Format one search result row with rank and relevance score."""
    score_str = f"  {CYAN}score={score:.2f}{RESET}" if score is not None else ""
    return format_memory_row(mem, index=rank) + score_str


# ── Stats display ──────────────────────────────────────────────────────────────

def format_stats(stats: dict, scheduler_status: dict = None) -> str:
    """
    Format full stats dict into a readable terminal report.
    Used by stats_cmd.
    """
    lines = [
        f"\n{BOLD}{'═' * 50}{RESET}",
        f"  {BOLD}CARETAKER MEMORY STATS{RESET}",
        f"{BOLD}{'═' * 50}{RESET}",
        f"  {BOLD}Total Memories:{RESET}  {WHITE}{stats.get('total', 0)}{RESET}",
        "",
        f"  {BOLD}By Status:{RESET}",
    ]

    for status, count in sorted(stats.get("by_status", {}).items()):
        lines.append(f"    {_colour_status(status):<30}  {WHITE}{count}{RESET}")

    lines += ["", f"  {BOLD}By Temperature (ACTIVE only):{RESET}"]
    for temp, count in sorted(stats.get("by_temperature", {}).items()):
        lines.append(f"    {_colour_temp(temp):<30}  {WHITE}{count}{RESET}")

    lines += ["", f"  {BOLD}By Type (ACTIVE only):{RESET}"]
    for mtype, count in sorted(stats.get("by_type", {}).items()):
        lines.append(f"    {_colour_type(mtype):<30}  {WHITE}{count}{RESET}")

    lines += ["", f"  {BOLD}By Agent:{RESET}"]
    for agent, count in sorted(stats.get("by_agent", {}).items()):
        lines.append(f"    {CYAN}{agent:<28}{RESET}  {WHITE}{count}{RESET}")

    if scheduler_status:
        lines += [
            "",
            f"  {BOLD}Scheduler:{RESET}",
            f"    Running   {GREEN + 'YES' + RESET if scheduler_status.get('running') else RED + 'NO' + RESET}",
            f"    Schedule  {scheduler_status.get('schedule', '—')}",
            f"    Next Run  {scheduler_status.get('next_run') or GREY + 'not scheduled' + RESET}",
        ]

    lines.append(f"{BOLD}{'═' * 50}{RESET}\n")
    return "\n".join(lines)


# ── Generic helpers ────────────────────────────────────────────────────────────

def print_success(msg: str):
    print(f"{GREEN}✓{RESET}  {msg}")


def print_error(msg: str):
    print(f"{RED}✗{RESET}  {msg}")


def print_warning(msg: str):
    print(f"{YELLOW}!{RESET}  {msg}")


def print_info(msg: str):
    print(f"{CYAN}i{RESET}  {msg}")


def confirm(prompt: str) -> bool:
    """Ask user yes/no. Returns True if confirmed."""
    answer = input(f"{YELLOW}?{RESET}  {prompt} [y/N]: ").strip().lower()
    return answer in ("y", "yes")