"""
caretaker/projects.py
Project scopes: keep memories from different projects apart.

A memory belongs to one project ('' = global). Recall inside a project sees
that project's memories plus global ones; with no project set, recall sees
everything (the pre-0.5 behaviour).

The project for a call comes from, in order:
  1. the `project` argument an agent passes to a tool,
  2. the CARETAKER_PROJECT environment variable (set it per workspace in an AI
     app's MCP config, e.g. a project-level .cursor/mcp.json),
  3. '' (global).
Facts about the person themselves (PERSONAL memories) are always global.
"""

import os
import re

PROJECT_ENV = "CARETAKER_PROJECT"
_INVALID = re.compile(r"[^a-z0-9._\-]+")


def normalise_project(name) -> str:
    """'My Project / v2' -> 'my-project-v2'. Empty or unusable names become ''."""
    if not name:
        return ""
    slug = _INVALID.sub("-", str(name).strip().lower()).strip("-.")
    return slug[:64]


def resolve_project(explicit: str = "") -> str:
    return normalise_project(explicit) or normalise_project(os.environ.get(PROJECT_ENV, ""))


GLOBAL_TYPES = {"PERSONAL"}
