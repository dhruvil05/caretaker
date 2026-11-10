"""
caretaker/dashboard
Local web dashboard: browse, edit and archive memories, see statistics and
the memory workflow. Runs only on this machine (127.0.0.1) and reads the same
local SQLite database the MCP server uses. Start it with `caretaker dashboard`.
"""

from .app import create_app, run_dashboard

__all__ = ["create_app", "run_dashboard"]
