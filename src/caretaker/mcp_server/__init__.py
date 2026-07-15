"""
caretaker.mcp_server
Public API for the MCP server tools, agent adaptation, and whisper injection.
"""

from .server import caretaker_get_context, caretaker_save_message
from .tools import get_context, save_message
from .agent_adapter import (
    adapt,
    normalise_agent_id,
    get_supported_agents,
    is_known_agent,
    get_agent_info,
)
from .injector import build_whisper

__all__ = [
    "caretaker_get_context",
    "caretaker_save_message",
    "get_context",
    "save_message",
    "adapt",
    "normalise_agent_id",
    "get_supported_agents",
    "is_known_agent",
    "get_agent_info",
    "build_whisper",
]
