"""
mcp_server/agent_adapter.py
Phase 3 — Agent-specific context formatting for Caretaker.

Every AI agent has slightly different expectations for how memory context
should be presented to it. This module normalises the whisper output from
injector.py into the correct format per agent.

Supported agents (agent_id values):
  "claude"       → Claude Desktop / Claude API (Anthropic)
  "chatgpt"      → ChatGPT with MCP connector / OpenAI API
  "gemini"       → Google Gemini / Vertex AI
  "cursor"       → Cursor IDE AI assistant
  "copilot"      → GitHub Copilot
  "custom"       → Any unknown/custom agent — safe neutral format

Key design principle (from caretaker_theory_v2.docx):
  The agent receives context, NOT source attribution.
  NOT: "User previously talked to Claude about X"
  YES: "User has been building X project using Y stack."
  The agent assumes continuity naturally. No awkward attribution.

How this plugs in:
  tools.py  get_context()
    → retrieval_engine.retrieve_context()      # get raw memories
    → injector.build_whisper(context)          # build raw whisper string
    → agent_adapter.adapt(whisper, agent_id)   # ← THIS FILE
    → return final formatted string to agent
"""

from typing import Literal

# Canonical agent IDs — normalised before lookup
AgentType = Literal["claude", "chatgpt", "gemini", "cursor", "copilot", "custom"]

# Agent ID aliases — map messy real-world values to canonical names
_AGENT_ALIASES: dict[str, AgentType] = {
    # Claude family
    "claude"              : "claude",
    "claude-desktop"      : "claude",
    "claude_desktop"      : "claude",
    "claude-api"          : "claude",
    "anthropic"           : "claude",

    # ChatGPT / OpenAI family
    "chatgpt"             : "chatgpt",
    "gpt"                 : "chatgpt",
    "gpt-4"               : "chatgpt",
    "gpt-4o"              : "chatgpt",
    "openai"              : "chatgpt",
    "chat_gpt"            : "chatgpt",

    # Gemini / Google family
    "gemini"              : "gemini",
    "google"              : "gemini",
    "vertex"              : "gemini",
    "gemini-pro"          : "gemini",
    "gemini-flash"        : "gemini",
    "bard"                : "gemini",

    # IDE / code agents
    "cursor"              : "cursor",
    "cursor-ai"           : "cursor",
    "copilot"             : "copilot",
    "github-copilot"      : "copilot",
    "github_copilot"      : "copilot",

    # Custom / unknown
    "custom"              : "custom",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def adapt(whisper: str, agent_id: str) -> str:
    """
    Format a raw whisper string for the target agent.

    Args:
        whisper:   Raw whisper string from injector.build_whisper()
        agent_id:  Agent identifier string (case-insensitive, alias-tolerant)

    Returns:
        Formatted context string ready to be prepended to the user message.
    """
    canonical = _normalise_agent_id(agent_id)
    formatter = _FORMATTERS.get(canonical, _format_custom)
    return formatter(whisper)


def normalise_agent_id(agent_id: str) -> AgentType:
    """
    Public accessor for canonical agent ID resolution.
    Used by tools.py and server.py to log clean agent names.
    """
    return _normalise_agent_id(agent_id)


def get_supported_agents() -> list[str]:
    """Return list of all recognised agent_id strings (aliases included)."""
    return sorted(_AGENT_ALIASES.keys())


def is_known_agent(agent_id: str) -> bool:
    """Return True if agent_id resolves to a known canonical agent."""
    return _normalise_agent_id(agent_id) != "custom"


# ── Internal normalisation ─────────────────────────────────────────────────────

def _normalise_agent_id(agent_id: str) -> AgentType:
    """
    Resolve any agent_id string to a canonical AgentType.
    Case-insensitive. Unknown values → "custom".
    """
    if not agent_id:
        return "claude"   # default
    key = agent_id.lower().strip()
    return _AGENT_ALIASES.get(key, "custom")


# ── Per-agent formatters ───────────────────────────────────────────────────────

def _format_claude(whisper: str) -> str:
    """
    Claude format — the native format Caretaker was designed for.
    Full system prompt injection style with explicit instruction block.
    Claude follows system-level instructions reliably — use directive language.
    """
    return f"""IMPORTANT - YOU HAVE MEMORY. READ THIS CAREFULLY:

{whisper}

INSTRUCTION: You already know everything above. Use this memory naturally in your response. Do not say you cannot remember. Do not ask user to remind you. You ALREADY know this information."""


def _format_chatgpt(whisper: str) -> str:
    """
    ChatGPT / OpenAI format.
    OpenAI models respond well to role-based framing.
    Uses "Your memory context:" framing rather than directive INSTRUCTION block
    since ChatGPT system prompts work better with descriptive context headers.
    No markdown headers (ChatGPT renders these literally in some UIs).
    """
    return f"""[MEMORY CONTEXT — read before responding]

{whisper}

Use this context naturally. Respond as if you already know this information about the user. Do not mention that you were given context. Do not say you don't remember things that appear above."""


def _format_gemini(whisper: str) -> str:
    """
    Google Gemini format.
    Gemini benefits from explicit XML-style tags around injected context
    (consistent with Google's recommended prompting patterns for Gemini).
    Uses <context> tag to signal structured input.
    """
    return f"""<context>
{whisper}
</context>

The above contains your persistent memory about this user. Use it naturally in your response. Do not reference that you were given context — respond as if this is prior knowledge you already have."""


def _format_cursor(whisper: str) -> str:
    """
    Cursor IDE AI format.
    Cursor operates primarily in a code context. Keep format minimal and
    code-editor-friendly. No heavy markdown — just a concise context block.
    Cursor works best with brief, direct injections.
    """
    # Strip heavy headers — keep just the essential memory lines
    condensed = _condense_whisper(whisper, max_lines=20)
    return f"""// [Caretaker Memory]
{condensed}
// [End Memory — use context above naturally]"""


def _format_copilot(whisper: str) -> str:
    """
    GitHub Copilot format.
    Copilot context injections work best as brief comments or natural prose.
    Strip the section headers — deliver as flat key-value context.
    """
    condensed = _condense_whisper(whisper, max_lines=15)
    return f"""/* Caretaker user context:
{condensed}
*/"""


def _format_custom(whisper: str) -> str:
    """
    Neutral format for unknown / custom agents.
    Safest possible format — no agent-specific assumptions.
    Plain text block with minimal structure.
    Any MCP-compatible agent should handle this.
    """
    return f"""[USER CONTEXT]
{whisper}
[END CONTEXT]

Note: The above is persistent memory about this user. Use it naturally."""


# ── Formatter registry ─────────────────────────────────────────────────────────

_FORMATTERS: dict[AgentType, callable] = {
    "claude"  : _format_claude,
    "chatgpt" : _format_chatgpt,
    "gemini"  : _format_gemini,
    "cursor"  : _format_cursor,
    "copilot" : _format_copilot,
    "custom"  : _format_custom,
}


# ── Whisper condensation helper ────────────────────────────────────────────────

def _condense_whisper(whisper: str, max_lines: int = 20) -> str:
    """
    Condense a full whisper string to a flat line list.
    Used by code-editor agents (Cursor, Copilot) that prefer brief injections.

    Strategy:
      - Strip section header lines (===, [CARETAKER CONTEXT], etc.)
      - Keep content lines only
      - Truncate to max_lines
    """
    skip_patterns = (
        "[CARETAKER CONTEXT]",
        "[END CARETAKER CONTEXT]",
        "=== CORE IDENTITY",
        "=== RECENT SESSIONS",
        "=== RELEVANT MEMORY",
        "===",
    )

    lines = whisper.splitlines()
    content_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(pat) or stripped == pat for pat in skip_patterns):
            continue
        content_lines.append(stripped)

    return "\n".join(content_lines[:max_lines])


# ── Agent info helper ─────────────────────────────────────────────────────────

def get_agent_info(agent_id: str) -> dict:
    """
    Return a metadata dict about a given agent_id.
    Used by CLI stats command and nightly maintenance log.

    Returns:
        {
            "raw_id"      : "gpt-4o",
            "canonical"   : "chatgpt",
            "known"       : True,
            "format_style": "system_prompt" | "context_block" | "xml_tag" | "code_comment" | "neutral",
        }
    """
    canonical = _normalise_agent_id(agent_id)

    style_map: dict[AgentType, str] = {
        "claude"  : "directive_system_prompt",
        "chatgpt" : "context_block",
        "gemini"  : "xml_tag",
        "cursor"  : "code_comment",
        "copilot" : "code_comment",
        "custom"  : "neutral",
    }

    return {
        "raw_id"      : agent_id,
        "canonical"   : canonical,
        "known"       : canonical != "custom",
        "format_style": style_map.get(canonical, "neutral"),
    }