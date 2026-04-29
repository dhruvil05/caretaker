"""
compression/templates.py
------------------------
Type-specific compression prompt templates for all 8 memory types.
Haiku uses these templates to generate SHORT summary + KEYWORDS from FULL memory text.

Phase 2 — Compression Tribe
"""

# ---------------------------------------------------------------------------
# SHORT summary instruction per type
# Each template is injected into the compressor prompt.
# Output must be ≤ 60 tokens. Clear. Dense. Fact-preserving.
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, dict] = {

    "PROJECT": {
        "instruction": (
            "Summarise this project memory in ≤60 tokens. "
            "Include: project name, tech stack, current status, and primary goal. "
            "Format: '[Project name] — [stack] — [status] — [goal]'. "
            "No filler words. Facts only."
        ),
        "example_short": "Caretaker — FastAPI + SQLite + ChromaDB — Phase 2 in progress — universal memory layer for AI agents.",
        "fact_focus": ["project name", "tech stack", "status", "goal"],
    },

    "PREFERENCE": {
        "instruction": (
            "Summarise this preference memory in ≤60 tokens. "
            "Include: what tool/style/format is preferred and any known reason. "
            "Format: 'Prefers [X] over [Y] for [reason if known]'. "
            "No filler words."
        ),
        "example_short": "Prefers Cursor over VS Code for AI-assisted coding. Uses dark theme.",
        "fact_focus": ["preferred item", "alternative", "reason"],
    },

    "PROBLEM": {
        "instruction": (
            "Summarise this problem memory in ≤60 tokens. "
            "Include: what broke or blocked, context, and resolution status. "
            "Format: '[Problem description] in [context]. Status: [resolved/unresolved]'. "
            "No filler words."
        ),
        "example_short": "ChromaDB index not persisting after server restart in Phase 2. Status: unresolved.",
        "fact_focus": ["what broke", "context", "resolution status"],
    },

    "DECISION": {
        "instruction": (
            "Summarise this decision memory in ≤60 tokens. "
            "Include: what was decided and the reason. "
            "Format: 'Decided to [X] because [reason]'. "
            "No filler words."
        ),
        "example_short": "Decided to use CLI over UI for memory management because faster to build and simpler to maintain.",
        "fact_focus": ["decision", "reason"],
    },

    "LEARNING": {
        "instruction": (
            "Summarise this learning memory in ≤60 tokens. "
            "Include: topic learned, skill level or depth, and context. "
            "Format: 'Learning/knows [topic] — [level] — [context]'. "
            "No filler words."
        ),
        "example_short": "Learning ChromaDB vector search — intermediate — applied in Caretaker Phase 2 semantic retrieval.",
        "fact_focus": ["topic", "level", "context"],
    },

    "PERSONAL": {
        "instruction": (
            "Summarise this personal memory in ≤60 tokens. "
            "Include: who the user is, their role, location, and any relevant background. "
            "Format: '[Name] — [role] — [location] — [background detail]'. "
            "No filler words."
        ),
        "example_short": "Dhruvil — developer — Ahmedabad — building AI tooling projects independently.",
        "fact_focus": ["name", "role", "location", "background"],
    },

    "EMOTION": {
        "instruction": (
            "Summarise this emotion memory in ≤60 tokens. "
            "Include: the feeling, what triggered it, and the topic or project. "
            "Format: 'Felt [emotion] about [topic] because [reason if known]'. "
            "No filler words."
        ),
        "example_short": "Felt excited about Caretaker Phase 2 starting. Motivated by seeing Phase 1 complete.",
        "fact_focus": ["emotion", "topic", "trigger"],
    },

    "CORRECTION": {
        "instruction": (
            "Summarise this correction memory in ≤60 tokens. "
            "Include: what was wrong, what is now correct, and context. "
            "Format: 'Corrected: [old belief/fact] → [new correct fact] in context of [topic]'. "
            "No filler words."
        ),
        "example_short": "Corrected: importance threshold default was 0.5 → now 0.4 as per final config decision.",
        "fact_focus": ["old fact", "new fact", "context"],
    },
}


# ---------------------------------------------------------------------------
# Helper: build the full prompt string for the compressor
# ---------------------------------------------------------------------------

def build_compression_prompt(memory_type: str, full_text: str) -> str:
    """
    Build the complete prompt to send to Haiku for compression.

    Args:
        memory_type: One of the 8 TYPE values (PROJECT, PREFERENCE, etc.)
        full_text:   The raw FULL memory content to compress.

    Returns:
        A complete prompt string ready to send to the Haiku API.
    """
    template = TEMPLATES.get(memory_type.upper())

    if not template:
        # Fallback: generic compression if type is unrecognised
        instruction = (
            "Summarise this memory in ≤60 tokens. "
            "Keep all key facts. Remove filler. No markdown."
        )
    else:
        instruction = template["instruction"]

    prompt = f"""You are a memory compression engine. Your only job is to compress memory text.

TASK: {instruction}

RULES:
- Output ONLY the SHORT summary. Nothing else.
- Maximum 60 tokens. Shorter is better if facts are preserved.
- No markdown, no bullet points, no quotes around output.
- No preamble like "Here is the summary:" — just the summary itself.
- Preserve all proper nouns, version numbers, technical names exactly.

MEMORY TYPE: {memory_type.upper()}

FULL MEMORY TEXT:
{full_text}

SHORT SUMMARY:"""

    return prompt


def build_keyword_prompt(short_text: str) -> str:
    """
    Build the prompt to extract 3-7 keywords from a SHORT summary.

    Args:
        short_text: The compressed SHORT memory text.

    Returns:
        A prompt string for keyword extraction.
    """
    prompt = f"""You are a keyword extraction engine.

TASK: Extract 3 to 7 keywords from the memory summary below.

RULES:
- Output ONLY a JSON array of lowercase keyword strings. Example: ["fastapi", "python", "caretaker"]
- 3 minimum, 7 maximum keywords.
- Prefer specific technical terms, proper nouns, project names, tool names.
- Do NOT include generic words like "user", "memory", "project", "thing".
- No markdown, no explanation, just the JSON array.

MEMORY SUMMARY:
{short_text}

KEYWORDS (JSON array only):"""

    return prompt


# ---------------------------------------------------------------------------
# Utility: get fact_focus hints for a given type (used by entity extractor)
# ---------------------------------------------------------------------------

def get_fact_focus(memory_type: str) -> list[str]:
    """Return the list of fact dimensions to focus on for a given type."""
    template = TEMPLATES.get(memory_type.upper(), {})
    return template.get("fact_focus", [])


def get_supported_types() -> list[str]:
    """Return all supported memory type keys."""
    return list(TEMPLATES.keys())