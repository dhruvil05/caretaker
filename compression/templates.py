"""
compression/templates.py
Type-specific compression prompt templates for all 8 memory types.
Used by both HaikuCompressor and LocalCompressor.
"""

COMPRESSION_TEMPLATES = {
    "PROJECT": {
        "system": (
            "You are a memory compression engine. Extract the most important facts about "
            "a project from the user's message. Return ONLY a JSON object with two keys: "
            "'short' (max 60 tokens, one clear sentence about what is being built, stack used, "
            "and current status) and 'keywords' (array of 3-7 key terms). "
            "No preamble. No explanation. Only JSON."
        ),
        "user_prefix": "Compress this PROJECT memory:",
        "short_hint": "Building [what] using [stack]. Current status: [status].",
    },
    "PREFERENCE": {
        "system": (
            "You are a memory compression engine. Extract user preferences from the message. "
            "Return ONLY a JSON object with two keys: "
            "'short' (max 60 tokens, one clear sentence about what the user prefers and why) "
            "and 'keywords' (array of 3-7 key terms). "
            "No preamble. No explanation. Only JSON."
        ),
        "user_prefix": "Compress this PREFERENCE memory:",
        "short_hint": "Prefers [tool/style] for [reason/context].",
    },
    "PROBLEM": {
        "system": (
            "You are a memory compression engine. Extract the core problem or blocker. "
            "Return ONLY a JSON object with two keys: "
            "'short' (max 60 tokens, one clear sentence about what the problem is and where it occurs) "
            "and 'keywords' (array of 3-7 key terms). "
            "No preamble. No explanation. Only JSON."
        ),
        "user_prefix": "Compress this PROBLEM memory:",
        "short_hint": "Problem: [what broke/failed] in [context]. Impact: [effect].",
    },
    "DECISION": {
        "system": (
            "You are a memory compression engine. Extract the decision made and reason. "
            "Return ONLY a JSON object with two keys: "
            "'short' (max 60 tokens, one clear sentence about what was decided and why) "
            "and 'keywords' (array of 3-7 key terms). "
            "No preamble. No explanation. Only JSON."
        ),
        "user_prefix": "Compress this DECISION memory:",
        "short_hint": "Decided to [choice] because [reason].",
    },
    "LEARNING": {
        "system": (
            "You are a memory compression engine. Extract what was learned or studied. "
            "Return ONLY a JSON object with two keys: "
            "'short' (max 60 tokens, one clear sentence about the topic learned and skill level) "
            "and 'keywords' (array of 3-7 key terms). "
            "No preamble. No explanation. Only JSON."
        ),
        "user_prefix": "Compress this LEARNING memory:",
        "short_hint": "Studied [topic]. Skill level: [level]. Focus: [specific area].",
    },
    "PERSONAL": {
        "system": (
            "You are a memory compression engine. Extract personal facts about the user. "
            "Return ONLY a JSON object with two keys: "
            "'short' (max 60 tokens, one clear sentence about who the user is, role, location, background) "
            "and 'keywords' (array of 3-7 key terms). "
            "No preamble. No explanation. Only JSON."
        ),
        "user_prefix": "Compress this PERSONAL memory:",
        "short_hint": "[Name], [role], based in [location]. Background: [context].",
    },
    "EMOTION": {
        "system": (
            "You are a memory compression engine. Extract the user's emotional state or feeling. "
            "Return ONLY a JSON object with two keys: "
            "'short' (max 60 tokens, one clear sentence about how the user felt and about what) "
            "and 'keywords' (array of 3-7 key terms). "
            "No preamble. No explanation. Only JSON."
        ),
        "user_prefix": "Compress this EMOTION memory:",
        "short_hint": "Felt [emotion] about [topic/situation].",
    },
    "CORRECTION": {
        "system": (
            "You are a memory compression engine. Extract what was corrected or changed. "
            "Return ONLY a JSON object with two keys: "
            "'short' (max 60 tokens, one clear sentence about what was wrong before and what is correct now) "
            "and 'keywords' (array of 3-7 key terms). "
            "No preamble. No explanation. Only JSON."
        ),
        "user_prefix": "Compress this CORRECTION memory:",
        "short_hint": "Corrected: [old assumption]. Now: [correct fact].",
    },
}

# Fallback template for unknown types
DEFAULT_TEMPLATE = {
    "system": (
        "You are a memory compression engine. Summarize the key facts from the message. "
        "Return ONLY a JSON object with two keys: "
        "'short' (max 60 tokens, one clear summary sentence) "
        "and 'keywords' (array of 3-7 key terms). "
        "No preamble. No explanation. Only JSON."
    ),
    "user_prefix": "Compress this memory:",
    "short_hint": "Key fact: [main point].",
}


def get_template(memory_type: str) -> dict:
    """Return the compression template for a given memory type."""
    return COMPRESSION_TEMPLATES.get(memory_type.upper(), DEFAULT_TEMPLATE)