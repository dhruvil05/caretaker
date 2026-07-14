"""
retrieval/memory_selector.py
Decides whether to include SHORT or FULL text for each memory unit.
Maximizes information density within the available token budget.
Strategy:
  - PRIORITY_HOT → always FULL (most critical context)
  - HOT          → FULL if budget allows, else SHORT
  - WARM         → SHORT preferred, FULL only if surplus budget
  - COLD         → SHORT only, skip if budget tight
"""

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# Approx token cost per form
# SHORT ≈ 60 tokens + 20 metadata overhead = 80
# FULL  ≈ 300 tokens + 20 metadata overhead = 320
SHORT_TOKEN_COST = 80
FULL_TOKEN_COST = 320


def select_memory_forms(
    memories: List[Dict],
    token_budget: int,
) -> Tuple[List[Dict], int]:
    """
    Given a ranked list of memories and a token budget,
    assign each memory either SHORT or FULL form.

    Args:
        memories: ranked list of memory dicts (must have 'temperature', 'short', 'full_text')
        token_budget: total tokens available for context window

    Returns:
        (selected_memories: List[Dict], tokens_used: int)
        Each dict gets a new key: 'selected_form' = "SHORT" | "FULL"
    """
    remaining = token_budget
    selected = []

    # Sort: PRIORITY_HOT first, then by relevance_score
    tier_order = {"PRIORITY_HOT": 0, "HOT": 1, "WARM": 2, "COLD": 3}
    sorted_mems = sorted(
        memories,
        key=lambda m: (
            tier_order.get(m.get("temperature", "WARM"), 2),
            -m.get("relevance_score", 0.0),
        )
    )

    for mem in sorted_mems:
        tier = mem.get("temperature", "WARM")
        has_full = bool(mem.get("full_text", "").strip())
        has_short = bool(mem.get("short", "").strip())

        if tier == "PRIORITY_HOT":
            # Always try FULL
            if has_full and remaining >= FULL_TOKEN_COST:
                mem["selected_form"] = "FULL"
                remaining -= FULL_TOKEN_COST
                selected.append(mem)
            elif has_short and remaining >= SHORT_TOKEN_COST:
                mem["selected_form"] = "SHORT"
                remaining -= SHORT_TOKEN_COST
                selected.append(mem)
            # else: skip (no budget even for SHORT)

        elif tier == "HOT":
            # Prefer FULL, fallback SHORT
            if has_full and remaining >= FULL_TOKEN_COST:
                mem["selected_form"] = "FULL"
                remaining -= FULL_TOKEN_COST
                selected.append(mem)
            elif has_short and remaining >= SHORT_TOKEN_COST:
                mem["selected_form"] = "SHORT"
                remaining -= SHORT_TOKEN_COST
                selected.append(mem)

        elif tier == "WARM":
            # SHORT preferred
            if has_short and remaining >= SHORT_TOKEN_COST:
                mem["selected_form"] = "SHORT"
                remaining -= SHORT_TOKEN_COST
                selected.append(mem)
            elif has_full and remaining >= FULL_TOKEN_COST:
                # Fallback FULL if SHORT missing
                mem["selected_form"] = "FULL"
                remaining -= FULL_TOKEN_COST
                selected.append(mem)

        elif tier == "COLD":
            # SHORT only, skip if budget < 2x SHORT
            if has_short and remaining >= SHORT_TOKEN_COST * 2:
                mem["selected_form"] = "SHORT"
                remaining -= SHORT_TOKEN_COST
                selected.append(mem)

        if remaining <= SHORT_TOKEN_COST:
            logger.debug("[MemorySelector] Budget exhausted. Stopping selection.")
            break

    tokens_used = token_budget - remaining
    logger.info(
        f"[MemorySelector] Selected {len(selected)}/{len(memories)} memories. "
        f"Tokens used: {tokens_used}/{token_budget}"
    )

    return selected, tokens_used


def format_for_context(memories: List[Dict]) -> str:
    """
    Format selected memories into a context string for the LLM.
    Uses selected_form to pick SHORT or FULL text.
    """
    lines = []

    for mem in memories:
        form = mem.get("selected_form", "SHORT")
        mem_type = mem.get("memory_type", "MEMORY")
        temp = mem.get("temperature", "HOT")
        text = mem.get("full_text", "") if form == "FULL" else mem.get("short", "")

        if not text:
            text = mem.get("full_text", "") or mem.get("short", "")

        if text:
            lines.append(f"[{temp}][{mem_type}] {text.strip()}")

    return "\n".join(lines)