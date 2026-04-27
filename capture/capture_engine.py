import uuid
import json
from datetime import datetime, timezone

from capture.entity_extractor import extract_entities
from capture.type_classifier import classify_type
from storage.local_db import save_memory


def get_temperature(importance: float) -> str:
    if importance >= 0.7:
        return "PRIORITY_HOT"
    elif importance >= 0.5:
        return "HOT"
    elif importance >= 0.2:
        return "WARM"
    else:
        return "COLD"


def count_tokens_approx(text: str) -> int:
    return len(text.split())


def run_capture(message: str, agent_id: str = "claude") -> dict:
    token_count = count_tokens_approx(message)

    if token_count > 400:
        full_content = _handle_long_message(message)
    else:
        full_content = message.strip()

    entities   = extract_entities(message)
    classified = classify_type(message)

    now = datetime.now(timezone.utc).isoformat()

    memory = {
        "id":              str(uuid.uuid4()),
        "source_agent":    agent_id,
        "keywords":        json.dumps(entities["keywords"]),
        "short":           None,
        "full":            full_content,
        "type":            classified["type"],
        "subtype":         classified["subtype"],
        "fact_type":       classified["fact_type"],
        "status":          "ACTIVE",
        "superseded_by":   None,
        "importance":      classified["importance"],
        "decay_score":     1.0,
        "temperature":     get_temperature(classified["importance"]),
        "retrieval_count": 0,
        "created_at":      now,
        "updated_at":      now,
        "last_used":       now,
    }

    success = save_memory(memory)

    if success:
        print(f"[CAPTURE] Saved memory {memory['id']} | type={memory['type']} | temp={memory['temperature']}")
    else:
        print(f"[CAPTURE] Failed to save memory")

    return memory


def _handle_long_message(message: str) -> str:
    words     = message.split()
    chunks    = []
    chunk_size = 300

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    if len(chunks) == 1:
        return chunks[0]

    compressed = f"[LONG MESSAGE - {len(chunks)} parts] " + " | ".join(
        chunk[:100] + "..." for chunk in chunks
    )
    return compressed