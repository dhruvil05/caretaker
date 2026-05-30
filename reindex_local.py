"""
Emergency re-index script.
Generates SHORT from FULL text locally (no API needed).
Populates ChromaDB so semantic search works immediately.
Run once from caretaker root: python reindex_local.py
"""
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from storage.local_db import get_all_active_memories, update_compression
from storage.vector_db import VectorDB


def local_compress(full_text: str, memory_type: str) -> tuple[str, list[str]]:
    """
    Simple local compression — no API needed.
    Truncates full text to ~60 tokens and extracts keywords.
    """
    # Clean whitespace
    text = re.sub(r'\s+', ' ', full_text.strip())

    # Short = first 60 words (approx 60 tokens)
    words = text.split()
    short = ' '.join(words[:60])
    if len(words) > 60:
        short += '...'

    # Keywords = most significant words (filter stopwords, take top 7)
    STOPWORDS = {
        'the','a','an','is','are','was','were','be','been','being',
        'have','has','had','do','does','did','will','would','could',
        'should','may','might','shall','can','need','dare','ought',
        'i','me','my','we','our','you','your','he','she','it','they',
        'them','their','this','that','these','those','what','which',
        'who','whom','when','where','why','how','all','each','every',
        'both','few','more','most','other','some','such','no','nor',
        'not','only','same','so','than','too','very','just','but',
        'and','or','if','in','on','at','to','for','of','with','by',
        'from','up','about','into','through','during','before','after',
        'above','below','between','out','off','over','under','again',
        'further','then','once','here','there','s','t','re','ve','ll',
        'using','use','used','also','like','get','got','make','made',
    }

    word_freq = {}
    for w in text.lower().split():
        w_clean = re.sub(r'[^a-z0-9_\-]', '', w)
        if w_clean and len(w_clean) > 2 and w_clean not in STOPWORDS:
            word_freq[w_clean] = word_freq.get(w_clean, 0) + 1

    # Sort by frequency, take top 7
    keywords = sorted(word_freq, key=lambda k: word_freq[k], reverse=True)[:7]

    return short, keywords


def main():
    print("[REINDEX] Loading config...")
    config_path = Path("config.json")
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    chromadb_path = config.get("database", {}).get(
        "chromadb_path", "./data/chromadb"
    )

    print(f"[REINDEX] ChromaDB path: {chromadb_path}")
    print("[REINDEX] Initializing VectorDB...")

    vdb = VectorDB(persist_directory=chromadb_path)
    vdb.initialize()
    print(f"[REINDEX] ChromaDB current count: {vdb.count()}")

    print("[REINDEX] Loading memories from SQLite...")
    memories = get_all_active_memories()
    print(f"[REINDEX] Total active memories: {len(memories)}")

    no_short  = [m for m in memories if not m.get("short")]
    has_short = [m for m in memories if m.get("short")]

    print(f"[REINDEX] Has short: {len(has_short)} | No short: {len(no_short)}")

    # Step 1: memories that already have short — just upsert to ChromaDB
    print("\n[REINDEX] Step 1: Upserting memories that already have short...")
    for m in has_short:
        kws = []
        try:
            kws = json.loads(m.get("keywords") or "[]")
        except Exception:
            pass
        vdb.upsert(
            memory_id=m["id"],
            short=m["short"],
            keywords=kws,
            temperature=m.get("temperature", "HOT"),
            memory_type=m.get("type", "LEARNING"),
            importance_score=float(m.get("importance") or 0.5),
        )
    print(f"[REINDEX] Step 1 done: {len(has_short)} upserted.")

    # Step 2: memories with no short — generate locally and upsert
    print("\n[REINDEX] Step 2: Generating short locally for memories without...")
    done = 0
    for m in no_short:
        full_text = m.get("full") or ""
        if not full_text.strip():
            print(f"  [SKIP] {m['id']} — empty full text")
            continue

        short, keywords = local_compress(full_text, m.get("type", "LEARNING"))

        # Update SQLite
        update_compression(
            memory_id=m["id"],
            short=short,
            keywords=keywords,
        )

        # Upsert to ChromaDB
        vdb.upsert(
            memory_id=m["id"],
            short=short,
            keywords=keywords,
            temperature=m.get("temperature", "HOT"),
            memory_type=m.get("type", "LEARNING"),
            importance_score=float(m.get("importance") or 0.5),
        )
        done += 1
        if done % 10 == 0:
            print(f"  [{done}/{len(no_short)}] processed...")

    print(f"\n[REINDEX] Step 2 done: {done} memories compressed + indexed.")
    print(f"[REINDEX] ChromaDB final count: {vdb.count()}")
    print("\n[REINDEX] DONE. Restart server now. Memory should work!")


if __name__ == "__main__":
    main()