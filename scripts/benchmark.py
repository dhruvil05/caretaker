"""
scripts/benchmark.py — how Caretaker behaves as memory grows.

    uv run python scripts/benchmark.py            # 1,000 and 10,000 memories
    uv run python scripts/benchmark.py 1000 50000

Runs in a throwaway data folder (never touches your memories). Uses the keyword
(full-text) path, i.e. the core install; semantic search is not measured here.

Measures, per size:
  save        caretaker_save_message (privacy check, classification, duplicate check, insert)
  recall      caretaker_get_context end to end
  search      caretaker_search_memories
  accuracy    share of 30 planted "needle" facts that come back for a question about them
  scan        privacy scan of every memory (report only)
  dedup       nightly duplicate check — incremental (default) and full sweep
"""

import asyncio
import json
import os
import random
import statistics
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone

WORDS = ("api cache queue schema index deploy docker router client server token login invoice export "
         "report chart webhook cron backup billing search upload profile admin mobile layout theme "
         "python fastapi react postgres redis kafka celery nginx terraform kubernetes swift kotlin").split()
TYPES = ["PROJECT", "DECISION", "PROBLEM", "LEARNING", "PREFERENCE"]
DATABASES = ["PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "DynamoDB"]


def seed(db, n, rng):
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(n):
        words = rng.sample(WORDS, 6)
        created = (now - timedelta(days=rng.randint(0, 365), minutes=i % 1440)).isoformat()
        rows.append({
            "id": str(uuid.uuid4()), "source_agent": rng.choice(["claude", "cursor", "chatgpt"]),
            "keywords": json.dumps(words[:4]), "short": None,
            "full": f"Working on the {words[0]} {words[1]} with {words[2]} and {words[3]}; next: {words[4]} {words[5]} (#{i})",
            "type": rng.choice(TYPES), "subtype": None, "fact_type": "ADDITIVE", "status": "ACTIVE",
            "superseded_by": None, "importance": round(rng.uniform(0.2, 0.9), 2), "decay_score": 1.0,
            "temperature": rng.choice(["PRIORITY_HOT", "HOT", "WARM", "WARM", "COLD"]), "retrieval_count": 0,
            "created_at": created, "updated_at": created, "last_used": None,
            "project": rng.choice(["", "", "orbit", "crm", "nimbus"]), "sensitivity": "none",
            "privacy_findings": None, "privacy_reviewed": 0,
        })
    cols = list(rows[0])
    with db.get_connection() as conn:
        conn.executemany(
            f"INSERT INTO memories ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})", rows)


def plant_needles(db, rng, count=30):
    needles = []
    now = datetime.now(timezone.utc).isoformat()
    for i in range(count):
        name = f"Zylo{i:03d}"
        database = rng.choice(DATABASES)
        mid = str(uuid.uuid4())
        db.save_memory({
            "id": mid, "source_agent": "claude", "keywords": json.dumps([name.lower(), database.lower(), "database"]),
            "short": None, "full": f"Project {name} stores its data in {database}", "type": "DECISION",
            "subtype": None, "fact_type": "ADDITIVE", "status": "ACTIVE", "superseded_by": None,
            "importance": 0.5, "decay_score": 1.0, "temperature": "WARM", "retrieval_count": 0,
            "created_at": now, "updated_at": now, "last_used": None,
        })
        needles.append((mid, f"Which database does project {name} use?"))
    return needles


def timed(fn, repeat):
    times = []
    for i in range(repeat):
        t = time.perf_counter()
        fn(i)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times), max(times)


def run(size):
    from caretaker.mcp_server import tools
    from caretaker.privacy.review import scan_memories
    from caretaker.retrieval.retrieval_engine import retrieve_context
    from caretaker.scheduler.nightly_maintenance import NightlyMaintenance
    import caretaker.storage.local_db as db

    folder = tempfile.mkdtemp(prefix=f"caretaker-bench-{size}-")
    db.DB_PATH = os.path.join(folder, "bench.db")
    db._FTS_CACHE.clear()
    db.run_migrations()
    rng = random.Random(size)

    t = time.perf_counter()
    seed(db, size, rng)
    needles = plant_needles(db, rng)
    seed_s = time.perf_counter() - t

    save_med, save_max = timed(lambda i: tools.save_message(
        f"I decided the {rng.choice(WORDS)} service should use {rng.choice(WORDS)} for the {rng.choice(WORDS)} layer"), 30)
    recall_med, recall_max = timed(lambda i: tools.get_context(
        f"how does the {rng.choice(WORDS)} {rng.choice(WORDS)} work?"), 30)
    search_med, _ = timed(lambda i: tools.search_memories(rng.choice(WORDS)), 30)

    hits = 0
    for mid, question in needles:
        relevant = retrieve_context(question, dry_run=True)["relevant"]
        hits += any(m.get("id") == mid for m in relevant)

    t = time.perf_counter()
    scan_memories(apply=False)
    scan_s = time.perf_counter() - t

    maint = NightlyMaintenance(local_db=db, vector_db=None, config={})
    t = time.perf_counter()
    asyncio.run(maint._task_deduplication())
    dedup_inc_s = time.perf_counter() - t
    maint.config = {"dedup_lookback_days": 0}
    t = time.perf_counter()
    asyncio.run(maint._task_deduplication())
    dedup_full_s = time.perf_counter() - t

    return {
        "size": size, "seed_s": seed_s,
        "save_ms": save_med, "save_max_ms": save_max,
        "recall_ms": recall_med, "recall_max_ms": recall_max,
        "search_ms": search_med, "accuracy": hits / len(needles),
        "scan_s": scan_s, "dedup_incremental_s": dedup_inc_s, "dedup_full_s": dedup_full_s,
    }


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [1000, 10000]
    os.environ.setdefault("CARETAKER_DATA_DIR", tempfile.mkdtemp(prefix="caretaker-bench-data-"))
    import logging
    logging.disable(logging.WARNING)
    print(f"{'memories':>9} | {'save':>8} | {'recall':>8} | {'search':>8} | {'accuracy':>8} | {'privacy scan':>12} | {'dedup (incr / full)':>20}")
    print("-" * 95)
    for size in sizes:
        r = run(size)
        print(f"{r['size']:>9,} | {r['save_ms']:>6.1f}ms | {r['recall_ms']:>6.1f}ms | {r['search_ms']:>6.1f}ms | "
              f"{r['accuracy']:>8.0%} | {r['scan_s']:>11.1f}s | {r['dedup_incremental_s']:>8.2f}s / {r['dedup_full_s']:>7.2f}s")
    print("\nMedians of 30 runs. Recall uses the keyword path (core install); semantic search is not measured.")


if __name__ == "__main__":
    main()
