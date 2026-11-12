# Developing Caretaker

How Caretaker works inside, and how to work on it. For installing and using
Caretaker, see the [README](../README.md).

## 0.5 additions at a glance

| Module | What it does |
|---|---|
| `caretaker/storage/schema.py` | Versioned migrations; backs up the database before upgrading it |
| `caretaker/privacy/` | On-device detection of secrets / IDs / contact details, the redact/mask/flag policy, and the stored-data scan |
| `caretaker/projects.py` | Project scopes (`project` tool argument, `CARETAKER_PROJECT`) |
| `caretaker/doctor.py` | Setup checks + AI-app detection used by `caretaker doctor` and the dashboard Setup page |
| `caretaker/dashboard/api_v05.py` | Dashboard API for preview, privacy, review, bulk actions, projects, setup |
| `scripts/benchmark.py` | Speed and recall accuracy at 1k / 10k / … memories |

Design rules worth knowing:
- **SQLite is the source of truth.** Only the MCP server writes to ChromaDB; the CLI and dashboard write
  SQLite, archived memories are skipped at recall, and nightly maintenance re-indexes changed memories.
- **Privacy runs first.** `run_capture` redacts before classification, summarisation (which may call
  Haiku), storage and indexing. Imports, cloud restores, dashboard edits and recall output are checked too.
- **Recall only counts real matches.** Semantic hits below `min_relevance` are neither injected nor reheated.

## Phases Completed

| Phase | Title | Status |
|-------|-------|--------|
| 1 | Core Memory MVP | ✅ Complete |
| 2 | Intelligence Layer | ✅ Complete |
| 3 | Multi-Agent + CLI + Cloud | ✅ Complete |
| 4 | Production Polish | 🔜 Next |


## Memory System

### Memory Types

| Type | Fact Type | Example | Conflict Behaviour |
|------|-----------|---------|-------------------|
| PROJECT | REPLACEABLE | "Building FastAPI project" | New replaces old |
| PREFERENCE | REPLACEABLE | "Prefer Python over JS" | New replaces old |
| PROBLEM | ADDITIVE | "Getting 404 on /api/users" | Both kept |
| DECISION | ADDITIVE | "Decided to use Supabase" | Both kept |
| LEARNING | ADDITIVE | "Learning transformer models" | Both kept |
| PERSONAL | REPLACEABLE | "My name is Dhruvil" | New replaces old |
| EMOTION | ADDITIVE | "Excited about the progress" | Both kept |
| CORRECTION | REPLACEABLE | "Actually meant PostgreSQL" | New replaces old |

### Temperature Tiers

| Temperature | Score Condition | Retrieval |
|-------------|----------------|-----------|
| PRIORITY_HOT | importance > 0.7 | Always fetched first |
| HOT | score > 0.5 | Fetched in standard retrieval |
| WARM | 0.2 ≤ score ≤ 0.5 | Fetched only if semantically relevant |
| COLD | score < 0.2 | Never fetched — skipped in search |
| ARCHIVED | Manual or score < 0.2 | Never fetched — Supabase cold store |

### Token Budget (Smart Auto)

| Level | Signal | Budget | Memory Form |
|-------|--------|--------|-------------|
| L0 | "hi", "hello", "hey" | 80 tokens | Core identity only |
| L1 | "quick", "list", "simple" | 200 tokens | Core only |
| L2 | "explain", "what is", "help" | 350 tokens | Core + recent SHORT |
| L3 | "code", "debug", "implement" | 500 tokens | Core + relevant SHORT |
| L4 | "architecture", "design", "full flow" | 650 tokens | Core + relevant FULL |
| L5 | "remember everything about" | 800 tokens | All relevant FULL |


## Nightly Maintenance

Runs automatically every night at `maintenance_time` (default 02:00 UTC).

Trigger manually anytime:

```bash
caretaker maintenance
```

Pipeline runs 8 tasks in order:

1. **Batch Decay** — HOT → WARM after 7 days idle, WARM → COLD after 14 days
2. **ChromaDB Sync** — Remove OUTDATED and COLD entries from vector index
3. **Stale Cleanup** — Archive memories with score < 0.2
4. **Deduplication** — Merge near-identical memories (>70% keyword overlap)
5. **Importance Boost** — +0.02 per retrieval above threshold (max +0.15)
6. **Cloud Sync** — Encrypt + push last 24h updates to Supabase
7. **ChromaDB Reindex** — Re-add any ACTIVE memories missing from vector index
8. **Stats Report** — Write health summary to `<data_dir>/logs/maintenance.log`


## Storage Architecture

```
LOCAL (primary) — all under data_dir (default: public/):
  data/caretaker.db  ← SQLite: all memory records, source of truth
  data/chromadb/     ← ChromaDB: SHORT embeddings for semantic search (HOT + WARM only)
  logs/              ← server log + maintenance reports
  exports/           ← JSON exports
  cache/             ← model downloads
  config.json        ← user settings (project root)

CLOUD (backup):
  Supabase PostgreSQL  ← encrypted full memory dump, nightly sync
```

All data that leaves the local machine is encrypted with AES-256-GCM before upload. Supabase never stores plaintext.


## Project Structure

The package lives under `src/caretaker/` (PEP 517 `src/` layout, built with hatchling). `python -m caretaker` and the `caretaker` CLI both resolve to `src/caretaker/cli/main.py`.

```
caretaker/
├── src/
│   └── caretaker/
│       ├── __init__.py
│       ├── paths.py            # Where all data lives (data_dir → db, chromadb, logs, exports, cache)
│       ├── __main__.py         # Enables `python -m caretaker`
│       ├── mcp_server/
│       │   ├── server.py       # FastMCP entry point (imports stdout_guard first)
│       │   ├── stdout_guard.py # Routes ALL output to stderr (protects JSON-RPC)
│       │   ├── tools.py        # caretaker_get_context + caretaker_save_message
│       │   ├── injector.py     # Builds whisper context string
│       │   └── agent_adapter.py# Formats whisper per agent type (Phase 3)
│       ├── capture/
│       │   ├── capture_engine.py   # Main capture pipeline
│       │   ├── entity_extractor.py # Pulls facts, tools, names from message
│       │   ├── type_classifier.py  # Assigns TYPE + SUBTYPE + FACT_TYPE
│       │   └── long_message_handler.py  # Splits/compresses messages >400 tokens
│       ├── retrieval/
│       │   ├── retrieval_engine.py # Main retrieval controller
│       │   ├── topic_detector.py   # Message complexity L0–L5
│       │   ├── keyword_extractor.py# Key term extraction
│       │   ├── budget_engine.py    # Smart token budget calculator
│       │   ├── memory_selector.py  # Picks SHORT or FULL per memory
│       │   └── semantic_searcher.py# ChromaDB semantic search
│       ├── memory/
│       │   ├── conflict_checker.py # REPLACEABLE vs ADDITIVE conflict resolution
│       │   ├── temperature_engine.py # HOT/WARM/COLD tier assignment
│       │   └── importance_scorer.py# Initial importance score on capture
│       ├── compression/
│       │   ├── compressor.py       # Haiku API — generates SHORT + KEYWORDS
│       │   ├── local_compressor.py # FREE offline spaCy+TextRank compression
│       │   ├── templates.py        # Type-specific compression prompts
│       │   └── keyword_generator.py# Extract keywords from SHORT
│       ├── storage/
│       │   ├── local_db.py         # SQLite CRUD — source of truth
│       │   ├── vector_db.py        # ChromaDB handler
│       │   ├── cloud_sync.py       # Supabase upload + restore (Phase 3)
│       │   ├── encrypt.py          # AES-256-GCM encryption (Phase 3)
│       │   └── migrations/
│       │       └── v001_initial.sql
│       ├── scheduler/
│       │   ├── scheduler.py        # APScheduler nightly job (Phase 3)
│       │   ├── nightly_maintenance.py  # 8-task maintenance pipeline (Phase 3)
│       │   ├── maintenance.py      # Phase 2 async maintenance runner
│       │   └── compression_queue.py# Async background compression
│       ├── dashboard/          # Local web dashboard (`caretaker dashboard`)
│       │   ├── app.py          # Starlette app + JSON API (127.0.0.1 only, token-protected)
│       │   └── static/         # index.html, app.js, app.css (no external requests)
│       └── cli/
│           ├── main.py             # Click entry point (`caretaker` command)
│           ├── formatters.py       # ANSI terminal formatting
│           └── commands/
│               ├── list_cmd.py
│               ├── view_cmd.py
│               ├── search_cmd.py
│               ├── edit_cmd.py
│               ├── delete_cmd.py
│               ├── restore_cmd.py
│               ├── stats_cmd.py
│               ├── export_cmd.py
│               ├── import_cmd.py
│               ├── sync_cmd.py
│               └── config_cmd.py
├── tests/
│   ├── conftest.py             # Shared pytest fixtures + asyncio auto-mode
│   ├── fixtures/fixtures.py    # Shared test data
│   ├── phase1/                 # Core pipeline (capture, classify, db, mcp, retrieval)
│   ├── phase2/                 # Intelligence layer (compression, retrieval, decay)
│   └── phase3/                 # Multi-agent + CLI + cloud + maintenance
├── public/                  # Default data_dir: db, vector index, logs, exports, cache (gitignored)
├── config.example.json      # Template config (copy → config.json)
├── config.json              # System configuration (gitignored)
├── pyproject.toml          # Dependencies + hatchling build + CLI entry point
├── LICENSE                 # MIT
├── .github/workflows/      # CI: tests on Windows, macOS, Linux + core-install check
├── uv.lock                 # Locked dependency versions
└── README.md
```

> **Tests run against `src/`:** `pyproject.toml` sets `pythonpath = ["src"]` and `testpaths = ["tests"]`, so `pytest tests/` imports the real `caretaker` package from `src/`.


## Running Tests

Benchmark (never touches your data):

```bash
uv run python scripts/benchmark.py 1000 10000
```


```bash
# From the project root (pytest auto-discovers via pyproject config)
pytest tests/ -v

# Specific phase
pytest tests/phase1/ -v
pytest tests/phase2/ -v
pytest tests/phase3/ -v

# Cloud tests require Supabase credentials in config.json
# They auto-skip if not configured — safe to run anywhere
```

Expected: 200+ tests passing across all three phases. Phase 3 cloud tests skip automatically when Supabase is not configured.


## Tech Stack

| Technology | Purpose | Phase |
|-----------|---------|-------|
| Python 3.12+ | Core language | 1 |
| FastMCP | MCP server framework | 1 |
| Pydantic | Tool schemas / config models | 1 |
| SQLite | Local memory database | 1 |
| spaCy | NLP entity extraction | 1 |
| sentence-transformers | Local embeddings | 2 |
| ChromaDB | Vector search index | 2 |
| Anthropic Haiku | Memory compression | 2 |
| APScheduler | Nightly maintenance | 3 |
| Click | CLI framework | 3 |
| cryptography | AES-256-GCM encryption | 3 |
| Supabase | Cloud backup | 3 |
| pytest | Test framework | All |
| hatchling | Build backend (`src/` layout) | All |
| uv | Package manager | All |
