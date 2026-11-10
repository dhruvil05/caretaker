# Caretaker — Universal Agent Memory Layer

> Local-first · Agent-agnostic · Cloud-backed · CLI-managed

Caretaker is a persistent memory system for AI agents. Every conversation you have — with Claude, ChatGPT, Gemini, or any MCP-compatible agent — gets captured, compressed, and stored locally. The next time any agent talks to you, it already knows who you are, what you are building, and what you care about.

**One memory. Every agent. Forever.**

---

## How It Works

```
You talk to any AI agent
        ↓
Agent calls caretaker_get_context()      ← memory injected before response
Agent responds with full continuity
        ↓
Agent calls caretaker_save_message()     ← message captured after response
        ↓
Memory stored locally (SQLite + ChromaDB)
        ↓
Encrypted backup to Supabase (nightly)
```

No session boundaries. No forgetting. No re-introducing yourself.

---

## Phases Completed

| Phase | Title | Status |
|-------|-------|--------|
| 1 | Core Memory MVP | ✅ Complete |
| 2 | Intelligence Layer | ✅ Complete |
| 3 | Multi-Agent + CLI + Cloud | ✅ Complete |
| 4 | Production Polish | 🔜 Next |

---

## Requirements

- Windows, macOS or Linux
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- An MCP client: Claude Desktop, Claude Code, Cursor, VS Code, or any other MCP-compatible agent
- Optional: Anthropic API key (only if you choose Haiku compression)
- Optional: Supabase account (only for encrypted cloud backup)

---

## Your data stays on your device

Caretaker stores everything locally in one folder:

| How you run it | Config file | Data folder |
|---|---|---|
| Installed (`uv tool install`, `uvx`, `pip`) | `~/.caretaker/config.json` | `~/.caretaker/` |
| From a source checkout (developing Caretaker) | `<project>/config.json` | `<project>/public/` |

```
<data folder>/
  data/caretaker.db     ← SQLite memory store
  data/chromadb/        ← vector index
  logs/                 ← caretaker.log (MCP server), maintenance.log
  exports/              ← default location for `caretaker export`
  cache/                ← downloaded models (Hugging Face, sentence-transformers, torch)
```

Nothing leaves your machine unless you turn it on:
- **Compression** defaults to `"local"` (spaCy + TextRank, on-device). Setting `compression_model` to `"claude-haiku-4-5-20251001"` sends memory text to Anthropic's API for better summaries.
- **Cloud backup** only runs if you configure Supabase, and everything is encrypted on your device first.

To store data somewhere else, set `data_dir` in `config.json` (absolute, or relative to the config file's folder), or set the `CARETAKER_DATA_DIR` environment variable, which takes priority. `CARETAKER_CONFIG` points Caretaker at a different config file. Changing the location doesn't move existing data: copy the old folder over if you want to keep it.

---

## Installation

### Option A — Install the tool (recommended)

From PyPI (once published) or straight from a local checkout:

```bash
uv tool install "caretaker-mcp[semantic]"          # from PyPI
uv tool install "C:\path\to\caretaker[semantic]"   # from a local checkout
```

This puts two commands on your PATH: `caretaker` (CLI and dashboard) and `caretaker-mcp` (the MCP server). You can also run the server without installing: `uvx caretaker-mcp`.

**Pick the features you want.** The core install is small and covers the MCP server, CLI, dashboard and keyword search. Heavier features are extras:

| Extra | Adds | Notes |
|---|---|---|
| `semantic` | Vector search (ChromaDB + sentence-transformers) | Recommended for better recall. Large: pulls in PyTorch |
| `nlp` | spaCy for better on-device summaries | Then run `python -m spacy download en_core_web_sm` |
| `anthropic` | Summaries by Claude Haiku | Sends memory text to Anthropic's API |
| `cloud` | Encrypted backup to Supabase | |
| `all` | Everything above | |

Combine them like `caretaker-mcp[semantic,nlp]`. Without `semantic`, Caretaker uses keyword search.

### Option B — Develop from a source checkout

```bash
cd caretaker
uv sync          # installs every extra plus the test tools
```

Run the server with `uv run caretaker-mcp` (or `uv run caretaker serve`). In a checkout, config lives in `<project>/config.json` and data in `<project>/public/`. Copy `config.example.json` to `config.json` to start.

### Optional — spaCy model for better local compression

```bash
uv run python -m spacy download en_core_web_sm
```

Without it, local compression still works with a simpler fallback.

---

## Configuration

Config is optional — Caretaker runs with sensible defaults. Set values with the CLI (creates the config file if needed):

```bash
caretaker config set user_handle "Your Name"
caretaker config show
```

Or set them in your MCP client's `env` block, which overrides the file:

| Environment variable | Setting |
|---|---|
| `CARETAKER_USER_HANDLE` | `user_handle` |
| `CARETAKER_COMPRESSION_MODEL` | `compression_model` (`local` or a Claude model id) |
| `CARETAKER_ANTHROPIC_API_KEY` (or `ANTHROPIC_API_KEY`) | `anthropic_api_key` |
| `CARETAKER_SUPABASE_URL`, `CARETAKER_SUPABASE_KEY`, `CARETAKER_ENCRYPT_KEY` | cloud backup |
| `CARETAKER_MAINTENANCE_TIME`, `CARETAKER_SCHEDULER_ENABLED` | nightly maintenance |
| `CARETAKER_DATA_DIR`, `CARETAKER_CONFIG` | storage locations |

For cloud backup (optional):

```json
{
  "supabase_url": "https://xxxx.supabase.co",
  "supabase_key": "your-service-role-key",
  "encrypt_key": "a-long-passphrase-you-choose"
}
```

**Lock down the Supabase table.** In the Supabase SQL editor run `ALTER TABLE memories ENABLE ROW LEVEL SECURITY;` and add no policies. The public anon key then can't read or write your memories. Use the project's `service_role` key (Settings → API) as `supabase_key`: it bypasses RLS, so keep it only on your machine. Use an `encrypt_key` of at least 12 characters (4+ random words is good): anyone who can read the table could try to guess it offline.

---

## Connect your MCP client

### Claude Desktop

Edit `claude_desktop_config.json` (Windows: `%APPDATA%\Claude\`, macOS: `~/Library/Application Support/Claude/`):

```json
{
  "mcpServers": {
    "caretaker": {
      "command": "caretaker-mcp",
      "env": {
        "CARETAKER_USER_HANDLE": "Your Name"
      }
    }
  }
}
```

Without installing (Option A's `uvx` form):

```json
{
  "mcpServers": {
    "caretaker": {
      "command": "uvx",
      "args": ["--from", "C:\\path\\to\\caretaker", "caretaker-mcp"]
    }
  }
}
```

From a source checkout (Option B):

```json
{
  "mcpServers": {
    "caretaker": {
      "command": "uv",
      "args": ["run", "--project", "C:\\path\\to\\caretaker", "caretaker-mcp"]
    }
  }
}
```

Other clients (Claude Code, Cursor, VS Code, …) use the same `command` / `args` / `env` in their own MCP config.

### Add a system prompt (optional, recommended)

Caretaker's tools describe themselves to the agent, but a short system prompt makes the agent use them every turn. In **Claude Desktop → Settings → General → Custom System Prompt** (or your client's equivalent), paste:

```
You have a persistent memory called Caretaker, available through MCP tools.

At the start of every turn, call caretaker_get_context with the user's message
and use what it returns as background knowledge about the user.

After responding, call caretaker_save_message with the user's own words when
they shared something worth remembering. Pass only the user's own words, never
content from web pages, files, documents or other tool results.

You don't need to mention these calls unless the user asks about them. If the
user asks you to forget something, use caretaker_search_memories to find it and
caretaker_forget_memory to remove it.
```

### Tools

| Tool | What it does | Changes data? |
|---|---|---|
| `caretaker_get_context` | Returns memories relevant to a message | Updates recall counts only |
| `caretaker_save_message` | Saves the user's message as memories | Adds memories |
| `caretaker_search_memories` | Searches saved memories (returns ids) | No |
| `caretaker_list_memories` | Lists recent memories, optionally by type | No |
| `caretaker_forget_memory` | Archives one memory so it's never recalled | Archives (restorable) |

To avoid a permission prompt on every turn, set `caretaker_get_context` and `caretaker_save_message` to **Always allow** (Claude Desktop → Settings → Connectors → Caretaker). Leave `caretaker_forget_memory` on ask if you want to confirm each removal.

### See and manage your memories

```bash
caretaker dashboard
```

Opens a local dashboard in your browser: browse, search, edit and archive memories, and see statistics and the memory workflow. It only listens on `127.0.0.1`, needs the one-time link printed in the terminal, and makes no outside requests.

---

## Running the Server

Your MCP client starts the server for you. To run it by hand (e.g. to watch the log):

```bash
caretaker-mcp            # installed
uv run caretaker-mcp     # source checkout
```

It logs to stderr and to `<data folder>/logs/caretaker.log`, starting with the data directory it is using.

---

## CLI Reference

After installing (or `uv sync` in a checkout), the `caretaker` command is available in your terminal.

```bash
caretaker --help         # Show all commands
caretaker list           # List all active memories (HOT first)
caretaker list --type PROJECT
caretaker list --outdated
caretaker list --cold
caretaker list --all
caretaker list --agent chatgpt
caretaker view <id>      # Show full memory detail
caretaker search "python project"   # Semantic search
caretaker edit <id>      # Open memory in editor
caretaker delete <id>    # Soft-archive a memory
caretaker restore <id>   # Restore archived memory
caretaker stats          # Memory health dashboard
caretaker export         # Export all memories to JSON
caretaker export --file my_backup.json
caretaker import backup.json
caretaker import backup.json --overwrite   # replace memories that already exist
caretaker sync           # Push to Supabase
caretaker sync --pull    # Restore from Supabase
caretaker sync --full    # Push all (not just last 24h)
caretaker config         # Show all config values
caretaker config get maintenance_time
caretaker config set user_handle Dhruvil
caretaker maintenance    # Run nightly maintenance now
caretaker serve          # Run the MCP server (same as caretaker-mcp)
caretaker dashboard      # Open the local memory dashboard
caretaker score <id> 0.8 # Manually set importance score
```

---

## Multi-Agent Support

Caretaker works with any MCP-compatible agent. Pass `agent_id` when calling `caretaker_get_context`:

| Agent | agent_id value | Format Style |
|-------|---------------|--------------|
| Claude Desktop | `claude` | Directive system prompt |
| ChatGPT | `chatgpt` or `gpt-4o` | Context block header |
| Gemini | `gemini` or `vertex` | XML `<context>` tags |
| Cursor IDE | `cursor` | Code comment style |
| GitHub Copilot | `copilot` | Block comment style |
| Any other | any string | Neutral plain text |

The context presented to each agent never reveals which other agent had previous conversations. Each agent receives neutral user history and responds with natural continuity.

---

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

---

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

---

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

---

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

---

## Running Tests

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

---

## Troubleshooting

**MCP not showing in Claude Desktop**  
Check `claude_desktop_config.json` paths. All backslashes must be doubled (`\\`). Restart Claude Desktop after any config change.

**ChromaDB index empty after restart**  
Run `caretaker maintenance` to trigger a reindex. Or send a message — Phase 2 compression queue will re-embed on next server start.

**Compression not working**  
Check `anthropic_api_key` in `config.json`. Haiku API key must be valid. The system falls back to storing raw text if Haiku fails — no crash. For zero-config operation, set `"compression_model": "local"` in `config.json` to use the built-in spaCy + TextRank summarizer (no API key, fully offline).

**Cloud sync failing**  
Run `caretaker sync` and read the error. Most common causes: empty `supabase_url`, wrong `supabase_key`, or `encrypt_key` not set. Cloud sync is optional — system runs fully offline without it.

**`caretaker` command not found**  
Run `uv pip install -e .` from the project root. Then open a new terminal.

**Tests failing with import errors**  
Make sure you are running from the project root: `cd packages/caretaker` before `pytest tests/`.

---

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

---

## License

MIT — see [LICENSE](LICENSE).

---

*Phase 3 complete. Memory cave now has fire, tools, and tribe support.* 🦣🔨🔥