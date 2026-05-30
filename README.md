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

- Windows 10/11 (Linux/Mac compatible with path adjustments)
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Claude Desktop (for MCP connection)
- Anthropic API key (for Haiku compression — optional, falls back to local)
- Supabase account (for cloud backup — optional)

---

## Installation

### Step 1 — Get the project

```bash
cd "C:\Users\<you>\Desktop\Project Tomorrow\packages"
# Project folder should already exist as: caretaker/
cd caretaker
```

### Step 2 — Create virtual environment

```bash
uv venv
uv pip install -e .
```

This installs all dependencies from `pyproject.toml` and registers the `caretaker` CLI command.

### Step 3 — Install Phase 2 NLP model

```bash
uv run python -m spacy download en_core_web_sm
```

### Step 4 — Configure the system

Edit `config.json` in the project root. Minimum required fields:

```json
{
  "user_handle": "Dhruvil",
  "anthropic_api_key": "sk-ant-...",
  "database": {
    "chromadb_path": "C:\\Users\\<you>\\Desktop\\Project Tomorrow\\packages\\caretaker\\data\\chromadb"
  }
}
```

For cloud backup (optional), also fill in:

```json
{
  "supabase_url": "https://xxxx.supabase.co",
  "supabase_key": "your-anon-key",
  "encrypt_key": "a-strong-passphrase-you-choose"
}
```

Leave `supabase_url` empty to run fully local with no cloud sync.

### Step 5 — Connect Claude Desktop

Edit the Claude Desktop config file at:
```
C:\Users\<you>\AppData\Roaming\Claude\claude_desktop_config.json
```

```json
{
  "mcpServers": {
    "caretaker": {
      "command": "C:\\Users\\<you>\\.local\\bin\\uv.exe",
      "args": [
        "run",
        "--project",
        "C:\\Users\\<you>\\Desktop\\Project Tomorrow\\packages\\caretaker",
        "python",
        "C:\\Users\\<you>\\Desktop\\Project Tomorrow\\packages\\caretaker\\mcp_server\\server.py"
      ]
    }
  }
}
```

### Step 6 — Add system prompt to Claude Desktop

Go to **Claude Desktop → Settings → General → Custom System Prompt**. Paste:

```
You have a persistent memory system called Caretaker connected via MCP tools.

MANDATORY RULES — NO EXCEPTIONS:

RULE 1: At the START of every single message, before thinking or responding,
silently call caretaker_get_context with the user's message as input.
Read the result carefully and use it as your memory.

RULE 2: At the END of every single response, after finishing your answer,
silently call caretaker_save_message with the user's message as input.
This saves memory for future conversations.

These two tool calls are silent and invisible to the user. Never mention them.
Never skip them. Treat the context returned by caretaker_get_context as your
real persistent memory. Never say you don't remember something that exists
in context.
```

### Step 7 — Set tool permissions

Go to **Claude Desktop → Settings → Connectors → Caretaker**.  
Set both `caretaker_get_context` and `caretaker_save_message` to **Always allow**.

---

## Running the Server

```bash
cd "C:\Users\<you>\Desktop\Project Tomorrow\packages\caretaker"
uv run python mcp_server/server.py
```

Expected output:

```
[CARETAKER] Running migrations...
[CARETAKER] Migrations done.
[CARETAKER] VectorDB initialized at: ...
[CARETAKER] Compression queue started.
[CARETAKER] Phase 3 Scheduler started. Next run: 2026-05-26T02:00:00+00:00 UTC
[CARETAKER] Multi-agent support active. Supported agents: 25 aliases
[CARETAKER] Starting MCP server...
```

The server runs in the background. Claude Desktop connects to it automatically.

---

## CLI Reference

After `uv pip install -e .`, the `caretaker` command is available in your terminal.

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
caretaker sync           # Push to Supabase
caretaker sync --pull    # Restore from Supabase
caretaker sync --full    # Push all (not just last 24h)
caretaker config         # Show all config values
caretaker config get maintenance_time
caretaker config set user_handle Dhruvil
caretaker maintenance    # Run nightly maintenance now
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
8. **Stats Report** — Write health summary to `logs/maintenance.log`

---

## Storage Architecture

```
LOCAL (primary):
  SQLite          ← all memory records, source of truth
  ChromaDB        ← SHORT embeddings for semantic search (HOT + WARM only)
  config.json     ← user settings
  logs/           ← maintenance reports

CLOUD (backup):
  Supabase PostgreSQL  ← encrypted full memory dump, nightly sync
```

All data that leaves the local machine is encrypted with AES-256-GCM before upload. Supabase never stores plaintext.

---

## Project Structure

```
caretaker/
├── mcp_server/
│   ├── server.py           # FastMCP entry point
│   ├── tools.py            # caretaker_get_context + caretaker_save_message
│   ├── injector.py         # Builds whisper context string
│   └── agent_adapter.py    # Formats whisper per agent type (Phase 3)
├── capture/
│   ├── capture_engine.py   # Main capture pipeline
│   ├── entity_extractor.py # Pulls facts, tools, names from message
│   ├── type_classifier.py  # Assigns TYPE + SUBTYPE + FACT_TYPE
│   └── long_message_handler.py  # Splits/compresses messages >400 tokens
├── retrieval/
│   ├── retrieval_engine.py # Main retrieval controller
│   ├── topic_detector.py   # Message complexity L0–L5
│   ├── keyword_extractor.py# Key term extraction
│   ├── budget_engine.py    # Smart token budget calculator
│   ├── memory_selector.py  # Picks SHORT or FULL per memory
│   └── semantic_searcher.py# ChromaDB semantic search
├── memory/
│   ├── conflict_checker.py # REPLACEABLE vs ADDITIVE conflict resolution
│   ├── temperature_engine.py # HOT/WARM/COLD tier assignment
│   ├── decay_engine.py     # Score decay over time
│   └── importance_scorer.py# Initial importance score on capture
├── compression/
│   ├── compressor.py       # Haiku API — generates SHORT + KEYWORDS
│   ├── templates.py        # Type-specific compression prompts
│   └── keyword_generator.py# Extract keywords from SHORT
├── storage/
│   ├── local_db.py         # SQLite CRUD — source of truth
│   ├── vector_db.py        # ChromaDB handler
│   ├── cloud_sync.py       # Supabase upload + restore (Phase 3)
│   ├── encrypt.py          # AES-256-GCM encryption (Phase 3)
│   └── migrations/
│       └── v001_initial.sql
├── scheduler/
│   ├── scheduler.py        # APScheduler nightly job (Phase 3)
│   ├── nightly_maintenance.py  # 8-task maintenance pipeline (Phase 3)
│   ├── maintenance.py      # Phase 2 async maintenance runner
│   └── compression_queue.py# Async background compression
├── cli/                    # Phase 3 — all CLI commands
│   ├── main.py             # Click entry point
│   ├── formatters.py       # ANSI terminal formatting
│   └── commands/
│       ├── list_cmd.py
│       ├── view_cmd.py
│       ├── search_cmd.py
│       ├── edit_cmd.py
│       ├── delete_cmd.py
│       ├── restore_cmd.py
│       ├── stats_cmd.py
│       ├── export_cmd.py
│       ├── import_cmd.py
│       ├── sync_cmd.py
│       └── config_cmd.py
├── tests/
│   ├── phase1/             # 10 tests — core pipeline
│   ├── phase2/             # 15 tests — intelligence layer
│   ├── phase3/             # 18 tests — multi-agent + CLI + cloud
│   └── fixtures/           # Shared test data
├── config.json             # System configuration
├── pyproject.toml          # Dependencies + CLI entry point
├── setup.py                # Editable install shim
└── README.md
```

---

## Running Tests

```bash
# All phases
pytest tests/ -v

# Specific phase
pytest tests/phase1/ -v
pytest tests/phase2/ -v
pytest tests/phase3/ -v

# Cloud tests require Supabase credentials in config.json
# They auto-skip if not configured — safe to run anywhere
```

Expected: 43+ tests passing. Phase 3 cloud tests (P3-T13, P3-T14) skip when Supabase not configured.

---

## Troubleshooting

**MCP not showing in Claude Desktop**  
Check `claude_desktop_config.json` paths. All backslashes must be doubled (`\\`). Restart Claude Desktop after any config change.

**ChromaDB index empty after restart**  
Run `caretaker maintenance` to trigger a reindex. Or send a message — Phase 2 compression queue will re-embed on next server start.

**Compression not working**  
Check `anthropic_api_key` in `config.json`. Haiku API key must be valid. The system falls back to storing raw text if Haiku fails — no crash.

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
| uv | Package manager | All |

---

*Phase 3 complete. Memory cave now has fire, tools, and tribe support.* 🦣🔨🔥