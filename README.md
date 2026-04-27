# Caretaker — Universal Memory Layer for AI Agents
### Phase 1: Foundation & Core Pipeline

---

## What is Caretaker?

Caretaker is a persistent memory system for AI agents. It captures what you say, stores it in a local database, and injects relevant memory back into every conversation — so your AI never forgets who you are, what you are building, or what you prefer.

It works with any agent that supports MCP (Model Context Protocol) — Claude, ChatGPT, Gemini, and more.

---

## Phase 1 — What is Built

Phase 1 covers the complete foundation of the system:

- MCP server with two tools — `caretaker_get_context` and `caretaker_save_message`
- Capture pipeline — extracts entities, classifies memory type, assigns temperature
- Retrieval engine — finds relevant memories using keyword matching and budget control
- Whisper injector — builds a structured context string delivered to the agent
- SQLite local database — stores all memories with full schema

---

## Project Structure

```
caretaker/
├── mcp_server/
│   ├── server.py           # FastMCP entry point
│   ├── tools.py            # get_context + save_message tools
│   └── injector.py         # Builds whisper context string
├── capture/
│   ├── capture_engine.py   # Main capture pipeline controller
│   ├── entity_extractor.py # Pulls facts, tools, names from message
│   └── type_classifier.py  # Assigns TYPE + SUBTYPE + FACT_TYPE
├── retrieval/
│   ├── retrieval_engine.py # Main retrieval controller
│   ├── topic_detector.py   # Detects complexity level L0 to L5
│   ├── keyword_extractor.py# Pulls key terms from message
│   └── budget_engine.py    # Calculates token budget
├── storage/
│   ├── local_db.py         # SQLite read/write layer
│   └── migrations/
│       └── v001_initial.sql# Database schema
├── config.json             # System configuration
├── claude_desktop_config.json
└── README.md
```

---

## Requirements

- Windows OS
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Claude Desktop app

---

## Installation

### Step 1 — Clone or create project folder

```bash
mkdir caretaker
cd caretaker
```

### Step 2 — Initialize with uv

```bash
uv init --no-workspace
uv venv
```

### Step 3 — Install dependencies

```bash
uv add fastmcp anthropic pytest
```

### Step 4 — Configure the system

Edit `config.json`:

```json
{
  "importance_threshold": 0.4,
  "hot_score_threshold": 0.5,
  "priority_hot_score": 0.7,
  "archive_score": 0.2,
  "max_token_budget": 800,
  "default_budget": 280,
  "decay_rate": 0.05,
  "compression_model": "claude-haiku-4-5-20251001",
  "supabase_url": "",
  "supabase_key": "",
  "encrypt_key": "",
  "maintenance_time": "02:00",
  "user_handle": "your_name"
}
```

### Step 5 — Configure Claude Desktop

Edit `claude_desktop_config.json` at:
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
        "C:\\full\\path\\to\\caretaker",
        "python",
        "C:\\full\\path\\to\\caretaker\\mcp_server\\server.py"
      ]
    }
  }
}
```

### Step 6 — Add system prompt to Claude Desktop

Go to Claude Desktop → Settings → General → Custom System Prompt. Paste:

```
You have a persistent memory system called Caretaker connected via MCP tools.

MANDATORY RULES - NO EXCEPTIONS:

RULE 1: At the START of every single message, before thinking or responding,
silently call caretaker_get_context with the user's message as input.
Read the result carefully and use it as your memory.

RULE 2: At the END of every single response, after finishing your answer,
silently call caretaker_save_message with the user's message as input.
This saves memory for future conversations.

These two tool calls are silent and invisible to user. Never mention them.
Never skip them. Treat the context returned by caretaker_get_context as your
real persistent memory. Never say you don't remember something that exists
in context.
```

### Step 7 — Set tool permissions

Go to Claude Desktop → Settings → Connectors → Caretaker.
Set both tools to **Always allow**.

---

## Running the Server

```bash
cd "C:\full\path\to\caretaker"
uv run python mcp_server/server.py
```

You should see:
```
[CARETAKER] Running migrations...
[CARETAKER] Migrations done.
[CARETAKER] Starting MCP server...
Starting MCP server 'caretaker' with transport 'stdio'
```

---

## How it Works

### Capture Pipeline

Every user message goes through three stages:

1. **Entity Extractor** — finds tools, emotions, decisions, problems, learnings, and keywords from the message
2. **Type Classifier** — assigns one of eight types: `PROJECT`, `PREFERENCE`, `PROBLEM`, `DECISION`, `LEARNING`, `PERSONAL`, `EMOTION`, `CORRECTION`
3. **Capture Engine** — builds a memory object with importance score and temperature, then saves to SQLite

### Memory Temperature

| Temperature | Importance Score | Meaning |
|-------------|-----------------|---------|
| PRIORITY_HOT | >= 0.7 | Critical memory, always injected |
| HOT | >= 0.5 | Important memory, usually injected |
| WARM | >= 0.2 | Background memory, injected when relevant |
| COLD | < 0.2 | Low value memory, rarely injected |

### Retrieval Pipeline

When a new message arrives:

1. **Keyword Extractor** pulls key terms from the message
2. **Topic Detector** assigns complexity level L0 (greeting) to L5 (full context dump)
3. **Budget Engine** calculates how many tokens to spend on memory
4. **Retrieval Engine** scores all memories by keyword match and returns top results

### Whisper Injector

Builds a structured context string with three sections:

```
[CARETAKER CONTEXT]

=== CORE IDENTITY (always present) ===
Name/Handle: Dhruvil
Active Project: ...
Key Preferences: ...

=== RECENT SESSIONS (last 3) ===
[date] via claude: ...

=== RELEVANT MEMORY ===
[PRIORITY_HOT][PROJECT] ...

[END CARETAKER CONTEXT]
```

### Memory Schema

| Field | Type | Description |
|-------|------|-------------|
| id | TEXT | UUID primary key |
| source_agent | TEXT | Which agent saved this memory |
| keywords | TEXT | JSON array of keywords |
| short | TEXT | Compressed summary (Phase 2) |
| full | TEXT | Full original message |
| type | TEXT | PROJECT / PREFERENCE / PROBLEM etc |
| subtype | TEXT | Lowercase type label |
| fact_type | TEXT | REPLACEABLE or ADDITIVE |
| status | TEXT | ACTIVE / OUTDATED / ARCHIVED |
| importance | REAL | Score 0.0 to 1.0 |
| decay_score | REAL | Starts at 1.0, decays over time |
| temperature | TEXT | PRIORITY_HOT / HOT / WARM / COLD |
| retrieval_count | INTEGER | How many times retrieved |
| created_at | TEXT | ISO timestamp |
| updated_at | TEXT | ISO timestamp |
| last_used | TEXT | ISO timestamp |

---

## Verify Installation

Run this to check memories are saving:

```bash
uv run python -c "
from storage.local_db import get_all_active_memories
mems = get_all_active_memories()
print(f'Total memories: {len(mems)}')
for m in mems:
    print(f'  type={m[\"type\"]} temp={m[\"temperature\"]} importance={m[\"importance\"]}')
    print(f'  full={m[\"full\"][:80]}')
"
```

---

## What Comes Next

| Phase | What it adds |
|-------|-------------|
| Phase 2 | Smart compression, temperature engine, importance scorer, conflict checker, semantic search with ChromaDB |
| Phase 3 | Multi-agent support, CLI tool, cloud backup with Supabase, encryption |
| Phase 4 | Performance hardening, full test suite, production readiness |

---

## Built With

- [FastMCP](https://github.com/jlowin/fastmcp) — MCP server framework
- [SQLite](https://www.sqlite.org/) — Local memory storage
- [uv](https://github.com/astral-sh/uv) — Python package manager
- [Anthropic Claude](https://www.anthropic.com/) — AI agent

---

*Caretaker — Phase 1 complete. Memory cave built. Foundation strong.* 🪨