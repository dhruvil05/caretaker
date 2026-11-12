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
- **Summaries** are made on your device by default (`"compression_model": "local"`: the leading sentences, or
  spaCy + TextRank with the `nlp` extra). Setting `compression_model` to `"claude-haiku-4-5-20251001"` **and**
  an `anthropic_api_key` sends memory text to Anthropic's API for better summaries (secrets are removed first).
- **Cloud backup** only runs if you configure Supabase with an `encrypt_key`, and everything is encrypted on
  your device first.

The dashboard's sidebar and `caretaker doctor` always show whether anything leaves your device.

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

### Make the agent use memory every turn

Caretaker's tools describe themselves to the agent. To make the agent use them on every turn, either pick
Caretaker's built-in prompt **`use_caretaker_memory`** from your app's prompt/attachments menu (MCP prompts;
it takes an optional project name), or paste this into your app's custom instructions —
**Claude Desktop → Settings → General → Custom System Prompt**:

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

Every tool takes an optional `project` argument (see [Projects](#projects)).

To avoid a permission prompt on every turn, set `caretaker_get_context` and `caretaker_save_message` to **Always allow** (Claude Desktop → Settings → Connectors → Caretaker). Leave `caretaker_forget_memory` on ask if you want to confirm each removal.

### See and manage your memories

```bash
caretaker dashboard
```

Opens a local dashboard in your browser. It only listens on `127.0.0.1`, needs the one-time link printed in
the terminal, makes no outside requests, and refreshes live while you look at it.

| Page | What you can do |
|---|---|
| **Overview** | Statistics, growth, the memory workflow, what is recalled most |
| **Memories** | Search, filter, edit, archive/restore; select many for bulk archive, move to a project, change type |
| **Preview** | Type a message and see exactly what an AI would receive from Caretaker, and why each memory was picked (nothing is saved or counted) |
| **Review** | Memories Caretaker replaced with newer facts (undo if it got it wrong) and memories not used for 60 days |
| **Privacy** | What leaves your device, scan stored memories, the review queue, and your privacy settings |
| **Activity** | Recent saves, recent recalls, maintenance runs |
| **Setup** | Health checks and copy-paste config for Claude Desktop, Cursor, VS Code, Windsurf and Claude Code |

The project picker at the top of the sidebar filters every page. The dashboard works on phones and tablets
too, in light and dark mode.

---

## Running the Server

Your MCP client starts the server for you. To run it by hand (e.g. to watch the log):

```bash
caretaker-mcp            # installed
uv run caretaker-mcp     # source checkout
```

It logs to stderr and to `<data folder>/logs/caretaker.log`, starting with the data directory it is using.

---

## Privacy: sensitive data

Everything you tell an AI can become a memory, and memories are recalled into future conversations with
**every** connected AI. So Caretaker checks text on your device **before** it is stored, summarised or indexed:

| Found | Default | Example result |
|---|---|---|
| Secrets — API keys, tokens, passwords, private keys | **Removed** | `I use [REDACTED: Anthropic API key] for Orbit` (the rest of the message is kept) |
| Card numbers, IBAN, SSN, Aadhaar, PAN (checksum-validated) | **Masked** | `[card number ending 4242]` |
| Email addresses, phone numbers | **Kept, flagged for review** | shown in the dashboard's Privacy page |
| Health, beliefs, sexuality, money topics | **Kept, flagged for review** | |

The save confirmation says what happened (e.g. *Memory saved (privacy: 1 Anthropic API key redacted)*). Text
being recalled is checked again before it reaches an AI, and so are imports, cloud restores and dashboard edits.

Change any of this in the dashboard (Privacy → Settings) or in `config.json`:

```json
{
  "privacy": {
    "contact": "redact",
    "custom_patterns": [{"pattern": "ACME-\\d+", "label": "client id"}],
    "allowlist": ["team@example.com"]
  }
}
```

Actions are `redact`, `mask`, `flag` and `allow`. To check memories saved before you upgraded:

```bash
caretaker privacy scan            # report only
caretaker privacy scan --apply    # redact/mask in place
```

Detection is best-effort and runs only on your device. Backups and exports made earlier still contain the
original text — delete them if you don't need them.

---

## Projects

Keep memories from different projects apart. A memory belongs to one project or is global; recall inside a
project sees that project's memories plus global ones. Facts about you (name, where you live, …) are always global.

The project comes from the `project` argument an AI passes to the tools, or from the `CARETAKER_PROJECT`
environment variable — set it per workspace in your app's MCP config, e.g. a project's `.cursor/mcp.json`:

```json
{ "mcpServers": { "caretaker": { "command": "caretaker-mcp", "env": { "CARETAKER_PROJECT": "orbit" } } } }
```

With no project set, recall uses all memories. Move memories between projects in the dashboard (bulk action)
or filter with `caretaker list --project orbit`.

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
caretaker list --project orbit
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
caretaker doctor         # Check your setup, with fixes (--setup prints AI-app config)
caretaker privacy scan   # Find secrets / personal data in stored memories (--apply to fix)
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

## Troubleshooting

**Start with `caretaker doctor`** (or the dashboard's Setup page): it checks the data folder, config,
database, search, privacy settings and which AI apps have Caretaker configured, and says how to fix each issue.

**MCP not showing in Claude Desktop**  
Check `claude_desktop_config.json` paths. All backslashes must be doubled (`\\`). Restart Claude Desktop after any config change.

**ChromaDB index empty after restart**  
Run `caretaker maintenance` to trigger a reindex. Memories still waiting for a summary are also summarised
and re-indexed the next time the server starts.

**Compression not working**  
`caretaker doctor` shows how many memories are waiting for a summary. Summaries are made in the background
while the MCP server runs, and any left over are picked up when it next starts. With Haiku, check
`anthropic_api_key` in `config.json`: if the API fails, Caretaker falls back to the on-device summariser. For
fully offline operation set `"compression_model": "local"`. Until a summary exists, recall uses the memory's
leading sentences.

**Cloud sync failing**  
Run `caretaker sync` and read the error. Most common causes: empty `supabase_url`, wrong `supabase_key`, or `encrypt_key` not set. Cloud sync is optional — system runs fully offline without it.

**`caretaker` command not found**  
Run `uv pip install -e .` from the project root. Then open a new terminal.

---

## Learn more

- [How Caretaker works and how to develop it](docs/DEVELOPMENT.md)
- [Security policy and reporting vulnerabilities](SECURITY.md)
- [Changelog](CHANGELOG.md)

---

## License

MIT — see [LICENSE](LICENSE).

---

*Phase 3 complete. Memory cave now has fire, tools, and tribe support.* 🦣🔨🔥