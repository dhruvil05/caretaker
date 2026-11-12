# Changelog

## 0.5.0 — 2026-11-12

### Privacy
- **Sensitive-data protection, on your device.** Before text is stored, summarised, indexed or recalled:
  secrets (API keys, tokens, passwords, private keys, credentials in URLs) are removed; card numbers, IBAN,
  SSN, Aadhaar and PAN are masked (checksum-validated); email addresses and phone numbers are kept and
  flagged for review; health/beliefs/sexuality/money topics are flagged. Configurable per category, with
  custom patterns and an allowlist.
- Recall output, imports, cloud restores, dashboard edits and the background summariser are checked too.
- `caretaker privacy scan [--apply]` checks memories saved before 0.5.
- Dashboard **Privacy** page: what leaves your device, scan + apply, review queue, settings.

### Bigger memories
- Keyword search uses a SQLite full-text index (BM25) instead of loading every memory; recall accuracy on
  the benchmark went from 33% to 100% by searching the message's own words, not only extracted keywords.
- Recall only counts and reheats memories that really match (`min_relevance`), in one database transaction.
- Duplicate check compares each recently changed memory with its closest keyword neighbours (incremental by
  default, `dedup_lookback_days`), never across projects.
- Automatic replacement of outdated facts now actually works on stored memories (it silently never ran
  before), stays within a project, and records what replaced what so it can be undone.
- `scripts/benchmark.py` measures speed and accuracy at any size.

### Summaries
- Memories now get a short summary on the default install too. Before, the background summariser only
  ran when semantic search (`[semantic]`) was installed.
- Memories still waiting for a summary (the job was lost when the AI app closed, or they came from
  `caretaker import`) are picked up when the server starts.
- Until its summary exists, a memory is recalled with its leading sentences instead of being cut short
  or, for cold memories, left out. Recalled summaries are no longer cut to 80 characters.
- Fixed: a failing summary retried forever instead of three times.

### Projects
- Optional `project` on every tool, or `CARETAKER_PROJECT` per workspace. Recall in a project sees its own
  and global memories; personal facts are always global. `caretaker list --project`, dashboard switcher.

### Dashboard
- New pages: **Preview** (what an AI would receive, and why), **Review** (replaced memories with undo, unused
  memories), **Privacy**, **Setup** (health checks + copy-paste config for Claude Desktop, Cursor, VS Code,
  Windsurf, Claude Code). Bulk actions, project switcher, privacy badges.
- Redesigned layout: sidebar navigation with counts of what needs attention, project picker at the top,
  memory list and details side by side (slide-over panel on smaller screens), sub-tabs and "show more"
  instead of long pages, and long text, links and paths wrap instead of breaking the layout. Works on
  phones, tablets and desktops, in light and dark mode.
- The sidebar shows whether any data leaves the device, using the same rule as the code that sends it
  (summaries use Anthropic's API only with a model *and* an API key; cloud backup needs an encryption key).

### Setup and reliability
- `caretaker doctor [--setup]`: checks the installation, detects AI apps (including the Microsoft Store
  build of Claude Desktop) and prints config snippets. It warns when Supabase is set without an
  encryption key (backup stays off).
- The source package no longer picks up local files such as `.claude/settings.local.json`.
- Built-in MCP prompt `use_caretaker_memory`, so users don't have to write a system prompt.
- Versioned database migrations with an automatic backup before upgrading.
- WAL mode and lock timeouts so the MCP server, dashboard and CLI can use the database at the same time.
- The CLI no longer writes to ChromaDB while the server may be using it; nightly maintenance re-indexes.
- Fixed: `VectorDB.get/add/update` were missing, so nightly reindex, restore, edit and import never updated
  the search index; duplicate merging didn't know which memory was newer.

### Docs
- README is now a user guide; internals moved to `docs/DEVELOPMENT.md`. Added `SECURITY.md` and this file.

## 0.4.0

- Local dashboard (`caretaker dashboard`) with live refresh.
- New MCP tools: `caretaker_search_memories`, `caretaker_list_memories`, `caretaker_forget_memory`;
  neutral tool instructions.
- Installable package `caretaker-mcp` with `caretaker-mcp` / `caretaker serve`; per-user data folder
  (`~/.caretaker`) and settings from environment variables.
- Light core install with `semantic`, `nlp`, `anthropic`, `cloud`, `all` extras. MIT license. CI.
- Security: memories framed as notes not instructions, encryption v2 with AAD, hardened cloud restore,
  dependency upgrades.
