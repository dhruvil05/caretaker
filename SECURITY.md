# Security policy

Caretaker stores what people tell their AI assistants, so security and privacy
reports are taken seriously.

## Reporting a vulnerability

Please **don't open a public issue** for security problems. Use GitHub's private
reporting instead: open the repository's **Security** tab and choose
**Report a vulnerability**. Include what you found, how to reproduce it, and
which version you used (`caretaker --version` or `pip show caretaker-mcp`).

You should get a first reply within a week. Please give us a reasonable time to
release a fix before disclosing it publicly.

## Supported versions

Security fixes go into the latest released minor version.

| Version | Supported |
|---|---|
| 0.5.x | Yes |
| < 0.5 | No — please upgrade |

## How Caretaker protects data (and where it doesn't)

**What it does**
- Everything is stored on your device (`~/.caretaker`, or `data_dir`). Nothing leaves it unless you turn on
  Haiku summaries (memory text goes to Anthropic's API) or Supabase backup.
- Sensitive data is checked **on your device** before it is stored, summarised, indexed or recalled:
  secrets are removed, financial/ID numbers masked, contact details flagged (configurable).
- Recalled memories are presented to the AI as background notes, not instructions, and are sanitised so a
  stored memory can't break out of its block in the prompt.
- Cloud backups are encrypted on your device with AES-256-GCM (PBKDF2-HMAC-SHA256, 600,000 iterations),
  each field bound to its memory id. Restores reject unencrypted or tampered rows.
- The dashboard only listens on `127.0.0.1`, rejects other Host headers, needs a random per-run token, sends a
  strict Content-Security-Policy, and never renders memory text as HTML.

**What it doesn't do**
- **Sensitive-data detection is best-effort.** It uses patterns and checksums; it will miss some things and
  can't understand context. Don't rely on it as your only protection for highly sensitive data.
- **Data on disk is not encrypted by Caretaker.** Use your operating system's disk encryption
  (BitLocker, FileVault, LUKS). Database backups (`data/backups/`) and exports are plain files too.
- **Anything on your computer that runs as you can read your memories**, including other programs and MCP
  servers. The dashboard token protects against web pages and other users, not against your own processes.
- **Supabase:** turn on Row Level Security for the `memories` table and keep the `service_role` key only in
  your local config.
