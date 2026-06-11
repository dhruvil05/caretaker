"""
Run this from your caretaker root:
    python audit_stdout.py

It finds every print() and warning that can poison MCP stdio.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

PATTERNS = [
    (r'\bprint\s*\(', "PRINT TO STDOUT"),
    (r'warnings\.warn\b', "warnings.warn (may go stdout)"),
    (r'asyncio\.get_event_loop\(\)', "DEPRECATED get_event_loop (prints DeprecationWarning)"),
    (r'load_dotenv\((?![^)]*verbose\s*=\s*False)', "dotenv load without verbose=False"),
    (r'HfFolder|huggingface_hub|from_pretrained', "HuggingFace (prints progress to stdout)"),
]

SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'site-packages'}
SKIP_FILES = {'audit_stdout.py'}

found = []

for dirpath, dirnames, filenames in os.walk(ROOT):
    # prune
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fname in filenames:
        if not fname.endswith('.py'):
            continue
        if fname in SKIP_FILES:
            continue
        fpath = os.path.join(dirpath, fname)
        rel = os.path.relpath(fpath, ROOT)
        try:
            lines = open(fpath, encoding='utf-8', errors='replace').readlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            for pattern, label in PATTERNS:
                if re.search(pattern, line):
                    found.append((rel, i, label, stripped))

if not found:
    print("CLEAN — no stdout poisoners found!")
else:
    print(f"\n{'='*60}")
    print(f"FOUND {len(found)} STDOUT POISONERS — FIX ALL OF THESE")
    print(f"{'='*60}\n")
    current_file = None
    for rel, lineno, label, code in found:
        if rel != current_file:
            print(f"\n>>> {rel}")
            current_file = rel
        print(f"  Line {lineno:4d}  [{label}]")
        print(f"           {code}")
    print(f"\n{'='*60}")
    print("Fix: replace print() with logger.debug(). See fix_guide below.")

print("""
=== FIX GUIDE ===

1. REPLACE ALL print() in mcp path:
   BAD:  print("[CARETAKER] server started")
   GOOD: logger.debug("server started")   # goes to stderr only

2. FIX asyncio warning in server.py:
   BAD:  loop = asyncio.get_event_loop()
   GOOD: loop = asyncio.new_event_loop()
         asyncio.set_event_loop(loop)

3. FIX HuggingFace progress bars:
   Set env var before importing sentence_transformers:
   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   os.environ["TRANSFORMERS_VERBOSITY"] = "error"
   os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

4. FIX dotenv:
   load_dotenv(verbose=False, override=True)

5. ADD THIS as very first lines of server.py (before all imports):
   import sys, os, warnings, logging
   warnings.filterwarnings("ignore")
   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   os.environ["TRANSFORMERS_VERBOSITY"] = "error"
   os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
   os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
   logging.basicConfig(stream=sys.stderr, level=logging.INFO)
   # Silence noisy libs
   for lib in ["httpx","chromadb","sentence_transformers","huggingface_hub",
               "transformers","urllib3","dotenv"]:
       logging.getLogger(lib).setLevel(logging.ERROR)
""")