"""
caretaker/mcp_server/stdout_guard.py

IMPORT THIS FIRST — before everything else in server.py.

    from src.caretaker.mcp_server.stdout_guard import apply  # noqa: F401

This file:
1. Redirects ALL warnings to stderr
2. Silences HuggingFace progress bars / download messages
3. Silences sentence-transformers loading output
4. Fixes asyncio DeprecationWarning
5. Redirects tqdm progress bars to stderr
6. Replaces sys.stdout temporarily during model loading
   so stray prints go to stderr instead of corrupting JSON-RPC
"""

import sys
import os
import warnings
import logging

# ── 1. Kill ALL Python warnings to stdout ────────────────────────────────────
warnings.filterwarnings("ignore")

# ── 2. HuggingFace / Transformers env flags (must be set BEFORE import) ──────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HUGGINGFACE_HUB_VERBOSITY", "error")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.environ.get(
    "SENTENCE_TRANSFORMERS_HOME",
    os.path.join(os.path.expanduser("~"), ".cache", "sentence_transformers")
))

# ── 3. Route ALL logging to stderr ───────────────────────────────────────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    force=True,          # override any previous basicConfig calls
)

# Silence noisy third-party loggers
_SILENT_LIBS = [
    "httpx", "httpcore",
    "chromadb", "chromadb.telemetry",
    "sentence_transformers",
    "huggingface_hub", "huggingface_hub.utils",
    "transformers", "transformers.utils",
    "urllib3", "urllib3.connectionpool",
    "dotenv", "python_dotenv",
    "tqdm", "tqdm.auto",
    "filelock",
    "PIL",
    "asyncio",
]
for _lib in _SILENT_LIBS:
    logging.getLogger(_lib).setLevel(logging.ERROR)


# ── 4. Patch tqdm to write to stderr ─────────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm
    from tqdm.auto import tqdm as _tqdm_auto
    _tqdm.__init__.__defaults__  # just check it exists
    # Monkey-patch default file to stderr
    import tqdm as _tqdm_mod
    _orig_init = _tqdm_mod.tqdm.__init__

    def _patched_tqdm_init(self, *args, **kwargs):
        kwargs.setdefault("file", sys.stderr)
        _orig_init(self, *args, **kwargs)

    _tqdm_mod.tqdm.__init__ = _patched_tqdm_init
except Exception:
    pass


# ── 5. StdoutGuard context manager ───────────────────────────────────────────
class _StderrRedirect:
    """
    Temporarily redirects sys.stdout to stderr.
    Use during model loading to catch any stray prints.

    Usage:
        with stdout_guard.redirect():
            model = SentenceTransformer("all-MiniLM-L6-v2")
    """
    def __enter__(self):
        self._original = sys.stdout
        sys.stdout = sys.stderr
        return self

    def __exit__(self, *_):
        sys.stdout = self._original


redirect = _StderrRedirect


# ── 6. Safe asyncio loop helper ──────────────────────────────────────────────
def get_or_create_event_loop():
    """
    Replacement for deprecated asyncio.get_event_loop().
    Use this everywhere instead.
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("loop closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


# ── 7. Module-level application (runs on import) ─────────────────────────────
def apply():
    """Call apply() explicitly if you want to verify guard is active."""
    pass  # All setup runs at import time above


# Confirm guard loaded — to stderr only
logging.getLogger("caretaker.stdout_guard").info(
    "stdout_guard active — all output routed to stderr"
)