"""
caretaker/mcp_server/stdout_guard.py

IMPORT THIS FIRST — before everything else in server.py.

    from caretaker.mcp_server.stdout_guard import apply  # noqa: F401

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
# Model cache locations (HF_HOME, SENTENCE_TRANSFORMERS_HOME, TORCH_HOME) are
# set to <data_dir>/cache by caretaker/__init__.py → paths.configure_cache_env().

# ── 3. Route ALL logging to stderr, via a teardown-safe handler ─────────────
class _SafeStreamHandler(logging.StreamHandler):
    """
    Identical to logging.StreamHandler, except it silently drops write
    failures that happen because the underlying stream (sys.stderr) has
    already been closed by the OS.

    This happens legitimately during MCP stdio shutdown: when Claude
    Desktop closes the process's pipes, any thread — including a daemon
    background thread still finishing up async work — that tries to log
    afterward hits `ValueError: I/O operation on closed file`. Normally
    logging.Handler.handleError() re-raises during interpreter teardown,
    which Python can't process cleanly and dumps as a raw unraisable-
    exception trace ("lost sys.stderr"). Swallowing this specific,
    already-torn-down-pipe case is safe: there is no reader left to see
    the log line anyway, so dropping it loses nothing observable.
    """
    def emit(self, record):
        try:
            super().emit(record)
        except (ValueError, OSError):
            # Stream already closed (pipe torn down by parent process).
            # Nothing useful can be done — drop silently instead of
            # crashing interpreter finalization.
            pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    force=True,          # override any previous basicConfig calls
    handlers=[_SafeStreamHandler(sys.stderr)],
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
        # If the permanent proxy is installed (MCP server mode), text writes
        # already go to stderr. Swapping sys.stdout here would be actively
        # harmful: another thread (the MCP stdio transport) may read
        # sys.stdout.buffer at that moment and bind to stderr for good.
        self._swapped = not isinstance(sys.stdout, _StdoutTextToStderr)
        if self._swapped:
            self._original = sys.stdout
            sys.stdout = sys.stderr
        return self

    def __exit__(self, *_):
        if self._swapped:
            sys.stdout = self._original


redirect = _StderrRedirect


# ── 5b. Permanent, thread-safe stdout proxy (MCP server mode only) ───────────
class _StdoutTextToStderr:
    """
    Permanent replacement for sys.stdout in the MCP server process.

    - Text writes (print(), library chatter) from ANY thread go to stderr.
    - `.buffer` still returns the REAL stdout's binary buffer, which is what
      the MCP stdio transport binds to (mcp.server.stdio wraps
      sys.stdout.buffer), so JSON-RPC frames still reach the client.

    Unlike redirect(), this never swaps sys.stdout while other threads are
    running, so there is no window in which the transport can accidentally
    bind to stderr.
    """
    def __init__(self, real_stdout):
        self._real = real_stdout

    def write(self, s):
        try:
            return sys.stderr.write(s)
        except (ValueError, OSError):
            return 0

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        try:
            sys.stderr.flush()
        except (ValueError, OSError):
            pass

    @property
    def buffer(self):
        return self._real.buffer

    def __getattr__(self, name):
        return getattr(self._real, name)


def install_stdout_proxy():
    """
    Install the permanent stdout proxy. Call once from the MCP server's
    entry point on the main thread, BEFORE starting any background threads.
    Not applied at import time, so CLI commands that import this module
    keep normal stdout behaviour.
    """
    if not isinstance(sys.stdout, _StdoutTextToStderr):
        sys.stdout = _StdoutTextToStderr(sys.stdout)


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