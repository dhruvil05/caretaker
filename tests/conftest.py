import importlib.util
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'tests' / 'phase1'))

# ── Keep tests out of the real data directory ────────────────────────────────
# Runs before any test module imports `caretaker`. paths.py is loaded on its
# own (without the package __init__) to find the REAL model cache, so already
# downloaded models are reused; everything else (database, vector index, logs,
# exports) goes to a throwaway folder for this test session.
_paths_file = Path(__file__).parent.parent / "src" / "caretaker" / "paths.py"
_spec = importlib.util.spec_from_file_location("_caretaker_paths_bootstrap", _paths_file)
_paths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paths)

_paths.configure_cache_env()   # HF_HOME etc. → real <data_dir>/cache (setdefault)
os.environ["CARETAKER_DATA_DIR"] = tempfile.mkdtemp(prefix="caretaker-test-data-")
