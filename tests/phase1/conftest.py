import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
# Insert the src/ layout dir (not the project root) so `import caretaker`
# resolves to src/caretaker. The project root has a stray __init__.py that
# would otherwise shadow the real package.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
import os

TEST_DB = Path(__file__).parent.parent.parent / "caretaker_test.db"

import caretaker.storage.local_db as db_module
db_module.DB_PATH = TEST_DB

@pytest.fixture(autouse=True, scope="function")
def clean_db():
    if TEST_DB.exists():
        try:
            os.remove(TEST_DB)
        except PermissionError:
            pass
    db_module.run_migrations()
    yield
    if TEST_DB.exists():
        try:
            os.remove(TEST_DB)
        except PermissionError:
            pass