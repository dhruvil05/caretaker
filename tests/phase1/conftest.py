import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import os

TEST_DB = Path(__file__).parent.parent.parent / "caretaker_test.db"

import storage.local_db as db_module
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