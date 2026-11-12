"""Phase 2 tests use the default database; make sure its schema exists when
this folder is run on its own (other folders' fixtures used to create it)."""
import pytest


@pytest.fixture(autouse=True, scope="session")
def _phase2_schema():
    from caretaker.storage.local_db import run_migrations
    run_migrations()
