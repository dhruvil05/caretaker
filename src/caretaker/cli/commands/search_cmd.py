"""
cli/commands/search_cmd.py
Phase 3 — caretaker search command.

Uses semantic search (ChromaDB) when available.
Falls back to SQLite keyword search if ChromaDB not running.

Usage:
    caretaker search "python project"
    caretaker search "editor preferences" --limit 5
"""

import click
from caretaker.cli.formatters import (
    format_list_header, format_search_result,
    print_info, print_warning, print_error,
    GREY, RESET, BOLD, CYAN
)


@click.command("search")
@click.argument("query")
@click.option("--limit", default=10, help="Max results to return (default 10)")
def search_cmd(query, limit):
    """Semantic search through memories. Shows ranked results."""

    # ── Try semantic search via ChromaDB ──────────────────────────────────
    semantic_results = None
    semantic_error   = None
    search_mode      = "keyword"   # default — overridden if semantic succeeds

    try:
        from caretaker.storage.vector_db import VectorDB
        from caretaker.retrieval.semantic_searcher import SemanticSearcher
        from pathlib import Path
        import json

        config_path = Path(__file__).parent.parent.parent.parent.parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        chromadb_path = config.get("database", {}).get("chromadb_path", "data/chromadb")
        vector_db = VectorDB(persist_directory=chromadb_path)
        vector_db.initialize()

        searcher = SemanticSearcher(vector_db=vector_db, config=config)
        raw_results = searcher.search(query=query, n_results=limit)

        semantic_results = raw_results
        search_mode      = "semantic"

    except Exception as e:
        semantic_error = e
        search_mode    = "keyword"

    # ── Fall back to SQLite keyword search ─────────────────────────────────
    if semantic_results is None:
        from caretaker.storage.local_db import search_memories_by_keyword
        results = search_memories_by_keyword(query, limit=limit)
        if semantic_error:
            print_warning(f"ChromaDB unavailable — using keyword search. ({semantic_error})")
    else:
        results = semantic_results

    if not results:
        print_info(f"No memories found for: '{query}'")
        return

    # ── Print results ──────────────────────────────────────────────────────
    mode_label = CYAN + f"[{search_mode}]" + RESET
    print(f"\n{BOLD}Search results for:{RESET} \"{query}\"  {mode_label}")
    print(format_list_header())

    for rank, mem in enumerate(results, start=1):
        score = mem.get("_score")
        print(format_search_result(mem, rank=rank, score=score))

    print(f"\n{GREY}{len(results)} result(s) returned.{RESET}\n")