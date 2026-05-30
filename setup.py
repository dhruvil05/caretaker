"""
setup.py
Phase 3 — Thin shim for editable install compatibility.

All real config lives in pyproject.toml.
This file exists only so that older pip / tools that don't yet
fully support PEP 660 can still do: pip install -e .

Preferred install command:
    uv pip install -e .

This registers the 'caretaker' terminal command via the
[project.scripts] entry point in pyproject.toml:
    caretaker = "cli.main:main"

After install, test with:
    caretaker --help
    caretaker stats
    caretaker list
"""

from setuptools import setup

setup()