import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from capture.entity_extractor import extract_entities
from caretaker.tests.fixtures.fixtures import SAMPLE_MESSAGES


class TestEntityExtractor:

    def test_returns_required_keys(self):
        result = extract_entities(SAMPLE_MESSAGES["project"])
        assert "tools"     in result
        assert "emotions"  in result
        assert "decisions" in result
        assert "problems"  in result
        assert "learnings" in result
        assert "keywords"  in result

    def test_detects_tools_in_message(self):
        result = extract_entities("I use Python and SQLite for my project.")
        assert "python" in result["tools"]
        assert "sqlite" in result["tools"]

    def test_detects_emotions_in_message(self):
        result = extract_entities("I am frustrated and confused about this error.")
        assert "frustrated" in result["emotions"]
        assert "confused"   in result["emotions"]

    def test_detects_decisions_in_message(self):
        result = extract_entities("I decided to go with FastMCP for the server.")
        assert len(result["decisions"]) > 0

    def test_detects_problems_in_message(self):
        result = extract_entities("There is a bug causing a crash in my code.")
        assert len(result["problems"]) > 0

    def test_detects_learnings_in_message(self):
        result = extract_entities("I am learning about vector databases.")
        assert len(result["learnings"]) > 0

    def test_keywords_max_seven(self):
        result = extract_entities(SAMPLE_MESSAGES["project"])
        assert len(result["keywords"]) <= 7

    def test_keywords_is_list(self):
        result = extract_entities(SAMPLE_MESSAGES["project"])
        assert isinstance(result["keywords"], list)

    def test_empty_message_no_crash(self):
        result = extract_entities("")
        assert isinstance(result, dict)
        assert "keywords" in result

    def test_short_message_no_crash(self):
        result = extract_entities("Hi")
        assert isinstance(result, dict)

    def test_multiple_tools_detected(self):
        result = extract_entities("I use FastAPI with SQLite and ChromaDB.")
        assert "fastapi" in result["tools"]
        assert "sqlite"  in result["tools"]

    def test_keywords_no_stop_words(self):
        result = extract_entities("This is that with have from they will been")
        stop_words = {"this", "that", "with", "have", "from", "they", "will", "been"}
        for kw in result["keywords"]:
            assert kw not in stop_words

    def test_long_message_no_crash(self):
        result = extract_entities(SAMPLE_MESSAGES["long"])
        assert isinstance(result, dict)
