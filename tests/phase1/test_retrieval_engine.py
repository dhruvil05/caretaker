import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from retrieval.topic_detector import detect_topic
from retrieval.keyword_extractor import extract_keywords
from retrieval.budget_engine import calculate_budget
from retrieval.retrieval_engine import retrieve_context
from storage.local_db import run_migrations, save_memory
from caretaker.tests.fixtures.fixtures import SAMPLE_MESSAGES, VALID_LEVELS
from capture.capture_engine import run_capture


@pytest.fixture(autouse=True)
def setup_db():
    run_migrations()


class TestTopicDetector:

    def test_returns_required_keys(self):
        result = detect_topic("Hello!")
        assert "level"  in result
        assert "budget" in result

    def test_greeting_is_L0(self):
        result = detect_topic("Hey! How are you?")
        assert result["level"] == "L0"

    def test_simple_question_is_L1(self):
        result = detect_topic("What is Python?")
        assert result["level"] == "L1"

    def test_explain_is_L2(self):
        result = detect_topic("Can you explain how memory works?")
        assert result["level"] == "L2"

    def test_code_is_L3(self):
        result = detect_topic("Help me debug this error in my code.")
        assert result["level"] == "L3"

    def test_architecture_is_L4(self):
        result = detect_topic("Give me the full architecture overview of the system.")
        assert result["level"] == "L4"

    def test_full_context_is_L5(self):
        result = detect_topic("Remember everything and give me full context.")
        assert result["level"] == "L5"

    def test_level_always_valid(self):
        for key, msg in SAMPLE_MESSAGES.items():
            result = detect_topic(msg)
            assert result["level"] in VALID_LEVELS

    def test_budget_is_positive_integer(self):
        result = detect_topic("Hello!")
        assert isinstance(result["budget"], int)
        assert result["budget"] > 0

    def test_higher_level_higher_budget(self):
        l0 = detect_topic("Hi")
        l5 = detect_topic("Remember everything about me full context")
        assert l5["budget"] > l0["budget"]

    def test_unknown_message_defaults_to_L2(self):
        result = detect_topic("something random xyz")
        assert result["level"] == "L2"


class TestKeywordExtractor:

    def test_returns_list(self):
        result = extract_keywords(SAMPLE_MESSAGES["project"])
        assert isinstance(result, list)

    def test_max_seven_keywords(self):
        result = extract_keywords(SAMPLE_MESSAGES["project"])
        assert len(result) <= 7

    def test_no_stop_words(self):
        stop_words = {
            "this", "that", "with", "have", "from", "they",
            "will", "been", "were", "your", "about"
        }
        result = extract_keywords(SAMPLE_MESSAGES["project"])
        for kw in result:
            assert kw not in stop_words

    def test_empty_message_returns_empty_list(self):
        result = extract_keywords("")
        assert result == []

    def test_keywords_are_lowercase(self):
        result = extract_keywords("Building FastAPI Project Caretaker Memory")
        for kw in result:
            assert kw == kw.lower()

    def test_short_words_excluded(self):
        result = extract_keywords("I am a builder of big systems")
        for kw in result:
            assert len(kw) >= 3


class TestBudgetEngine:

    def test_returns_required_keys(self):
        result = calculate_budget("Hello", [])
        assert "level"    in result
        assert "budget"   in result
        assert "use_full" in result

    def test_budget_within_bounds(self):
        for key, msg in SAMPLE_MESSAGES.items():
            result = calculate_budget(msg, [])
            assert 80 <= result["budget"] <= 800

    def test_use_full_false_for_low_budget(self):
        result = calculate_budget("Hi", [])
        if result["budget"] < 480:
            assert result["use_full"] is False

    def test_use_full_true_for_high_budget(self):
        result = calculate_budget("Remember everything full context", [])
        if result["budget"] >= 480:
            assert result["use_full"] is True

    def test_high_importance_memory_increases_budget(self):
        low_mem  = [{"importance": 0.3, "temperature": "WARM"}]
        high_mem = [{"importance": 0.95, "temperature": "PRIORITY_HOT"}]
        msg = "Help me with my project"
        low_budget  = calculate_budget(msg, low_mem)["budget"]
        high_budget = calculate_budget(msg, high_mem)["budget"]
        assert high_budget >= low_budget


class TestRetrievalEngine:

    def test_returns_required_keys(self):
        result = retrieve_context("What project am I building?")
        assert "relevant"  in result
        assert "recent"    in result
        assert "budget"    in result
        assert "use_full"  in result
        assert "level"     in result
        assert "keywords"  in result

    def test_relevant_is_list(self):
        result = retrieve_context("What project am I building?")
        assert isinstance(result["relevant"], list)

    def test_recent_is_list(self):
        result = retrieve_context("What project am I building?")
        assert isinstance(result["recent"], list)

    def test_budget_within_bounds(self):
        result = retrieve_context("What project am I building?")
        assert 80 <= result["budget"] <= 800

    def test_keywords_is_list(self):
        result = retrieve_context("What project am I building?")
        assert isinstance(result["keywords"], list)

    def test_retrieves_relevant_memory_after_save(self):
        run_capture("I am building a FastAPI project called Caretaker.", "claude")
        result = retrieve_context("What project am I building?")
        assert len(result["relevant"]) > 0

    def test_increments_retrieval_count(self):
        from storage.local_db import get_all_active_memories
        run_capture("I prefer Python for all projects.", "claude")
        retrieve_context("What do I prefer?")
        mems = get_all_active_memories()
        counts = [m["retrieval_count"] for m in mems]
        assert any(c > 0 for c in counts)

    def test_empty_db_no_crash(self):
        result = retrieve_context("What project am I building?")
        assert isinstance(result, dict)
