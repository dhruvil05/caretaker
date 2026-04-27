import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import json
from capture.capture_engine import run_capture, get_temperature, count_tokens_approx
from caretaker.tests.fixtures.fixtures import SAMPLE_MESSAGES, VALID_TEMPERATURES, VALID_TYPES, VALID_FACT_TYPES


class TestCaptureEngine:

    def test_returns_memory_dict(self):
        mem = run_capture(SAMPLE_MESSAGES["project"])
        assert isinstance(mem, dict)

    def test_memory_has_required_fields(self):
        mem = run_capture(SAMPLE_MESSAGES["project"])
        required = [
            "id", "source_agent", "keywords", "short", "full",
            "type", "subtype", "fact_type", "status", "superseded_by",
            "importance", "decay_score", "temperature", "retrieval_count",
            "created_at", "updated_at", "last_used"
        ]
        for field in required:
            assert field in mem, f"Missing field: {field}"

    def test_memory_id_is_uuid_string(self):
        mem = run_capture(SAMPLE_MESSAGES["project"])
        assert isinstance(mem["id"], str)
        assert len(mem["id"]) == 36
        assert mem["id"].count("-") == 4

    def test_default_agent_is_claude(self):
        mem = run_capture(SAMPLE_MESSAGES["project"])
        assert mem["source_agent"] == "claude"

    def test_custom_agent_id_saved(self):
        mem = run_capture(SAMPLE_MESSAGES["project"], agent_id="gpt4")
        assert mem["source_agent"] == "gpt4"

    def test_status_is_active(self):
        mem = run_capture(SAMPLE_MESSAGES["project"])
        assert mem["status"] == "ACTIVE"

    def test_decay_score_starts_at_one(self):
        mem = run_capture(SAMPLE_MESSAGES["project"])
        assert mem["decay_score"] == 1.0

    def test_retrieval_count_starts_at_zero(self):
        mem = run_capture(SAMPLE_MESSAGES["project"])
        assert mem["retrieval_count"] == 0

    def test_temperature_is_valid(self):
        mem = run_capture(SAMPLE_MESSAGES["project"])
        assert mem["temperature"] in VALID_TEMPERATURES

    def test_type_is_valid(self):
        mem = run_capture(SAMPLE_MESSAGES["project"])
        assert mem["type"] in VALID_TYPES

    def test_fact_type_is_valid(self):
        mem = run_capture(SAMPLE_MESSAGES["project"])
        assert mem["fact_type"] in VALID_FACT_TYPES

    def test_full_content_saved(self):
        msg = SAMPLE_MESSAGES["project"]
        mem = run_capture(msg)
        assert mem["full"] == msg.strip()

    def test_keywords_is_json_string(self):
        mem = run_capture(SAMPLE_MESSAGES["project"])
        parsed = json.loads(mem["keywords"])
        assert isinstance(parsed, list)

    def test_importance_between_zero_and_one(self):
        mem = run_capture(SAMPLE_MESSAGES["project"])
        assert 0.0 <= mem["importance"] <= 1.0

    def test_timestamps_exist(self):
        mem = run_capture(SAMPLE_MESSAGES["project"])
        assert mem["created_at"] is not None
        assert mem["updated_at"] is not None
        assert mem["last_used"]  is not None

    def test_long_message_handled(self):
        mem = run_capture(SAMPLE_MESSAGES["long"])
        assert mem["full"] is not None
        assert isinstance(mem["full"], str)


class TestGetTemperature:

    def test_priority_hot_at_07(self):
        assert get_temperature(0.7) == "PRIORITY_HOT"

    def test_priority_hot_at_09(self):
        assert get_temperature(0.9) == "PRIORITY_HOT"

    def test_hot_at_05(self):
        assert get_temperature(0.5) == "HOT"

    def test_hot_at_06(self):
        assert get_temperature(0.6) == "HOT"

    def test_warm_at_02(self):
        assert get_temperature(0.2) == "WARM"

    def test_warm_at_04(self):
        assert get_temperature(0.4) == "WARM"

    def test_cold_at_01(self):
        assert get_temperature(0.1) == "COLD"

    def test_cold_at_zero(self):
        assert get_temperature(0.0) == "COLD"


class TestTokenCounter:

    def test_counts_words_approximately(self):
        msg = "hello world this is a test"
        assert count_tokens_approx(msg) == 6

    def test_empty_string_returns_zero(self):
        assert count_tokens_approx("") == 0

    def test_long_message_over_400(self):
        msg = " ".join(["word"] * 450)
        assert count_tokens_approx(msg) == 450
