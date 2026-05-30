import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from capture.type_classifier import classify_type
from caretaker.tests.fixtures.fixtures import (
    SAMPLE_MESSAGES,
    EXPECTED_TYPES,
    EXPECTED_FACT_TYPES,
    VALID_TYPES,
    VALID_FACT_TYPES,
)


class TestTypeClassifier:

    def test_returns_required_keys(self):
        result = classify_type(SAMPLE_MESSAGES["project"])
        assert "type"       in result
        assert "subtype"    in result
        assert "fact_type"  in result
        assert "importance" in result

    def test_classifies_project_message(self):
        result = classify_type(SAMPLE_MESSAGES["project"])
        assert result["type"] == "PROJECT"

    def test_classifies_preference_message(self):
        result = classify_type(SAMPLE_MESSAGES["preference"])
        assert result["type"] == "PREFERENCE"

    def test_classifies_problem_message(self):
        result = classify_type(SAMPLE_MESSAGES["problem"])
        assert result["type"] == "PROBLEM"

    def test_classifies_decision_message(self):
        result = classify_type(SAMPLE_MESSAGES["decision"])
        assert result["type"] == "DECISION"

    def test_classifies_learning_message(self):
        result = classify_type(SAMPLE_MESSAGES["learning"])
        assert result["type"] == "LEARNING"

    def test_classifies_personal_message(self):
        result = classify_type(SAMPLE_MESSAGES["personal"])
        assert result["type"] == "PERSONAL"

    def test_classifies_emotion_message(self):
        # Phase 2 note: fixture updated to pure emotion (no problem-domain words).
        # Original "I am frustrated with this error" scored PROBLEM due to "error" signal.
        result = classify_type(SAMPLE_MESSAGES["emotion"])
        assert result["type"] == "EMOTION"

    def test_classifies_correction_message(self):
        result = classify_type(SAMPLE_MESSAGES["correction"])
        assert result["type"] == "CORRECTION"

    def test_type_always_valid(self):
        # Phase 2 added SKIP type for questions/noise. VALID_TYPES includes SKIP.
        for key, msg in SAMPLE_MESSAGES.items():
            result = classify_type(msg)
            assert result["type"] in VALID_TYPES, (
                f"Message '{key}' returned invalid type '{result['type']}'"
            )

    def test_fact_type_always_valid(self):
        # Phase 2 added fact_type = "SKIP" for noise/question messages.
        # VALID_FACT_TYPES includes SKIP.
        for key, msg in SAMPLE_MESSAGES.items():
            result = classify_type(msg)
            assert result["fact_type"] in VALID_FACT_TYPES, (
                f"Message '{key}' returned invalid fact_type '{result['fact_type']}'"
            )

    def test_replaceable_types_correct(self):
        for key in ["project", "preference", "personal", "correction"]:
            result = classify_type(SAMPLE_MESSAGES[key])
            assert result["fact_type"] == "REPLACEABLE", f"Failed for {key}"

    def test_additive_types_correct(self):
        for key in ["problem", "decision", "learning", "emotion"]:
            result = classify_type(SAMPLE_MESSAGES[key])
            assert result["fact_type"] == "ADDITIVE", f"Failed for {key}"

    def test_subtype_is_lowercase_of_type(self):
        # Phase 2 note: SKIP type returns subtype = "noise" (not "skip").
        # For all other types, subtype == type.lower() still holds.
        for key, msg in SAMPLE_MESSAGES.items():
            result = classify_type(msg)
            if result["type"] == "SKIP":
                assert result["subtype"] == "noise", (
                    f"SKIP type should have subtype 'noise', got '{result['subtype']}'"
                )
            else:
                assert result["subtype"] == result["type"].lower(), (
                    f"Message '{key}': subtype '{result['subtype']}' "
                    f"!= type.lower() '{result['type'].lower()}'"
                )

    def test_importance_between_zero_and_one(self):
        for key, msg in SAMPLE_MESSAGES.items():
            result = classify_type(msg)
            assert 0.0 <= result["importance"] <= 1.0, f"Failed for {key}"

    def test_importance_is_float(self):
        result = classify_type(SAMPLE_MESSAGES["project"])
        assert isinstance(result["importance"], float)

    def test_empty_message_defaults_to_skip(self):
        # Phase 2: empty string < MIN_USEFUL_LENGTH (20) → SKIP
        result = classify_type("")
        assert result["type"] == "SKIP"

    def test_short_unknown_message_returns_skip(self):
        # Phase 2: short message < MIN_USEFUL_LENGTH → SKIP
        result = classify_type("blah blah xyz abc")
        assert result["type"] == "SKIP"

    def test_long_unknown_message_defaults_to_learning(self):
        # Phase 1 default preserved for long messages with no signals → LEARNING
        result = classify_type(
            "xyzzy foobar baz qux quux corge grault garply waldo fred plugh"
        )
        assert result["type"] == "LEARNING"