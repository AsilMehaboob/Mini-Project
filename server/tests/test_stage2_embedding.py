"""Tests for Stage 2 – Embedding Classifier."""
import pytest
from memory.classifier.stage_2.embedding import classify


def _top(scores: dict) -> str:
    return max(scores, key=lambda k: scores[k])


def _sums_to_one(scores: dict) -> bool:
    return abs(sum(scores.values()) - 1.0) < 1e-9


class TestClassifyOutputShape:
    def test_returns_three_tiers(self):
        s = classify("Some text.", {"source": "user"})
        assert set(s.keys()) == {"SCRATCH", "SESSION", "LONGTERM"}

    def test_probabilities_sum_to_one(self):
        s = classify("I always prefer dark mode.", {"source": "user"})
        assert _sums_to_one(s)

    def test_all_values_between_zero_and_one(self):
        s = classify("Loading step 3 retry...", {"source": "system"})
        for v in s.values():
            assert 0.0 <= v <= 1.0


class TestClassifyTierBias:
    def test_longterm_for_personal_facts(self):
        s = classify("My name is Jordan and I always use Vim.", {"source": "user"})
        assert _top(s) == "LONGTERM"

    def test_longterm_for_preferences(self):
        s = classify("I prefer TypeScript over JavaScript in general.", {"source": "user"})
        assert _top(s) == "LONGTERM"

    def test_session_for_task_reminders(self):
        s = classify("Remind me to follow up on the open PR assigned to me.", {"source": "user"})
        assert _top(s) == "SESSION"

    def test_scratch_for_status_log(self):
        s = classify("Loading... step 3 iteration 5 retry attempt.", {"source": "system"})
        assert _top(s) == "SCRATCH"

    def test_scratch_for_bare_number(self):
        s = classify("42", {"source": "system"})
        assert _top(s) == "SCRATCH"

    def test_longterm_for_long_content(self):
        # Very long content biases toward LONGTERM
        content = "This is an important fact about the system. " * 20
        s = classify(content, {"source": "user"})
        assert _top(s) == "LONGTERM"
