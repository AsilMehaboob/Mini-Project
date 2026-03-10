"""Tests for Stage 3 – Confidence Threshold."""
import pytest
from memory.classifier.stage_3.threshold import check_confidence, DEFAULT_THRESHOLD


class TestCheckConfidence:
    def test_confident_when_above_threshold(self):
        scores = {"SCRATCH": 0.05, "SESSION": 0.05, "LONGTERM": 0.90}
        tier, confident = check_confidence(scores)
        assert confident is True
        assert tier == "LONGTERM"

    def test_not_confident_when_below_threshold(self):
        scores = {"SCRATCH": 0.35, "SESSION": 0.35, "LONGTERM": 0.30}
        tier, confident = check_confidence(scores)
        assert confident is False

    def test_exactly_at_threshold_is_confident(self):
        scores = {"SCRATCH": 0.175, "SESSION": 0.175, "LONGTERM": DEFAULT_THRESHOLD}
        tier, confident = check_confidence(scores, threshold=DEFAULT_THRESHOLD)
        assert confident is True
        assert tier == "LONGTERM"

    def test_custom_threshold(self):
        scores = {"SCRATCH": 0.10, "SESSION": 0.20, "LONGTERM": 0.70}
        _, confident_high = check_confidence(scores, threshold=0.80)
        _, confident_low  = check_confidence(scores, threshold=0.60)
        assert confident_high is False
        assert confident_low  is True

    def test_returns_top_tier_regardless(self):
        scores = {"SCRATCH": 0.40, "SESSION": 0.35, "LONGTERM": 0.25}
        tier, _ = check_confidence(scores)
        assert tier == "SCRATCH"

    def test_empty_scores_returns_none(self):
        tier, confident = check_confidence({})
        assert tier is None
        assert confident is False
