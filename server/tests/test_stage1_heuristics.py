"""Tests for Stage 1 - Deterministic Heuristics."""
import pytest
from memory.classifier.stage_1.heuristics import (
    run_heuristics,
    detect_sensitive_patterns,
    detect_temporal_patterns,
)


def _run(content: str, metadata: dict) -> tuple[bool, str]:
    """Run Stage 1 and return (result, human-readable summary)."""
    result = run_heuristics(content, metadata)
    verdict = "PASS" if result else "REJECT"
    preview = repr(content[:60] + "..." if len(content) > 60 else content)
    return result, f"Stage 1 [{verdict}]  input={preview}  meta={metadata}"


# ---------------------------------------------------------------------------
# run_heuristics - acceptance cases
# ---------------------------------------------------------------------------

class TestRunHeuristicsAccept:
    def test_plain_fact(self):
        result, summary = _run("I prefer Python for backend services.", {"source": "user"})
        print(f"\n  {summary}")
        assert result

    def test_medium_length(self):
        result, summary = _run("The capital of France is Paris.", {"source": "user"})
        print(f"\n  {summary}")
        assert result

    def test_born_exception_passes(self):
        result, summary = _run("I was born in 1990.", {"source": "user"})
        print(f"\n  {summary}")
        assert result

    def test_minimal_valid(self):
        result, summary = _run("abc", {"source": "user"})
        print(f"\n  {summary}")
        assert result

    def test_extra_metadata_fields_allowed(self):
        result, summary = _run("Some content.", {"source": "agent", "tags": ["x"]})
        print(f"\n  {summary}")
        assert result


# ---------------------------------------------------------------------------
# run_heuristics - rejection cases
# ---------------------------------------------------------------------------

class TestRunHeuristicsReject:
    def test_empty_string(self):
        result, summary = _run("", {"source": "user"})
        print(f"\n  {summary}  reason=empty string")
        assert not result

    def test_whitespace_only(self):
        result, summary = _run("   ", {"source": "user"})
        print(f"\n  {summary}  reason=whitespace only")
        assert not result

    def test_too_short(self):
        result, summary = _run("ab", {"source": "user"})
        print(f"\n  {summary}  reason=content too short (< 3 chars)")
        assert not result

    def test_too_long(self):
        result, summary = _run("x" * 10_001, {"source": "user"})
        print(f"\n  {summary}  reason=content too long (> 10 000 chars)")
        assert not result

    def test_missing_metadata(self):
        result, summary = _run("Some content.", None)
        print(f"\n  {summary}  reason=metadata is None")
        assert not result

    def test_missing_source_field(self):
        result, summary = _run("Some content.", {"tags": ["x"]})
        print(f"\n  {summary}  reason=missing required field 'source'")
        assert not result

    def test_temporal_immediate(self):
        content = "I am at the moment busy with this task right now."
        result, summary = _run(content, {"source": "user"})
        print(f"\n  {summary}  reason=temporal (immediate + right now, score >= 3)")
        assert not result

    def test_temporal_future_intent(self):
        content = "I will go to the store. I'm going to buy milk."
        result, summary = _run(content, {"source": "user"})
        print(f"\n  {summary}  reason=temporal (future intent x2, score >= 3)")
        assert not result

    def test_sensitive_api_key(self):
        content = "My secret token is sk-abc123def456ghi789jkl012"
        result, summary = _run(content, {"source": "user"})
        print(f"\n  {summary}  reason=sensitive pattern: api_key")
        assert not result

    def test_sensitive_email_pass_combo(self):
        content = "Login: user@example.com:password123"
        result, summary = _run(content, {"source": "user"})
        print(f"\n  {summary}  reason=sensitive pattern: email_pass_combo")
        assert not result

    def test_sensitive_credit_card(self):
        content = "Card number: 4111 1111 1111 1111"
        result, summary = _run(content, {"source": "user"})
        print(f"\n  {summary}  reason=sensitive pattern: credit_card")
        assert not result


# ---------------------------------------------------------------------------
# detect_sensitive_patterns
# ---------------------------------------------------------------------------

class TestDetectSensitivePatterns:
    def test_api_key(self):
        text = "sk-abc123def456ghi789jkl012mno345pqr678stu"
        result = detect_sensitive_patterns(text)
        print(f"\n  detect_sensitive_patterns({text!r}) -> {result}  (api_key pattern)")
        assert result

    def test_uuid(self):
        text = "id: 550e8400-e29b-41d4-a716-446655440000"
        result = detect_sensitive_patterns(text)
        print(f"\n  detect_sensitive_patterns({text!r}) -> {result}  (numeric_id/UUID pattern)")
        assert result

    def test_clean_text(self):
        text = "I enjoy hiking on weekends."
        result = detect_sensitive_patterns(text)
        print(f"\n  detect_sensitive_patterns({text!r}) -> {result}  (no pattern matched)")
        assert not result


# ---------------------------------------------------------------------------
# detect_temporal_patterns
# ---------------------------------------------------------------------------

class TestDetectTemporalPatterns:
    def test_question_mark(self):
        text = "What time is it?"
        result = detect_temporal_patterns(text)
        print(f"\n  detect_temporal_patterns({text!r}) -> {result}  (ends with '?')")
        assert result

    def test_immediately_temporary(self):
        text = "I am busy for now and currently working."
        result = detect_temporal_patterns(text)
        print(f"\n  detect_temporal_patterns({text!r}) -> {result}  (for now + currently, score >= 3)")
        assert result

    def test_permanent_fact_exception(self):
        text = "I live in Berlin since 2010."
        result = detect_temporal_patterns(text)
        print(f"\n  detect_temporal_patterns({text!r}) -> {result}  (permanent fact exception: 'i live in')")
        assert not result

    def test_clean_statement(self):
        text = "Python is a dynamically typed language."
        result = detect_temporal_patterns(text)
        print(f"\n  detect_temporal_patterns({text!r}) -> {result}  (no pattern matched)")
        assert not result
