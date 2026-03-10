"""Shared pytest fixtures and markers."""
import os
import pytest

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_gemini: test makes a real Gemini API call — skipped when GEMINI_API_KEY is unset",
    )


@pytest.fixture(scope="session")
def gemini_key() -> str:
    """Return the Gemini API key, or skip the test if it is not set."""
    if not GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY environment variable is not set")
    return GEMINI_API_KEY
