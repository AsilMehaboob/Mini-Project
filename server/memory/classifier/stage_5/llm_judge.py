import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

VALID_TIERS = frozenset({"SCRATCH", "SESSION", "LONGTERM"})

GEMINI_MODEL    = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

_SYSTEM_PROMPT = (
    "You are a memory classification system for an AI agent. "
    "Classify each piece of text into exactly one memory tier:\n\n"
    "  SCRATCH  - Ephemeral / transient information (errors, status logs, bare values, debug output).\n"
    "  SESSION  - Short-term working memory for the current conversation or task.\n"
    "  LONGTERM - Permanent facts, user preferences, skills, rules, and important personal information.\n\n"
    "Respond with a JSON object and nothing else:\n"
    '{"tier": "SCRATCH"|"SESSION"|"LONGTERM", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}'
)


def _user_prompt(
    content: str,
    metadata: Dict[str, Any],
    stage2_scores: Dict[str, float],
) -> str:
    scores_str = ", ".join(f"{k}: {v:.3f}" for k, v in stage2_scores.items())
    return (
        f"Memory content: {content!r}\n"
        f"Metadata: {json.dumps(metadata)}\n"
        f"Embedding classifier scores (below confidence threshold): {scores_str}\n\n"
        "Classify this memory into the most appropriate tier."
    )


def _call_gemini(
    content: str,
    metadata: Dict[str, Any],
    stage2_scores: Dict[str, float],
    api_key: str,
    base_url: str,
    model: str,
) -> Optional[Dict[str, Any]]:
    try:
        import httpx
    except ImportError:
        logger.error("Stage 5 | httpx is not installed; run: uv add httpx")
        return None

    url = f"{base_url.rstrip('/')}/{model}:generateContent?key={api_key}"

    body = {
        "system_instruction": {
            "parts": [{"text": _SYSTEM_PROMPT}]
        },
        "contents": [{
            "role": "user",
            "parts": [{"text": _user_prompt(content, metadata, stage2_scores)}],
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
        },
    }

    last_exc: Optional[Exception] = None
    for attempt in range(3):
        if attempt > 0:
            wait = 2 ** attempt   # 2s, 4s
            logger.warning("Stage 5 | retry %d after %ds (last error: %s)", attempt, wait, last_exc)
            time.sleep(wait)
        try:
            with httpx.Client(timeout=40.0) as client:
                resp = client.post(url, json=body)
                resp.raise_for_status()

            data      = resp.json()
            candidate = data.get("candidates", [{}])[0]

            if candidate.get("finishReason") not in ("STOP", "MAX_TOKENS"):
                raise ValueError(f"Unexpected finishReason: {candidate.get('finishReason')!r}")

            parts = candidate.get("content", {}).get("parts", [])
            if not parts:
                raise ValueError("No parts in Gemini response content")

            raw_text = parts[0].get("text", "").strip()
            start = raw_text.find("{")
            end   = raw_text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError(f"No JSON object in response: {raw_text!r}")

            return json.loads(raw_text[start:end])

        except Exception as exc:
            last_exc = exc
            # Only retry on rate-limit (429) or server errors (5xx)
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status not in (429, 500, 502, 503, 504):
                break

    logger.error("Stage 5 | Gemini call failed after retries: %s", last_exc)
    return None


def _fallback(stage2_scores: Dict[str, float], reason: str) -> Tuple[str, float, str]:
    tier = max(stage2_scores, key=lambda k: stage2_scores[k])
    return tier, stage2_scores[tier], f"fallback ({reason})"


def judge(
    content: str,
    metadata: Dict[str, Any],
    stage2_scores: Dict[str, float],
    api_key: Optional[str] = None,
    base_url: str = GEMINI_BASE_URL,
    model: str = GEMINI_MODEL,
) -> Tuple[str, float, str]:
    """
    Ask Gemini to resolve ambiguous memory tier classification.

    Uses the native Gemini generateContent API with JSON output mode.
    Set GEMINI_API_KEY in the environment or pass api_key directly.

    Returns:
        (tier, confidence, reasoning)

    Falls back gracefully to the top Stage 2 score when:
    - No API key is configured
    - The HTTP call fails
    - The response is unparseable or contains an invalid tier
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")

    if not resolved_key:
        logger.warning("Stage 5 | no API key; using Stage 2 fallback")
        return _fallback(stage2_scores, "no API key configured")

    result = _call_gemini(content, metadata, stage2_scores, resolved_key, base_url, model)
    if result is None:
        return _fallback(stage2_scores, "Gemini call failed")

    raw_tier = str(result.get("tier", "")).upper()
    if raw_tier not in VALID_TIERS:
        logger.warning("Stage 5 | invalid tier %r from Gemini; falling back", raw_tier)
        return _fallback(stage2_scores, f"invalid tier {raw_tier!r} from Gemini")

    confidence = min(1.0, max(0.0, float(result.get("confidence", 0.5))))
    reasoning  = str(result.get("reasoning", ""))

    logger.debug(
        "Stage 5 | tier=%s confidence=%.3f reasoning=%r",
        raw_tier, confidence, reasoning,
    )

    return raw_tier, confidence, reasoning
