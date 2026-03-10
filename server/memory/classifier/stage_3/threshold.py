import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.65


def check_confidence(
    scores: Dict[str, float],
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[Optional[str], bool]:
    """
    Evaluate whether Stage 2 probability scores exceed the confidence threshold.

    Returns:
        (top_tier, is_confident)

        is_confident=True  → score is decisive; skip Stage 5 and go straight to Final Decision.
        is_confident=False → score is ambiguous; route to Stage 5 (LLM Judge).
    """
    if not scores:
        logger.warning("Stage 3 | received empty scores")
        return None, False

    top_tier = max(scores, key=lambda k: scores[k])
    top_score = scores[top_tier]
    is_confident = top_score >= threshold

    logger.debug(
        "Stage 3 | top_tier=%s top_score=%.3f threshold=%.3f confident=%s",
        top_tier,
        top_score,
        threshold,
        is_confident,
    )

    return top_tier, is_confident
