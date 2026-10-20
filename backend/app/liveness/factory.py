"""Factory providing pluggable liveness detector instances based on runtime configuration."""

import logging
from typing import Optional

from app.core.config import settings
from app.liveness.base import BaseLivenessDetector
from app.liveness.detector import (
    ChallengeResponseLivenessDetector,
    PassthroughLivenessDetector,
    TemporalHeuristicLivenessDetector,
)

logger = logging.getLogger(__name__)

_singleton_detector: Optional[BaseLivenessDetector] = None


def get_liveness_detector(method: Optional[str] = None) -> BaseLivenessDetector:
    """Instantiate and return a configured BaseLivenessDetector instance.
    
    When method is None, reuses a singleton instance so temporal buffers persist across frames.
    When method is specified, creates a dedicated instance for that method.
    """
    global _singleton_detector

    if not settings.LIVENESS_ENABLED:
        return PassthroughLivenessDetector()

    if method is not None:
        target = method.strip().lower()
        if target in ("disabled", "passthrough"):
            return PassthroughLivenessDetector()
        if target in ("challenge", "challenge_response"):
            return ChallengeResponseLivenessDetector()
        if target == "heuristic":
            return TemporalHeuristicLivenessDetector(
                required_frames=settings.LIVENESS_REQUIRED_FRAMES,
                pose_variance_threshold=settings.LIVENESS_POSE_VARIANCE_THRESHOLD,
                min_score=settings.LIVENESS_MIN_SCORE,
            )
        return TemporalHeuristicLivenessDetector()

    # Default runtime: reuse singleton instance
    if _singleton_detector is not None:
        return _singleton_detector

    target_method = settings.LIVENESS_METHOD.strip().lower()
    if target_method in ("disabled", "passthrough"):
        _singleton_detector = PassthroughLivenessDetector()
    elif target_method in ("challenge", "challenge_response"):
        _singleton_detector = ChallengeResponseLivenessDetector()
    else:
        _singleton_detector = TemporalHeuristicLivenessDetector(
            required_frames=settings.LIVENESS_REQUIRED_FRAMES,
            pose_variance_threshold=settings.LIVENESS_POSE_VARIANCE_THRESHOLD,
            min_score=settings.LIVENESS_MIN_SCORE,
        )
    return _singleton_detector


def reset_liveness_detector() -> None:
    """Reset the singleton detector instance and its buffers."""
    global _singleton_detector
    if _singleton_detector is not None:
        _singleton_detector.reset()
