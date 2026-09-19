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


def get_liveness_detector(method: Optional[str] = None) -> BaseLivenessDetector:
    """Instantiate and return a configured BaseLivenessDetector instance.
    
    Args:
        method: Optional override ("heuristic", "challenge_response", "disabled", "model").
                If not specified, defaults to settings.LIVENESS_METHOD.
                
    Returns:
        Concrete BaseLivenessDetector implementation instance.
    """
    if not settings.LIVENESS_ENABLED:
        logger.info("Liveness detection is globally DISABLED. Using PassthroughLivenessDetector.")
        return PassthroughLivenessDetector()

    target_method = (method or settings.LIVENESS_METHOD).strip().lower()

    if target_method == "heuristic":
        logger.info("Instantiating TemporalHeuristicLivenessDetector (temporal landmark & patch cues).")
        return TemporalHeuristicLivenessDetector(
            required_frames=settings.LIVENESS_REQUIRED_FRAMES,
            pose_variance_threshold=settings.LIVENESS_POSE_VARIANCE_THRESHOLD,
            min_score=settings.LIVENESS_MIN_SCORE,
        )

    elif target_method in ("challenge", "challenge_response"):
        logger.info("Instantiating ChallengeResponseLivenessDetector (interactive voluntary response).")
        return ChallengeResponseLivenessDetector()

    elif target_method in ("disabled", "passthrough"):
        logger.info("Instantiating PassthroughLivenessDetector.")
        return PassthroughLivenessDetector()

    elif target_method == "model":
        # Placeholder / hook for dedicated deep learning PAD model (e.g. MiniFASNet / ONNX PAD)
        logger.warning(
            "Dedicated model PAD requested but no separate weights specified; "
            "falling back to TemporalHeuristicLivenessDetector."
        )
        return TemporalHeuristicLivenessDetector()

    else:
        logger.warning(
            "Unknown liveness method '%s'. Defaulting to TemporalHeuristicLivenessDetector.",
            target_method,
        )
        return TemporalHeuristicLivenessDetector()
