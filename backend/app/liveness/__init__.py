"""Modular Anti-Spoofing and Liveness Verification Package.

Exports clean interfaces and detectors for mitigating presentation attacks.
"""

from app.liveness.base import (
    BaseLivenessDetector,
    ChallengeType,
    LivenessResult,
    LivenessStatus,
)
from app.liveness.detector import (
    ChallengeResponseLivenessDetector,
    PassthroughLivenessDetector,
    TemporalHeuristicLivenessDetector,
)
from app.liveness.evaluator import LivenessEvaluator, get_liveness_evaluator
from app.liveness.factory import get_liveness_detector

__all__ = [
    "BaseLivenessDetector",
    "ChallengeType",
    "LivenessResult",
    "LivenessStatus",
    "TemporalHeuristicLivenessDetector",
    "ChallengeResponseLivenessDetector",
    "PassthroughLivenessDetector",
    "LivenessEvaluator",
    "get_liveness_evaluator",
    "get_liveness_detector",
]
