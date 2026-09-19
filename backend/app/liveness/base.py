"""Core interfaces, data models, and contracts for anti-spoofing and liveness detection.

DISCLAIMER & THREAT MODEL NOTICE:
This module provides a practical, lightweight experimental anti-spoofing layer designed
primarily to mitigate low-effort presentation attacks such as static printed 2D photos
and motionless mobile/tablet displays held in front of the camera.

It is NOT certified production-grade anti-spoofing (ISO/IEC 30107-3 PAD) and cannot reliably
stop sophisticated attacks such as animated video replays, warped photos, 3D silicone masks,
or generative AI deepfakes.

The architecture strictly decouples liveness evaluation via the `BaseLivenessDetector`
abstract contract, allowing drop-in replacement with dedicated deep-learning models
(such as MiniFASNet, Silent-Face-Anti-Spoofing, or multi-modal IR/depth sensors) without
altering the facial recognition or attendance pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from app.ml.models import DetectedFace


class LivenessStatus(str, Enum):
    """Liveness outcome status categories."""

    LIVE = "LIVE"                        # Presentation verified as genuine/live
    SPOOF = "SPOOF"                      # Presentation attack detected (e.g. static photo)
    UNCERTAIN = "UNCERTAIN"              # Insufficient temporal cues or ambiguous score
    PENDING_CHALLENGE = "PENDING_CHALLENGE"  # Awaiting completion of an active cue challenge


class ChallengeType(str, Enum):
    """Types of active interactive liveness challenges."""

    TURN_LEFT = "TURN_LEFT"      # Turn head towards subject's left (camera's right)
    TURN_RIGHT = "TURN_RIGHT"    # Turn head towards subject's right (camera's left)
    NOD_HEAD = "NOD_HEAD"        # Pitch head downwards / nod
    BLINK = "BLINK"              # Perform an eye blink
    ANY_MOVEMENT = "ANY_MOVEMENT"  # Any natural head movement or micro-pose variation


@dataclass
class LivenessResult:
    """Standardized result returned by all liveness and anti-spoofing detectors."""

    is_live: bool
    score: float  # Confidence score in [0.0, 1.0] where 1.0 is highest certainty of being live
    status: LivenessStatus
    reason: str  # Human-readable explanation of the verdict
    cues: Dict[str, Any] = field(default_factory=dict)  # Measured telemetry (pose_var, blink_count, etc.)
    timestamp: float = field(default_factory=time.time)
    active_challenge: Optional[ChallengeType] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert liveness result into a serializable dictionary representation."""
        return {
            "is_live": self.is_live,
            "score": round(float(self.score), 4),
            "status": self.status.value,
            "reason": self.reason,
            "cues": {k: round(v, 4) if isinstance(v, float) else v for k, v in self.cues.items()},
            "timestamp": self.timestamp,
            "active_challenge": self.active_challenge.value if self.active_challenge else None,
        }


class BaseLivenessDetector(ABC):
    """Abstract Base Class defining the modular liveness detector interface.
    
    Implementations must be independent of ArcFace recognition and can range from
    lightweight temporal heuristics to dedicated deep-learning anti-spoofing neural nets.
    """

    @abstractmethod
    def is_live(
        self,
        frame: np.ndarray,
        face: Optional[Union[DetectedFace, Any]] = None,
        bbox: Optional[List[float]] = None,
        landmarks: Optional[List[List[float]]] = None,
        track_id: Optional[int] = None,
        **kwargs: Any,
    ) -> LivenessResult:
        """Evaluate presentation liveness for a given face/frame observation.
        
        Args:
            frame: Full BGR frame or cropped face image as a NumPy array.
            face: Optional DetectedFace or TrackedFace instance providing bounding box/landmarks.
            bbox: Optional [x1, y1, x2, y2] bounding box coordinates.
            landmarks: Optional list of 5 facial landmarks [[x, y], ...].
            track_id: Optional persistent track identifier for temporal cue accumulation.
            **kwargs: Additional metadata or context passed to specific implementations.
            
        Returns:
            LivenessResult containing verdict, score, status, reason, and telemetry cues.
        """
        pass

    @abstractmethod
    def reset(self, track_id: Optional[int] = None) -> None:
        """Reset internal temporal buffers and state tracking.
        
        Args:
            track_id: If provided, resets state only for that track; if None, resets all tracks.
        """
        pass
