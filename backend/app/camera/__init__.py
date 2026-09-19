"""Real-time camera recognition and tracking package."""

from app.camera.annotator import FrameAnnotator
from app.camera.capture import CameraCapture
from app.camera.pipeline import (
    CameraRecognitionPipeline,
    ProcessedFrameResult,
)
from app.camera.temporal import (
    IdentityObservation,
    TemporalFaceVerifier,
    TemporalIdentityState,
    TrackTemporalState,
)
from app.camera.tracker import FaceTracker, TrackedFace, compute_iou

__all__ = [
    "CameraCapture",
    "TrackedFace",
    "FaceTracker",
    "compute_iou",
    "TemporalIdentityState",
    "IdentityObservation",
    "TrackTemporalState",
    "TemporalFaceVerifier",
    "FrameAnnotator",
    "CameraRecognitionPipeline",
    "ProcessedFrameResult",
]
