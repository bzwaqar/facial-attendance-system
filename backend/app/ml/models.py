"""Data models and structured result definitions for face processing."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class DetectedFace:
    """Structured representation of a single detected face and its ArcFace embedding."""

    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float  # SCRFD detection confidence score
    embedding: Optional[np.ndarray] = None  # Normalized 512-dimensional ArcFace embedding vector
    landmarks: Optional[List[List[float]]] = None  # 5 facial landmark keypoints [[x, y], ...]

    @property
    def embedding_list(self) -> List[float]:
        """Return embedding vector as a standard Python list of floats."""
        if self.embedding is None:
            return []
        return self.embedding.tolist()

    @property
    def embedding_dim(self) -> int:
        """Return dimension of the facial embedding vector (typically 512)."""
        if self.embedding is None:
            return 0
        return int(self.embedding.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        """Convert detected face into a serializable dictionary representation."""
        return {
            "bbox": [round(float(coord), 2) for coord in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "landmarks": (
                [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in self.landmarks]
                if self.landmarks is not None
                else None
            ),
            "embedding_dim": self.embedding_dim,
            "embedding": self.embedding_list,
        }


@dataclass
class FaceProcessingResult:
    """Result payload containing all detected and embedded faces from an input image."""

    total_detected: int
    faces: List[DetectedFace] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @property
    def has_faces(self) -> bool:
        """True if at least one face was detected exceeding the threshold."""
        return len(self.faces) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert processing result into a dictionary."""
        return {
            "total_detected": self.total_detected,
            "faces": [face.to_dict() for face in self.faces],
            "processing_time_ms": round(self.processing_time_ms, 2),
        }
