"""Pydantic schemas for face recognition prediction and identity matching."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RecognizedFaceMatch(BaseModel):
    """Recognition and anti-spoofing verdict for a single detected face."""

    face_index: int
    matched: bool = Field(..., description="Whether identity matched an enrolled student above threshold.")
    student_id: Optional[str] = Field(None, description="Matched student ID if identified.")
    student_name: Optional[str] = Field(None, description="Matched student name if registered.")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score.")
    detection_confidence: float = Field(..., ge=0.0, le=1.0, description="SCRFD face detection confidence.")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2].")
    landmarks: Optional[List[List[float]]] = Field(None, description="5-point facial landmarks.")
    is_live: Optional[bool] = Field(None, description="Whether anti-spoofing verification passed.")
    liveness_score: Optional[float] = Field(None, description="Liveness confidence score.")
    liveness_status: Optional[str] = Field(None, description="Liveness status string (LIVE, SPOOF, UNCERTAIN).")
    liveness_reason: Optional[str] = Field(None, description="Explanation of liveness verdict.")


class RecognitionPredictResponse(BaseModel):
    """Response payload for on-demand face recognition queries."""

    total_faces_detected: int = Field(..., description="Total faces detected in the image.")
    faces: List[RecognizedFaceMatch] = Field(default_factory=list, description="Per-face recognition results.")
    inference_time_ms: float = Field(..., description="Total processing latency in milliseconds.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Evaluation timestamp.")


class RecognitionPredictBase64Request(BaseModel):
    """Request payload containing a base64 encoded image."""

    image_base64: str = Field(..., description="Base64 encoded JPEG or PNG image string.")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Custom similarity threshold override.")
    check_liveness: bool = Field(default=True, description="Whether to evaluate anti-spoofing cues.")
