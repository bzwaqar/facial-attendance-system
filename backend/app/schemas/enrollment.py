"""Pydantic schemas for facial enrollment requests and responses."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ImageDiagnostic(BaseModel):
    """Detailed validation outcome and diagnostic metadata for a submitted image."""

    filename: str = Field(..., description="Original filename of the submitted image.")
    status: str = Field(
        ...,
        description="Validation outcome: 'accepted' or 'rejected'.",
        examples=["accepted", "rejected"],
    )
    reason: Optional[str] = Field(
        None,
        description="Explanation if the image was rejected (e.g. no face, multiple faces, blurry).",
    )
    confidence: Optional[float] = Field(
        None,
        description="Detection confidence score returned by SCRFD.",
    )
    quality_score: Optional[float] = Field(
        None,
        description="Composite quality score [0.0 - 1.0] evaluating sharpness and exposure.",
    )
    bbox: Optional[List[float]] = Field(
        None,
        description="Detected face bounding box [x1, y1, x2, y2].",
    )


class EnrollmentResponse(BaseModel):
    """Payload returned upon successful student facial enrollment."""

    student_id: str = Field(..., description="Student ID to which embeddings were enrolled.")
    total_submitted: int = Field(..., description="Number of images received in the request.")
    total_enrolled: int = Field(..., description="Number of valid reference embeddings persisted.")
    model_name: str = Field(..., description="Recognition model architecture (e.g. ArcFace).")
    model_version: str = Field(..., description="Backbone version / weights identifier.")
    images_summary: List[ImageDiagnostic] = Field(
        ...,
        description="Itemized processing diagnostics for each submitted image.",
    )
    message: str = Field(..., description="Human-readable status summary message.")


class EnrollmentErrorDetail(BaseModel):
    """Structured failure detail returned when enrollment validation fails (HTTP 400/422)."""

    message: str = Field(..., description="Summary error message explaining why enrollment was aborted.")
    total_submitted: int = Field(..., description="Total images provided in request.")
    accepted_count: int = Field(..., description="Count of images that passed all face criteria.")
    rejected_count: int = Field(..., description="Count of images that failed criteria.")
    diagnostics: List[ImageDiagnostic] = Field(
        ...,
        description="Diagnostic breakdown for each image explaining the failure cause.",
    )
