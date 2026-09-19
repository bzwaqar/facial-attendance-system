"""Biometric face quality assessment and embedding vector validation."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from app.ml.models import DetectedFace

logger = logging.getLogger(__name__)


@dataclass
class FaceQualityAssessment:
    """Detailed biometric face quality evaluation metrics."""

    is_valid: bool
    reason: Optional[str] = None
    confidence: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0
    quality_score: float = 0.0
    face_dimensions: Tuple[int, int] = (0, 0)  # (width, height) in pixels


def assess_face_quality(
    bgr_image: np.ndarray,
    detected_face: DetectedFace,
    min_confidence: float = 0.65,
    min_face_size: int = 80,
    min_sharpness: float = 25.0,
    min_brightness: float = 30.0,
    max_brightness: float = 235.0,
) -> FaceQualityAssessment:
    """Evaluate detected face against quality requirements for biometric enrollment.
    
    Checks:
    - SCRFD detection confidence threshold
    - Minimum face resolution / bounding box dimensions
    - Focus / motion blur via Laplacian variance
    - Underexposure and overexposure via luminance distribution
    
    Args:
        bgr_image: Full 3-channel BGR image array.
        detected_face: DetectedFace instance containing bounding box and confidence.
        min_confidence: Minimum acceptable SCRFD detection confidence score.
        min_face_size: Minimum required face bounding box width and height in pixels.
        min_sharpness: Minimum Laplacian variance threshold (below is blurry).
        min_brightness: Minimum mean grayscale brightness (below is underexposed).
        max_brightness: Maximum mean grayscale brightness (above is overexposed).
        
    Returns:
        FaceQualityAssessment indicating if the face is usable and diagnostic metrics.
    """
    img_h, img_w = bgr_image.shape[:2]
    x1, y1, x2, y2 = detected_face.bbox

    # 1. Detection confidence check
    if detected_face.confidence < min_confidence:
        return FaceQualityAssessment(
            is_valid=False,
            reason=(
                f"Detection confidence ({detected_face.confidence:.2f}) is below "
                f"the minimum required threshold ({min_confidence:.2f})."
            ),
            confidence=round(detected_face.confidence, 4),
        )

    # 2. Boundary and resolution check
    clamped_x1 = max(0, min(int(round(x1)), img_w - 1))
    clamped_y1 = max(0, min(int(round(y1)), img_h - 1))
    clamped_x2 = max(0, min(int(round(x2)), img_w))
    clamped_y2 = max(0, min(int(round(y2)), img_h))

    face_w = clamped_x2 - clamped_x1
    face_h = clamped_y2 - clamped_y1

    if face_w < min_face_size or face_h < min_face_size:
        return FaceQualityAssessment(
            is_valid=False,
            reason=(
                f"Detected face resolution is too small ({face_w}x{face_h} px). "
                f"Minimum required resolution is {min_face_size}x{min_face_size} px."
            ),
            confidence=round(detected_face.confidence, 4),
            face_dimensions=(face_w, face_h),
        )

    # Extract cropped face region
    face_crop = bgr_image[clamped_y1:clamped_y2, clamped_x1:clamped_x2]
    if face_crop.size == 0:
        return FaceQualityAssessment(
            is_valid=False,
            reason="Face crop region is empty or invalid.",
            confidence=round(detected_face.confidence, 4),
        )

    gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # 3. Blur / Sharpness check using Laplacian variance
    sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    if sharpness < min_sharpness:
        return FaceQualityAssessment(
            is_valid=False,
            reason=(
                f"Face image is excessively blurry (sharpness score {sharpness:.1f} < "
                f"minimum required {min_sharpness:.1f}). Please provide a clear, in-focus photo."
            ),
            confidence=round(detected_face.confidence, 4),
            sharpness=round(sharpness, 2),
            face_dimensions=(face_w, face_h),
        )

    # 4. Illumination / Brightness check
    brightness = float(np.mean(gray_crop))
    if brightness < min_brightness:
        return FaceQualityAssessment(
            is_valid=False,
            reason=(
                f"Face is severely underexposed / too dark (brightness {brightness:.1f} < "
                f"{min_brightness:.1f}). Ensure adequate, even frontal lighting."
            ),
            confidence=round(detected_face.confidence, 4),
            sharpness=round(sharpness, 2),
            brightness=round(brightness, 2),
            face_dimensions=(face_w, face_h),
        )

    if brightness > max_brightness:
        return FaceQualityAssessment(
            is_valid=False,
            reason=(
                f"Face is severely overexposed / washed out (brightness {brightness:.1f} > "
                f"{max_brightness:.1f}). Avoid intense direct glare."
            ),
            confidence=round(detected_face.confidence, 4),
            sharpness=round(sharpness, 2),
            brightness=round(brightness, 2),
            face_dimensions=(face_w, face_h),
        )

    # 5. Compute composite quality score [0.0 - 1.0]
    # Confidence weight: 40%, Sharpness weight: 35%, Illumination balance weight: 25%
    conf_comp = min(1.0, max(0.0, (detected_face.confidence - 0.5) / 0.5))
    sharp_comp = min(1.0, max(0.0, sharpness / 200.0))
    illum_comp = 1.0 - (abs(brightness - 128.0) / 128.0)
    composite_score = (0.40 * conf_comp) + (0.35 * sharp_comp) + (0.25 * illum_comp)
    composite_score = round(float(np.clip(composite_score, 0.0, 1.0)), 4)

    return FaceQualityAssessment(
        is_valid=True,
        confidence=round(detected_face.confidence, 4),
        sharpness=round(sharpness, 2),
        brightness=round(brightness, 2),
        quality_score=composite_score,
        face_dimensions=(face_w, face_h),
    )


def validate_embedding_vector(
    vector: Union[np.ndarray, List[float]],
    expected_dim: int = 512,
) -> bool:
    """Validate ArcFace embedding vector integrity.
    
    Verifies:
    - Array is 1D with expected dimension (default: 512).
    - Contains no NaN or Inf values.
    - L2 norm is strictly non-zero and approximately unit length (1.0).
    
    Args:
        vector: 512-dimensional embedding as NumPy array or list.
        expected_dim: Expected number of float elements (512 for ArcFace).
        
    Returns:
        True if the embedding passes all integrity checks.
        
    Raises:
        ValueError: If any validation condition is violated.
    """
    arr = np.asarray(vector, dtype=np.float32)

    if arr.ndim != 1 or arr.shape[0] != expected_dim:
        raise ValueError(
            f"Invalid embedding dimension: expected ({expected_dim},), got {arr.shape}."
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError("Embedding vector contains non-finite values (NaN or Inf).")

    norm = float(np.linalg.norm(arr))
    if norm < 1e-4:
        raise ValueError("Embedding vector has near-zero norm (all zeros or degenerate).")

    if abs(norm - 1.0) > 0.05:
        raise ValueError(f"Embedding vector must be L2-normalized to 1.0 (found norm {norm:.4f}).")

    return True
