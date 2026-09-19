"""Unit tests for biometric face quality assessment and embedding validation."""

import numpy as np
import pytest

from app.ml.models import DetectedFace
from app.ml.quality import assess_face_quality, validate_embedding_vector


def test_validate_embedding_vector_valid() -> None:
    """Verify that a normalized 512-d float array passes validation."""
    vec = np.random.randn(512).astype(np.float32)
    vec = vec / float(np.linalg.norm(vec))
    assert validate_embedding_vector(vec) is True
    assert validate_embedding_vector(vec.tolist()) is True


def test_validate_embedding_vector_invalid_dimension() -> None:
    """Verify that invalid vector dimensions raise ValueError."""
    # 128-d instead of 512-d
    vec_128 = np.ones(128, dtype=np.float32)
    vec_128 /= float(np.linalg.norm(vec_128))
    with pytest.raises(ValueError, match="Invalid embedding dimension"):
        validate_embedding_vector(vec_128)

    # 2D array
    with pytest.raises(ValueError, match="Invalid embedding dimension"):
        validate_embedding_vector(np.ones((512, 1), dtype=np.float32))


def test_validate_embedding_vector_nan_inf() -> None:
    """Verify that NaN and Inf values are rejected."""
    vec = np.ones(512, dtype=np.float32) / np.sqrt(512)
    vec[10] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        validate_embedding_vector(vec)

    vec[10] = np.inf
    with pytest.raises(ValueError, match="non-finite values"):
        validate_embedding_vector(vec)


def test_validate_embedding_vector_zero_norm() -> None:
    """Verify that zero / degenerate vector raises ValueError."""
    vec = np.zeros(512, dtype=np.float32)
    with pytest.raises(ValueError, match="near-zero norm"):
        validate_embedding_vector(vec)


def test_assess_face_quality_valid_face() -> None:
    """Verify that a sharp, well-lit, properly sized face passes quality assessment."""
    # Create synthetic image: 400x400 with high texture
    np.random.seed(42)
    img = np.random.randint(50, 200, (400, 400, 3), dtype=np.uint8)

    face = DetectedFace(
        bbox=[100.0, 100.0, 300.0, 300.0],
        confidence=0.95,
        embedding=np.zeros(512, dtype=np.float32),
        landmarks=[[150.0, 150.0], [250.0, 150.0], [200.0, 200.0], [160.0, 250.0], [240.0, 250.0]],
    )

    assessment = assess_face_quality(img, face, min_confidence=0.65)
    assert assessment.is_valid is True
    assert assessment.quality_score > 0.5
    assert assessment.sharpness > 25.0
    assert 30.0 <= assessment.brightness <= 235.0


def test_assess_face_quality_low_confidence() -> None:
    """Verify rejection when SCRFD detection confidence is below threshold."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 128
    face = DetectedFace(
        bbox=[100.0, 100.0, 300.0, 300.0],
        confidence=0.45,  # Below 0.65
        embedding=np.zeros(512, dtype=np.float32),
    )

    assessment = assess_face_quality(img, face, min_confidence=0.65)
    assert assessment.is_valid is False
    assert "Detection confidence" in assessment.reason


def test_assess_face_quality_too_small() -> None:
    """Verify rejection when face bounding box is smaller than min_face_size (80px)."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 128
    face = DetectedFace(
        bbox=[100.0, 100.0, 150.0, 150.0],  # 50x50 px
        confidence=0.90,
        embedding=np.zeros(512, dtype=np.float32),
    )

    assessment = assess_face_quality(img, face, min_confidence=0.65, min_face_size=80)
    assert assessment.is_valid is False
    assert "resolution is too small" in assessment.reason


def test_assess_face_quality_blurry() -> None:
    """Verify rejection when face crop has low Laplacian variance (blurry / flat)."""
    # Completely uniform image -> variance = 0
    img = np.ones((400, 400, 3), dtype=np.uint8) * 120
    face = DetectedFace(
        bbox=[100.0, 100.0, 300.0, 300.0],
        confidence=0.90,
        embedding=np.zeros(512, dtype=np.float32),
    )

    assessment = assess_face_quality(img, face, min_sharpness=25.0)
    assert assessment.is_valid is False
    assert "blurry" in assessment.reason.lower()


def test_assess_face_quality_dark_and_bright() -> None:
    """Verify rejection when face is underexposed or overexposed."""
    # Near black image
    dark_img = np.ones((400, 400, 3), dtype=np.uint8) * 10
    face = DetectedFace(
        bbox=[50.0, 50.0, 250.0, 250.0],
        confidence=0.90,
        embedding=np.zeros(512, dtype=np.float32),
    )
    dark_assessment = assess_face_quality(dark_img, face, min_sharpness=0.0, min_brightness=30.0)
    assert dark_assessment.is_valid is False
    assert "underexposed" in dark_assessment.reason

    # Blown out white image
    bright_img = np.ones((400, 400, 3), dtype=np.uint8) * 250
    bright_assessment = assess_face_quality(bright_img, face, min_sharpness=0.0, max_brightness=235.0)
    assert bright_assessment.is_valid is False
    assert "overexposed" in bright_assessment.reason
