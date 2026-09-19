"""Automated unit and integration tests for the InsightFace face-processing pipeline."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from app.ml import FaceEngine, get_face_engine
from app.ml.exceptions import InvalidImageError
from app.ml.loader import InsightFaceModelLoader, get_model_loader

SAMPLE_IMAGE_PATH = Path(__file__).resolve().parent / "data" / "sample_student.jpg"


@pytest.fixture(scope="session")
def face_engine() -> FaceEngine:
    """Fixture providing a singleton FaceEngine instance for tests."""
    return get_face_engine()


def test_singleton_model_loader() -> None:
    """Verify that the model loader is a singleton and reuses the loaded model instance."""
    loader1 = get_model_loader()
    loader2 = get_model_loader()
    assert loader1 is loader2, "get_model_loader() must return the same singleton instance"

    model1 = loader1.get_model()
    model2 = loader2.get_model()
    assert model1 is model2, "FaceAnalysis model must be loaded once and reused across callers"


def test_invalid_image_inputs(face_engine: FaceEngine) -> None:
    """Verify that invalid, corrupt, or empty inputs raise InvalidImageError."""
    # 1. None input
    with pytest.raises(InvalidImageError):
        face_engine.process_image(None)  # type: ignore

    # 2. Empty byte payload
    with pytest.raises(InvalidImageError):
        face_engine.process_image(b"")

    # 3. Corrupted / non-image bytes
    with pytest.raises(InvalidImageError):
        face_engine.process_image(b"not-a-valid-image-format")

    # 4. Non-existent file path
    with pytest.raises(InvalidImageError):
        face_engine.process_image("non_existent_file_path_xyz.jpg")

    # 5. Empty numpy array
    with pytest.raises(InvalidImageError):
        face_engine.process_image(np.array([]))


def test_image_with_no_faces(face_engine: FaceEngine) -> None:
    """Verify that an image without faces cleanly returns an empty result with 0 faces."""
    # Create a blank black image
    blank_image = np.zeros((400, 400, 3), dtype=np.uint8)

    result = face_engine.process_image(blank_image)
    assert result.total_detected == 0
    assert len(result.faces) == 0
    assert not result.has_faces
    assert result.processing_time_ms >= 0.0


def test_face_detection_and_arcface_embedding(face_engine: FaceEngine) -> None:
    """Verify face detection, alignment, and normalized ArcFace 512-d embeddings on sample face."""
    assert SAMPLE_IMAGE_PATH.is_file(), f"Sample test image missing at {SAMPLE_IMAGE_PATH}"

    result = face_engine.process_image(SAMPLE_IMAGE_PATH, confidence_threshold=0.50)

    assert result.total_detected >= 1
    assert result.has_faces

    face = result.faces[0]

    # Verify Bounding Box
    assert len(face.bbox) == 4
    x1, y1, x2, y2 = face.bbox
    assert x2 > x1 and y2 > y1, "Bounding box coordinates must be valid (x2 > x1, y2 > y1)"

    # Verify Confidence
    assert face.confidence >= 0.50

    # Verify Landmarks
    assert face.landmarks is not None
    assert len(face.landmarks) == 5, "SCRFD must return 5 facial landmark keypoints"

    # Verify ArcFace Embedding
    assert face.embedding_dim == 512, "ArcFace embedding must have 512 dimensions"
    assert face.embedding.shape == (512,)
    assert face.embedding.dtype == np.float32

    # Verify L2 Normalization (norm must be unit length ~ 1.0)
    l2_norm = float(np.linalg.norm(face.embedding))
    assert abs(l2_norm - 1.0) < 1e-5, f"Embedding must be L2-normalized to 1.0, got {l2_norm}"

    # Verify serialization
    face_dict = face.to_dict()
    assert "bbox" in face_dict
    assert "confidence" in face_dict
    assert "landmarks" in face_dict
    assert len(face_dict["embedding"]) == 512


def test_multiple_faces_detection(face_engine: FaceEngine) -> None:
    """Verify that the pipeline detects multiple faces when present in an image."""
    # Load sample image and horizontally tile it to create a 2-face image
    img = cv2.imread(str(SAMPLE_IMAGE_PATH))
    assert img is not None

    two_faces_img = np.hstack([img, img])

    result = face_engine.process_image(two_faces_img, confidence_threshold=0.50)

    assert result.total_detected >= 2, f"Expected at least 2 detected faces, got {result.total_detected}"
    assert len(result.faces) >= 2

    for face in result.faces:
        assert face.embedding_dim == 512
        l2_norm = float(np.linalg.norm(face.embedding))
        assert abs(l2_norm - 1.0) < 1e-5


def test_confidence_threshold_filtering(face_engine: FaceEngine) -> None:
    """Verify that detections below a strict confidence threshold are filtered out."""
    # With a near-impossible threshold (0.999), detections should be filtered out
    result = face_engine.process_image(SAMPLE_IMAGE_PATH, confidence_threshold=0.999)
    assert result.total_detected == 0
    assert len(result.faces) == 0
