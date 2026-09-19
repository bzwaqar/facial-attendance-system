"""Integration tests for the real-time CameraRecognitionPipeline."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from app.camera.pipeline import CameraRecognitionPipeline
from app.camera.temporal import TemporalIdentityState
from app.liveness import PassthroughLivenessDetector, TemporalHeuristicLivenessDetector
from app.ml.engine import get_face_engine
from app.ml.vector_search import VectorMetadata, VectorSearchEngine
from app.services.matching_service import IdentityMatchingService

SAMPLE_IMAGE_PATH = Path(__file__).resolve().parent / "data" / "sample_student.jpg"


@pytest.fixture(scope="session")
def sample_face_image() -> np.ndarray:
    """Load sample test face image."""
    assert SAMPLE_IMAGE_PATH.is_file()
    img = cv2.imread(str(SAMPLE_IMAGE_PATH))
    assert img is not None
    return img


@pytest.fixture
def pipeline_with_enrolled_student(sample_face_image: np.ndarray) -> CameraRecognitionPipeline:
    """Create CameraRecognitionPipeline with one enrolled student matching sample_face_image."""
    face_engine = get_face_engine()

    # Extract embedding for sample face image and add to FAISS
    proc_result = face_engine.process_image(sample_face_image)
    assert proc_result.has_faces
    sample_emb = proc_result.faces[0].embedding

    vector_engine = VectorSearchEngine(dimension=512)
    vector_engine.build_index(
        vectors=sample_emb.reshape(1, -1),
        metadata_list=[
            VectorMetadata(student_id="CS-PIPELINE-001", student_pk=1, embedding_id=10)
        ],
    )

    matching_service = IdentityMatchingService(
        vector_engine=vector_engine,
        face_engine=face_engine,
    )

    pipeline = CameraRecognitionPipeline(
        face_engine=face_engine,
        matching_service=matching_service,
        liveness_detector=PassthroughLivenessDetector(),
        recognition_interval=5,  # Run recognition every 5 frames
        threshold=0.60,
    )
    pipeline.set_student_names({"CS-PIPELINE-001": "Pipeline Test Student"})
    return pipeline


def test_pipeline_periodic_recognition_and_compute_savings(
    pipeline_with_enrolled_student: CameraRecognitionPipeline,
    sample_face_image: np.ndarray,
) -> None:
    """Verify that ArcFace recognition executes periodically (every 5 frames), saving compute between frames."""
    total_frames = 12
    recognitions_per_frame = []
    tracks_per_frame = []

    for frame_idx in range(1, total_frames + 1):
        res = pipeline_with_enrolled_student.process_frame(sample_face_image)
        recognitions_per_frame.append(res.recognitions_executed_count)
        tracks_per_frame.append(len(res.tracks))

        assert res.faces_detected_count == 1
        assert res.annotated_frame.shape == sample_face_image.shape
        assert res.fps > 0

    # Recognition must have fired on frame 1 (initial detection)
    assert recognitions_per_frame[0] == 1, "Recognition must run on initial face appearance"

    # Frames 2, 3, 4, 5 (tracking between recognition) -> 0 recognitions! (90%+ compute saved)
    for f in range(1, 5):
        assert recognitions_per_frame[f] == 0, f"Frame {f+1} should track without running ArcFace"

    # Frame 6 (interval = 5 elapsed: 6 - 1 = 5) -> recognition fires again!
    assert recognitions_per_frame[5] == 1, "Recognition must fire again after interval elapses"

    # Total recognitions over 12 frames should be strictly less than 4 (saving over 75% ArcFace compute)
    total_rec = sum(recognitions_per_frame)
    assert total_rec == 3, f"Expected exactly 3 periodic recognitions, got {total_rec}"


def test_pipeline_temporal_confirmation_flow(
    pipeline_with_enrolled_student: CameraRecognitionPipeline,
    sample_face_image: np.ndarray,
) -> None:
    """Verify that multiple consistent observations transition a tracked face to CONFIRMED."""
    # Process 15 frames so that multiple periodic recognitions take place
    last_res = None
    for _ in range(15):
        last_res = pipeline_with_enrolled_student.process_frame(sample_face_image)

    assert last_res is not None
    assert len(last_res.tracks) == 1
    state = last_res.temporal_states[0]

    assert state is not None
    assert state.state == TemporalIdentityState.CONFIRMED
    assert state.confirmed_student_id == "CS-PIPELINE-001"
    assert state.confirmed_student_name == "Pipeline Test Student"
    assert state.confidence_score > 0.85


def test_pipeline_blank_image_no_faces(
    pipeline_with_enrolled_student: CameraRecognitionPipeline,
) -> None:
    """Verify that blank frames without faces are handled cleanly without error."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res = pipeline_with_enrolled_student.process_frame(blank)

    assert res.faces_detected_count == 0
    assert len(res.tracks) == 0
    assert res.recognitions_executed_count == 0
    assert res.annotated_frame.shape == (480, 640, 3)


def test_pipeline_rejects_static_photo_spoof(
    sample_face_image: np.ndarray,
) -> None:
    """Verify that CameraRecognitionPipeline with default heuristic liveness detects static photos as spoof."""
    face_engine = get_face_engine()
    proc_result = face_engine.process_image(sample_face_image)
    sample_emb = proc_result.faces[0].embedding

    vector_engine = VectorSearchEngine(dimension=512)
    vector_engine.build_index(
        vectors=sample_emb.reshape(1, -1),
        metadata_list=[
            VectorMetadata(student_id="CS-PIPELINE-001", student_pk=1, embedding_id=10)
        ],
    )
    matching_service = IdentityMatchingService(
        vector_engine=vector_engine,
        face_engine=face_engine,
    )

    pipeline = CameraRecognitionPipeline(
        face_engine=face_engine,
        matching_service=matching_service,
        liveness_detector=TemporalHeuristicLivenessDetector(required_frames=5),
        recognition_interval=5,
        threshold=0.60,
    )
    pipeline.set_student_names({"CS-PIPELINE-001": "Pipeline Test Student"})

    # Feed identical static photo 10 times
    last_res = None
    for _ in range(10):
        last_res = pipeline.process_frame(sample_face_image)

    assert last_res is not None
    state = last_res.temporal_states[0]
    assert state.is_spoof is True
    assert state.state != TemporalIdentityState.CONFIRMED
    assert state.confirmed_student_id is None
