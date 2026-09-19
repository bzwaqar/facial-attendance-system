"""Automated integration tests for EnrollmentService pipeline."""

from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.base import Base
from app.db.models.face_embedding import FaceEmbedding
from app.db.models.student import Student
from app.ml.engine import get_face_engine
from app.schemas.student import StudentCreate
from app.services.enrollment_service import EnrollmentService
from app.services.exceptions import (
    FaceEnrollmentValidationError,
    InvalidImageCountError,
    StudentNotFoundError,
)
from app.services.student_service import StudentService

SAMPLE_IMAGE_PATH = Path(__file__).resolve().parent / "data" / "sample_student.jpg"
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def test_session() -> AsyncSession:
    """Provide an isolated in-memory async database session."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture(scope="session")
def valid_face_image_bytes() -> bytes:
    """Return raw bytes of a valid face test image."""
    assert SAMPLE_IMAGE_PATH.is_file(), f"Sample test image missing at {SAMPLE_IMAGE_PATH}"
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        return f.read()


def _generate_valid_face_images(base_bytes: bytes, count: int = 5) -> List[Tuple[str, bytes]]:
    """Generate `count` distinct valid face image payloads from sample image."""
    arr = np.frombuffer(base_bytes, dtype=np.uint8)
    base_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    images = []
    for i in range(count):
        # Apply slight brightness modulation to ensure distinct byte payloads
        altered = cv2.convertScaleAbs(base_img, alpha=1.0, beta=(i - 2) * 3)
        _, buf = cv2.imencode(".jpg", altered, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        images.append((f"student_angle_{i+1}.jpg", buf.tobytes()))
    return images


@pytest.mark.asyncio
async def test_enrollment_rejects_under_5_images(
    test_session: AsyncSession,
    valid_face_image_bytes: bytes,
) -> None:
    """Verify that submitting fewer than 5 images raises InvalidImageCountError."""
    # Create student
    await StudentService.create_student(
        test_session,
        StudentCreate(
            student_id="ENROLL-TEST-001",
            name="Test User 1",
            department="CS",
            semester="1st",
            section="A",
        ),
    )
    await test_session.commit()

    service = EnrollmentService()
    # Provide only 3 images
    images = _generate_valid_face_images(valid_face_image_bytes, count=3)

    with pytest.raises(InvalidImageCountError) as exc_info:
        await service.enroll_student_faces(test_session, "ENROLL-TEST-001", images)

    assert "between 5 and 10" in str(exc_info.value)
    assert exc_info.value.received_count == 3


@pytest.mark.asyncio
async def test_enrollment_rejects_over_10_images(
    test_session: AsyncSession,
    valid_face_image_bytes: bytes,
) -> None:
    """Verify that submitting more than 10 images raises InvalidImageCountError."""
    await StudentService.create_student(
        test_session,
        StudentCreate(
            student_id="ENROLL-TEST-002",
            name="Test User 2",
            department="CS",
            semester="1st",
            section="A",
        ),
    )
    await test_session.commit()

    service = EnrollmentService()
    # Provide 11 images
    images = _generate_valid_face_images(valid_face_image_bytes, count=11)

    with pytest.raises(InvalidImageCountError) as exc_info:
        await service.enroll_student_faces(test_session, "ENROLL-TEST-002", images)

    assert "between 5 and 10" in str(exc_info.value)
    assert exc_info.value.received_count == 11


@pytest.mark.asyncio
async def test_enrollment_rejects_non_existent_student(
    test_session: AsyncSession,
    valid_face_image_bytes: bytes,
) -> None:
    """Verify that enrolling for a non-existent student raises StudentNotFoundError."""
    service = EnrollmentService()
    images = _generate_valid_face_images(valid_face_image_bytes, count=5)

    with pytest.raises(StudentNotFoundError):
        await service.enroll_student_faces(test_session, "NON-EXISTENT-STUDENT", images)


@pytest.mark.asyncio
async def test_enrollment_rejects_image_with_no_face(
    test_session: AsyncSession,
    valid_face_image_bytes: bytes,
) -> None:
    """Verify rejection when any submitted image has no face."""
    await StudentService.create_student(
        test_session,
        StudentCreate(
            student_id="ENROLL-TEST-003",
            name="Test User 3",
            department="CS",
            semester="1st",
            section="A",
        ),
    )
    await test_session.commit()

    service = EnrollmentService()
    images = _generate_valid_face_images(valid_face_image_bytes, count=4)

    # 5th image is a black blank square with 0 faces
    blank = np.zeros((400, 400, 3), dtype=np.uint8)
    _, blank_buf = cv2.imencode(".jpg", blank)
    images.append(("blank_no_face.jpg", blank_buf.tobytes()))

    with pytest.raises(FaceEnrollmentValidationError) as exc_info:
        await service.enroll_student_faces(test_session, "ENROLL-TEST-003", images)

    err = exc_info.value
    assert err.total_submitted == 5
    assert err.accepted_count == 4
    assert err.rejected_count == 1
    # Check that diagnostic has reason mentioning no face
    rejected_diag = [d for d in err.diagnostics if d.status == "rejected"][0]
    assert "no face" in rejected_diag.reason.lower()


@pytest.mark.asyncio
async def test_enrollment_rejects_image_with_multiple_faces(
    test_session: AsyncSession,
    valid_face_image_bytes: bytes,
) -> None:
    """Verify rejection when an image contains multiple faces."""
    await StudentService.create_student(
        test_session,
        StudentCreate(
            student_id="ENROLL-TEST-004",
            name="Test User 4",
            department="CS",
            semester="1st",
            section="A",
        ),
    )
    await test_session.commit()

    service = EnrollmentService()
    images = _generate_valid_face_images(valid_face_image_bytes, count=4)

    # 5th image has two faces side-by-side
    arr = np.frombuffer(valid_face_image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    two_faces = np.hstack([img, img])
    _, two_faces_buf = cv2.imencode(".jpg", two_faces)
    images.append(("two_faces.jpg", two_faces_buf.tobytes()))

    with pytest.raises(FaceEnrollmentValidationError) as exc_info:
        await service.enroll_student_faces(test_session, "ENROLL-TEST-004", images)

    err = exc_info.value
    assert err.rejected_count >= 1
    rejected_diag = [d for d in err.diagnostics if d.filename == "two_faces.jpg"][0]
    assert "multiple faces" in rejected_diag.reason.lower()


@pytest.mark.asyncio
async def test_enrollment_rejects_blurry_image(
    test_session: AsyncSession,
    valid_face_image_bytes: bytes,
) -> None:
    """Verify rejection when an image is severely blurred."""
    await StudentService.create_student(
        test_session,
        StudentCreate(
            student_id="ENROLL-TEST-005",
            name="Test User 5",
            department="CS",
            semester="1st",
            section="A",
        ),
    )
    await test_session.commit()

    service = EnrollmentService()
    images = _generate_valid_face_images(valid_face_image_bytes, count=4)

    # 5th image is heavily blurred
    arr = np.frombuffer(valid_face_image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    blurry = cv2.GaussianBlur(img, (65, 65), 0)
    _, blurry_buf = cv2.imencode(".jpg", blurry)
    images.append(("blurry_face.jpg", blurry_buf.tobytes()))

    with pytest.raises(FaceEnrollmentValidationError) as exc_info:
        await service.enroll_student_faces(test_session, "ENROLL-TEST-005", images)

    err = exc_info.value
    assert err.rejected_count >= 1
    rejected_diag = [d for d in err.diagnostics if d.filename == "blurry_face.jpg"][0]
    assert "blurry" in rejected_diag.reason.lower() or "no face" in rejected_diag.reason.lower()


@pytest.mark.asyncio
async def test_enrollment_successful_with_5_to_10_images(
    test_session: AsyncSession,
    valid_face_image_bytes: bytes,
) -> None:
    """Verify complete successful enrollment storing 5 embeddings with model name/version."""
    student = await StudentService.create_student(
        test_session,
        StudentCreate(
            student_id="ENROLL-SUCCESS-001",
            name="Commander Shepard",
            department="Alliance Navy",
            semester="N7",
            section="Normandy",
        ),
    )
    await test_session.commit()

    service = EnrollmentService()
    images = _generate_valid_face_images(valid_face_image_bytes, count=5)

    response = await service.enroll_student_faces(test_session, "ENROLL-SUCCESS-001", images)
    await test_session.commit()

    assert response.student_id == "ENROLL-SUCCESS-001"
    assert response.total_submitted == 5
    assert response.total_enrolled == 5
    assert response.model_name == "ArcFace"
    assert response.model_version == settings.INSIGHTFACE_MODEL_NAME
    assert len(response.images_summary) == 5

    for diag in response.images_summary:
        assert diag.status == "accepted"
        assert diag.confidence is not None
        assert diag.confidence >= settings.DETECTION_CONFIDENCE_THRESHOLD
        assert diag.quality_score is not None
        assert diag.quality_score > 0.0
        assert diag.bbox is not None

    # Query student back and verify persisted FaceEmbedding records in DB
    refreshed_student = await StudentService.get_student_by_student_id(
        test_session,
        "ENROLL-SUCCESS-001",
        load_embeddings=True,
    )
    assert refreshed_student is not None
    assert len(refreshed_student.face_embeddings) == 5

    for emb in refreshed_student.face_embeddings:
        assert emb.model_name == "ArcFace"
        assert emb.model_version == settings.INSIGHTFACE_MODEL_NAME
        assert emb.quality_score is not None
        # Check deserialized vector integrity
        vec = emb.get_vector()
        assert len(vec) == 512
        l2_norm = float(np.linalg.norm(np.array(vec, dtype=np.float32)))
        assert abs(l2_norm - 1.0) < 1e-4
