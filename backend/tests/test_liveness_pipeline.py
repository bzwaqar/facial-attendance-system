"""Integration tests for Anti-Spoofing & Liveness layer in Attendance and Camera Pipelines."""

from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.camera.pipeline import CameraRecognitionPipeline
from app.camera.temporal import TemporalFaceVerifier, TemporalIdentityState
from app.core.config import settings
from app.db.base import Base
from app.db.models.student import Student
from app.liveness import (
    BaseLivenessDetector,
    LivenessResult,
    LivenessStatus,
    PassthroughLivenessDetector,
    TemporalHeuristicLivenessDetector,
)
from app.schemas.attendance import (
    AttendanceMarkRequest,
    AttendanceSessionCreate,
    AttendanceStatus,
)
from app.schemas.matching import IdentityMatchResult, MatchCandidate, MatchStatus
from app.services.attendance_service import AttendanceService
from app.services.matching_service import IdentityMatchingService

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def test_session() -> AsyncSession:
    """Fixture providing isolated in-memory async SQLite database."""
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


@pytest_asyncio.fixture(scope="function")
async def sample_student(test_session: AsyncSession) -> Student:
    """Register sample active student."""
    student = Student(
        student_id="STU-LIVENESS-01",
        name="Grace Hopper",
        department="Computer Science",
        semester="6th",
        section="A",
        is_active=True,
    )
    test_session.add(student)
    await test_session.commit()
    return student


@pytest_asyncio.fixture(scope="function")
async def active_session(test_session: AsyncSession) -> str:
    """Create an active attendance session."""
    service = AttendanceService(db=test_session)
    sess = await service.create_session(
        AttendanceSessionCreate(
            session_id="CS101-LIVENESS-TEST",
            course_name="Biometric Security",
            min_similarity_threshold=0.60,
            min_confirmations=3,
        ),
        session=test_session,
    )
    return sess.session_id


def _create_mock_match(student_id: str = "STU-LIVENESS-01", similarity: float = 0.90) -> IdentityMatchResult:
    return IdentityMatchResult(
        status=MatchStatus.MATCHED,
        matched=True,
        student_id=student_id,
        similarity_score=similarity,
        threshold_used=0.60,
        candidates=[
            MatchCandidate(student_id=student_id, similarity_score=similarity, vector_id=1)
        ],
    )


def test_temporal_verifier_blocks_confirmation_on_spoof():
    """Verify TemporalFaceVerifier suppresses identity confirmation when liveness check fails."""
    verifier = TemporalFaceVerifier(window_size=5, min_confirmations=3)
    track_id = 42

    # Step 1: Liveness check flags track as SPOOF
    verifier.update_liveness(
        track_id=track_id,
        is_live=False,
        score=0.10,
        status=LivenessStatus.SPOOF.value,
        reason="Static photo attack suspected",
    )

    # Step 2: Feed 3 matching observations that would normally trigger CONFIRMED
    mock_match = _create_mock_match()
    for _ in range(3):
        state = verifier.add_observation(track_id, mock_match, student_name="Grace Hopper")

    # Confirmation MUST be blocked due to failed liveness!
    assert state.state != TemporalIdentityState.CONFIRMED
    assert state.confirmed_student_id is None
    assert state.is_spoof is True
    assert "SPOOF DETECTED" in state.display_label


def test_temporal_verifier_revokes_confirmed_status_if_spoof_detected():
    """Verify that if a track was previously confirmed, a subsequent spoof detection revokes it."""
    verifier = TemporalFaceVerifier(window_size=5, min_confirmations=3)
    track_id = 99

    # Confirm identity first
    mock_match = _create_mock_match()
    for _ in range(3):
        state = verifier.add_observation(track_id, mock_match, student_name="Grace Hopper")

    assert state.state == TemporalIdentityState.CONFIRMED
    assert state.confirmed_student_id == "STU-LIVENESS-01"

    # Now spoof attack is detected on same track
    updated_state = verifier.update_liveness(
        track_id=track_id,
        is_live=False,
        score=0.05,
        status="SPOOF",
        reason="Screen replay detected",
    )

    assert updated_state.state == TemporalIdentityState.UNKNOWN
    assert updated_state.confirmed_student_id is None
    assert updated_state.is_spoof is True


@pytest.mark.asyncio
async def test_attendance_service_rejects_spoofed_attendance(
    test_session: AsyncSession,
    sample_student: Student,
    active_session: str,
):
    """Verify AttendanceService rejects attendance when is_live is False."""
    service = AttendanceService(db=test_session)

    req = AttendanceMarkRequest(
        session_id=active_session,
        student_id=sample_student.student_id,
        similarity_score=0.95,
        observation_count=3,
        is_confirmed=True,
        is_live=False,  # Presentation attack flagged!
        liveness_score=0.15,
        liveness_status="SPOOF",
    )

    res = await service.mark_attendance(req, session=test_session)

    assert res.status == AttendanceStatus.REJECTED
    assert res.is_live is False
    assert "Liveness verification failed" in res.message


@pytest.mark.asyncio
async def test_attendance_service_accepts_live_attendance(
    test_session: AsyncSession,
    sample_student: Student,
    active_session: str,
):
    """Verify AttendanceService marks PRESENT when is_live is True."""
    service = AttendanceService(db=test_session)

    req = AttendanceMarkRequest(
        session_id=active_session,
        student_id=sample_student.student_id,
        similarity_score=0.92,
        observation_count=3,
        is_confirmed=True,
        is_live=True,  # Verified live presentation
        liveness_score=0.88,
        liveness_status="LIVE",
    )

    res = await service.mark_attendance(req, session=test_session)

    assert res.status == AttendanceStatus.PRESENT
    assert res.is_live is True
    assert res.record_id is not None
    assert "marked PRESENT" in res.message


@pytest.mark.asyncio
async def test_attendance_service_strict_mode_rejection(
    test_session: AsyncSession,
    sample_student: Student,
    active_session: str,
    monkeypatch,
):
    """Verify strict mode rejects attendance when is_live is omitted/None."""
    monkeypatch.setattr(settings, "LIVENESS_STRICT_MODE", True)
    service = AttendanceService(db=test_session)

    req = AttendanceMarkRequest(
        session_id=active_session,
        student_id=sample_student.student_id,
        similarity_score=0.92,
        observation_count=3,
        is_confirmed=True,
        is_live=None,  # Not provided
    )

    res = await service.mark_attendance(req, session=test_session)

    assert res.status == AttendanceStatus.REJECTED
    assert "Strict liveness verification required" in res.message


def test_camera_pipeline_with_custom_liveness_detector():
    """Verify CameraRecognitionPipeline integrates liveness results in ProcessedFrameResult."""
    # Create synthetic frame with dummy face
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    pipeline = CameraRecognitionPipeline(
        liveness_detector=PassthroughLivenessDetector(),
    )

    res = pipeline.process_frame(blank)
    assert hasattr(res, "liveness_results")
    assert isinstance(res.liveness_results, list)
