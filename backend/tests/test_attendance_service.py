"""Unit tests for AttendanceService verifying session management, business rules, and duplicate checks."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models.attendance import Attendance
from app.db.models.student import Student
from app.schemas.attendance import (
    AttendanceMarkRequest,
    AttendanceSessionCreate,
    AttendanceStatus,
)
from app.services.attendance_service import AttendanceService
from app.services.exceptions import DuplicateSessionError, SessionNotFoundError

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def test_session() -> AsyncSession:
    """Fixture providing an isolated in-memory async SQLite database."""
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
    """Create a sample registered student in the database."""
    student = Student(
        student_id="STU-101",
        name="Ada Lovelace",
        department="Computer Science",
        semester="6th",
        section="A",
        is_active=True,
    )
    test_session.add(student)
    await test_session.commit()
    return student


@pytest.mark.asyncio
async def test_create_and_get_session(test_session: AsyncSession) -> None:
    """Test creating and retrieving an attendance session."""
    service = AttendanceService(db=test_session)

    session_in = AttendanceSessionCreate(
        session_id="CS-601-2026",
        course_name="Computer Vision",
        instructor_name="Dr. Smith",
        min_similarity_threshold=0.65,
        min_confirmations=4,
    )
    created = await service.create_session(session_in)
    assert created.session_id == "CS-601-2026"
    assert created.course_name == "Computer Vision"
    assert created.is_active is True
    assert created.min_similarity_threshold == 0.65
    assert created.min_confirmations == 4

    fetched = await service.get_session("CS-601-2026")
    assert fetched.session_id == "CS-601-2026"
    assert fetched.is_active is True


@pytest.mark.asyncio
async def test_duplicate_session_prevention(test_session: AsyncSession) -> None:
    """Test that creating a session with an existing session_id raises DuplicateSessionError."""
    service = AttendanceService(db=test_session)

    session_in = AttendanceSessionCreate(session_id="DUP-SESSION-01")
    await service.create_session(session_in)

    with pytest.raises(DuplicateSessionError) as exc_info:
        await service.create_session(session_in)
    assert "already exists" in exc_info.value.message


@pytest.mark.asyncio
async def test_session_not_found(test_session: AsyncSession) -> None:
    """Test that querying a non-existent session raises SessionNotFoundError."""
    service = AttendanceService(db=test_session)

    with pytest.raises(SessionNotFoundError) as exc_info:
        await service.get_session("NON-EXISTENT-SESSION")
    assert "was not found" in exc_info.value.message


@pytest.mark.asyncio
async def test_mark_attendance_present_and_prevent_duplicate(
    test_session: AsyncSession,
    sample_student: Student,
) -> None:
    """Test that verified recognition marks PRESENT exactly once and subsequent attempts return ALREADY_MARKED."""
    service = AttendanceService(db=test_session)

    # 1. Create active session
    await service.create_session(AttendanceSessionCreate(session_id="SESSION-DUP-TEST"))

    # 2. Mark attendance for student (stable recognition with 4 observations, score 0.88)
    req1 = AttendanceMarkRequest(
        session_id="SESSION-DUP-TEST",
        student_id=sample_student.student_id,
        similarity_score=0.88,
        observation_count=4,
        is_confirmed=True,
    )
    res1 = await service.mark_attendance(req1)

    assert res1.status == AttendanceStatus.PRESENT
    assert res1.student_id == sample_student.student_id
    assert res1.student_name == sample_student.name
    assert res1.record_id is not None
    assert "marked PRESENT" in res1.message

    # 3. Second attempt for identical student and session - must return ALREADY_MARKED
    req2 = AttendanceMarkRequest(
        session_id="SESSION-DUP-TEST",
        student_id=sample_student.student_id,
        similarity_score=0.92,
        observation_count=5,
        is_confirmed=True,
    )
    res2 = await service.mark_attendance(req2)

    assert res2.status == AttendanceStatus.ALREADY_MARKED
    assert res2.student_id == sample_student.student_id
    assert res2.record_id == res1.record_id
    assert "already marked PRESENT" in res2.message

    # 4. Verify only one attendance record exists in database
    summary = await service.get_session_attendance("SESSION-DUP-TEST")
    assert summary.total_present == 1
    assert len(summary.records) == 1
    assert summary.records[0].student_id == sample_student.student_id


@pytest.mark.asyncio
async def test_mark_attendance_unknown_identity(test_session: AsyncSession) -> None:
    """Test that UNKNOWN recognition identity returns UNKNOWN status and records nothing."""
    service = AttendanceService(db=test_session)
    await service.create_session(AttendanceSessionCreate(session_id="SESSION-UNKNOWN-TEST"))

    req = AttendanceMarkRequest(
        session_id="SESSION-UNKNOWN-TEST",
        student_id=None,
        similarity_score=0.45,
        observation_count=5,
        is_confirmed=False,
    )
    res = await service.mark_attendance(req)

    assert res.status == AttendanceStatus.UNKNOWN
    assert res.record_id is None
    assert "UNKNOWN" in res.message

    # Verify no records added
    summary = await service.get_session_attendance("SESSION-UNKNOWN-TEST")
    assert summary.total_present == 0


@pytest.mark.asyncio
async def test_mark_attendance_rejected_low_similarity(
    test_session: AsyncSession,
    sample_student: Student,
) -> None:
    """Test that recognition similarity below configured threshold returns REJECTED."""
    service = AttendanceService(db=test_session)
    # Session requires min similarity 0.75
    await service.create_session(
        AttendanceSessionCreate(
            session_id="SESSION-LOW-SCORE",
            min_similarity_threshold=0.75,
        )
    )

    req = AttendanceMarkRequest(
        session_id="SESSION-LOW-SCORE",
        student_id=sample_student.student_id,
        similarity_score=0.68,  # Below 0.75
        observation_count=5,
        is_confirmed=True,
    )
    res = await service.mark_attendance(req)

    assert res.status == AttendanceStatus.REJECTED
    assert "below required threshold" in res.message


@pytest.mark.asyncio
async def test_mark_attendance_rejected_insufficient_temporal_observations(
    test_session: AsyncSession,
    sample_student: Student,
) -> None:
    """Test that unstable / insufficient temporal observations return REJECTED."""
    service = AttendanceService(db=test_session)
    await service.create_session(
        AttendanceSessionCreate(
            session_id="SESSION-TEMPORAL-TEST",
            min_confirmations=4,
        )
    )

    # 1. Received only 2 observations when 4 required
    req = AttendanceMarkRequest(
        session_id="SESSION-TEMPORAL-TEST",
        student_id=sample_student.student_id,
        similarity_score=0.85,
        observation_count=2,  # Insufficient
        is_confirmed=True,
    )
    res = await service.mark_attendance(req)
    assert res.status == AttendanceStatus.REJECTED
    assert "Temporal verification requirement not met" in res.message

    # 2. Received is_confirmed=False
    req_unconfirmed = AttendanceMarkRequest(
        session_id="SESSION-TEMPORAL-TEST",
        student_id=sample_student.student_id,
        similarity_score=0.85,
        observation_count=5,
        is_confirmed=False,
    )
    res_unconfirmed = await service.mark_attendance(req_unconfirmed)
    assert res_unconfirmed.status == AttendanceStatus.REJECTED


@pytest.mark.asyncio
async def test_mark_attendance_closed_session(
    test_session: AsyncSession,
    sample_student: Student,
) -> None:
    """Test that attempting to mark attendance in a closed session returns REJECTED."""
    service = AttendanceService(db=test_session)
    await service.create_session(AttendanceSessionCreate(session_id="SESSION-CLOSED-TEST"))

    # Close session
    closed_sess = await service.close_session("SESSION-CLOSED-TEST")
    assert closed_sess.is_active is False
    assert closed_sess.closed_at is not None

    req = AttendanceMarkRequest(
        session_id="SESSION-CLOSED-TEST",
        student_id=sample_student.student_id,
        similarity_score=0.88,
        observation_count=5,
        is_confirmed=True,
    )
    res = await service.mark_attendance(req)
    assert res.status == AttendanceStatus.REJECTED
    assert "closed" in res.message


@pytest.mark.asyncio
async def test_mark_attendance_inactive_student(
    test_session: AsyncSession,
) -> None:
    """Test that deactivated student account returns REJECTED."""
    inactive_stu = Student(
        student_id="INACTIVE-99",
        name="Ghost Student",
        department="Physics",
        semester="1st",
        section="C",
        is_active=False,
    )
    test_session.add(inactive_stu)
    await test_session.commit()

    service = AttendanceService(db=test_session)
    await service.create_session(AttendanceSessionCreate(session_id="SESSION-INACTIVE-TEST"))

    req = AttendanceMarkRequest(
        session_id="SESSION-INACTIVE-TEST",
        student_id="INACTIVE-99",
        similarity_score=0.88,
        observation_count=5,
        is_confirmed=True,
    )
    res = await service.mark_attendance(req)
    assert res.status == AttendanceStatus.REJECTED
    assert "deactivated" in res.message


@pytest.mark.asyncio
async def test_list_sessions_with_attendee_counts(
    test_session: AsyncSession,
    sample_student: Student,
) -> None:
    """Test listing sessions and verifying accurate attendee counts."""
    service = AttendanceService(db=test_session)
    await service.create_session(AttendanceSessionCreate(session_id="SESS-A", course_name="Course A"))
    await service.create_session(AttendanceSessionCreate(session_id="SESS-B", course_name="Course B"))

    # Mark attendance in SESS-A
    await service.mark_attendance(
        AttendanceMarkRequest(
            session_id="SESS-A",
            student_id=sample_student.student_id,
            similarity_score=0.85,
            observation_count=4,
        )
    )

    sessions = await service.list_sessions()
    sess_dict = {s.session_id: s for s in sessions}

    assert "SESS-A" in sess_dict
    assert sess_dict["SESS-A"].total_attendees == 1
    assert "SESS-B" in sess_dict
    assert sess_dict["SESS-B"].total_attendees == 0
