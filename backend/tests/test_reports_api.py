"""Tests for Reporting and Analytics API endpoints."""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models.attendance import Attendance
from app.db.models.attendance_session import AttendanceSession
from app.db.models.student import Student
from app.db.session import get_db
from app.main import app

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def db_with_records():
    """Fixture initializing SQLite in-memory with test student and attendance data."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        # Create student
        student = Student(
            student_id="REP-STU-01",
            name="Margaret Hamilton",
            department="Software Engineering",
            semester="8th",
            section="B",
            is_active=True,
        )
        session.add(student)

        # Create session
        sess = AttendanceSession(
            session_id="REP-SESS-01",
            course_name="Software Verification",
            instructor_name="Prof. Turing",
            is_active=True,
        )
        session.add(sess)
        await session.flush()

        # Create attendance record
        att = Attendance(
            student_id=student.id,
            session_id="REP-SESS-01",
            similarity_score=0.91,
            status="PRESENT",
            is_live=True,
            liveness_score=0.95,
        )
        session.add(att)
        await session.commit()

        yield session_factory

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def test_get_reports_overview(db_with_records):
    """Verify overview report returns accurate student, session, and attendance counts."""
    async def override_get_db():
        async with db_with_records() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    try:
        response = client.get("/api/v1/reports/overview")
        assert response.status_code == 200
        data = response.json()
        assert data["total_students"] >= 1
        assert data["active_students"] >= 1
        assert data["total_sessions"] >= 1
        assert data["active_sessions"] >= 1
        assert data["total_attendance_records"] >= 1
        assert len(data["recent_activity"]) >= 1
        assert data["recent_activity"][0]["student_id"] == "REP-STU-01"
        assert data["recent_activity"][0]["is_live"] is True
    finally:
        app.dependency_overrides.clear()


def test_get_session_detailed_report(db_with_records):
    """Verify detailed report for a specific session returns present attendee list."""
    async def override_get_db():
        async with db_with_records() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    try:
        response = client.get("/api/v1/reports/sessions/REP-SESS-01")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "REP-SESS-01"
        assert data["total_present"] == 1
        assert len(data["present_students"]) == 1
        assert data["present_students"][0]["student_name"] == "Margaret Hamilton"
    finally:
        app.dependency_overrides.clear()


def test_get_session_report_not_found(db_with_records):
    """Verify 404 returned for nonexistent session."""
    async def override_get_db():
        async with db_with_records() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    try:
        response = client.get("/api/v1/reports/sessions/NONEXISTENT-SESSION-999")
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()
