"""Integration tests for Attendance FastAPI endpoints."""

from typing import AsyncGenerator
import httpx
import pytest
import pytest_asyncio
from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models.student import Student
from app.db.session import get_db
from app.main import app

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provide an isolated in-memory async database session and test HTTP client."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.pop(get_db, None)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_attendance_session_api_crud(async_client: httpx.AsyncClient) -> None:
    """Test creating, listing, and retrieving attendance sessions via HTTP API."""
    # 1. Create a session
    payload = {
        "session_id": "CS101-L01",
        "course_name": "Intro to Computer Systems",
        "instructor_name": "Prof. Turing",
        "min_similarity_threshold": 0.65,
        "min_confirmations": 3,
    }
    create_res = await async_client.post("/api/v1/attendance/sessions", json=payload)
    assert create_res.status_code == status.HTTP_201_CREATED
    data = create_res.json()
    assert data["session_id"] == "CS101-L01"
    assert data["course_name"] == "Intro to Computer Systems"
    assert data["is_active"] is True

    # 2. Prevent duplicate session_id
    dup_res = await async_client.post("/api/v1/attendance/sessions", json=payload)
    assert dup_res.status_code == status.HTTP_409_CONFLICT
    assert "already exists" in dup_res.json()["detail"]

    # 3. List sessions
    list_res = await async_client.get("/api/v1/attendance/sessions")
    assert list_res.status_code == status.HTTP_200_OK
    sessions = list_res.json()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "CS101-L01"

    # 4. Get session attendance
    get_res = await async_client.get("/api/v1/attendance/sessions/CS101-L01")
    assert get_res.status_code == status.HTTP_200_OK
    summary = get_res.json()
    assert summary["session_id"] == "CS101-L01"
    assert summary["total_present"] == 0
    assert summary["records"] == []


@pytest.mark.asyncio
async def test_attendance_marking_api_flow(async_client: httpx.AsyncClient) -> None:
    """Test full attendance marking pipeline: student registration, mark, duplicate check, and session close."""
    # 1. Register student
    student_payload = {
        "student_id": "STU-API-001",
        "name": "Grace Hopper",
        "department": "Computer Science",
        "semester": "8th",
        "section": "A",
    }
    stu_res = await async_client.post("/api/v1/students", json=student_payload)
    assert stu_res.status_code == status.HTTP_201_CREATED

    # 2. Create attendance session
    sess_payload = {
        "session_id": "SESSION-FLOW-01",
        "course_name": "Compiler Design",
        "min_similarity_threshold": 0.60,
        "min_confirmations": 3,
    }
    sess_res = await async_client.post("/api/v1/attendance/sessions", json=sess_payload)
    assert sess_res.status_code == status.HTTP_201_CREATED

    # 3. Mark attendance PRESENT
    mark_payload = {
        "session_id": "SESSION-FLOW-01",
        "student_id": "STU-API-001",
        "similarity_score": 0.88,
        "observation_count": 4,
        "is_confirmed": True,
    }
    mark_res = await async_client.post("/api/v1/attendance/mark", json=mark_payload)
    assert mark_res.status_code == status.HTTP_200_OK
    data = mark_res.json()
    assert data["status"] == "PRESENT"
    assert data["student_id"] == "STU-API-001"
    assert data["student_name"] == "Grace Hopper"
    assert data["record_id"] is not None

    # 4. Repeat attendance marking -> ALREADY_MARKED
    dup_res = await async_client.post("/api/v1/attendance/mark", json=mark_payload)
    assert dup_res.status_code == status.HTTP_200_OK
    dup_data = dup_res.json()
    assert dup_data["status"] == "ALREADY_MARKED"
    assert dup_data["record_id"] == data["record_id"]

    # 5. Unknown student recognition -> UNKNOWN
    unknown_payload = {
        "session_id": "SESSION-FLOW-01",
        "student_id": None,
        "similarity_score": 0.40,
        "observation_count": 3,
        "is_confirmed": False,
    }
    unk_res = await async_client.post("/api/v1/attendance/mark", json=unknown_payload)
    assert unk_res.status_code == status.HTTP_200_OK
    assert unk_res.json()["status"] == "UNKNOWN"

    # 6. Low similarity score -> REJECTED
    low_score_payload = {
        "session_id": "SESSION-FLOW-01",
        "student_id": "STU-API-001",
        "similarity_score": 0.42,  # Below 0.60
        "observation_count": 5,
        "is_confirmed": True,
    }
    low_res = await async_client.post("/api/v1/attendance/mark", json=low_score_payload)
    assert low_res.status_code == status.HTTP_200_OK
    assert low_res.json()["status"] == "REJECTED"

    # 7. Unstable / insufficient observations -> REJECTED
    unstable_payload = {
        "session_id": "SESSION-FLOW-01",
        "student_id": "STU-API-001",
        "similarity_score": 0.85,
        "observation_count": 1,  # Below required 3
        "is_confirmed": True,
    }
    unst_res = await async_client.post("/api/v1/attendance/mark", json=unstable_payload)
    assert unst_res.status_code == status.HTTP_200_OK
    assert unst_res.json()["status"] == "REJECTED"

    # 8. Query session attendance report
    rep_res = await async_client.get("/api/v1/attendance/sessions/SESSION-FLOW-01")
    assert rep_res.status_code == status.HTTP_200_OK
    report = rep_res.json()
    assert report["total_present"] == 1
    assert report["records"][0]["student_id"] == "STU-API-001"
    assert report["records"][0]["student_name"] == "Grace Hopper"

    # 9. Close session
    close_res = await async_client.patch("/api/v1/attendance/sessions/SESSION-FLOW-01/close")
    assert close_res.status_code == status.HTTP_200_OK
    assert close_res.json()["is_active"] is False

    # 10. Attempt mark on closed session -> REJECTED
    closed_mark_res = await async_client.post("/api/v1/attendance/mark", json=mark_payload)
    assert closed_mark_res.status_code == status.HTTP_200_OK
    assert closed_mark_res.json()["status"] == "REJECTED"
