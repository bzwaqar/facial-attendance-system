"""Integration tests for Students and Enrollment FastAPI endpoints."""

from pathlib import Path
from typing import AsyncGenerator
import cv2
import httpx
import numpy as np
import pytest
import pytest_asyncio
from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app

SAMPLE_IMAGE_PATH = Path(__file__).resolve().parent / "data" / "sample_student.jpg"
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provide an isolated database session and an async test HTTP client."""
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
async def test_student_crud_api_flow(async_client: httpx.AsyncClient) -> None:
    """Test full student lifecycle via API: creation, duplicate rejection, query, and list."""
    # 1. Create a student
    payload = {
        "student_id": "API-CS-2026-001",
        "name": "Gordon Freeman",
        "department": "Physics & Engineering",
        "semester": "Graduate",
        "section": "BlackMesa",
        "is_active": True,
    }
    response = await async_client.post("/api/v1/students", json=payload)
    assert response.status_code == status.HTTP_201_CREATED, response.text
    data = response.json()
    assert data["student_id"] == "API-CS-2026-001"
    assert data["name"] == "Gordon Freeman"
    assert data["face_embeddings_count"] == 0
    assert data["is_enrolled"] is False

    # 2. Reject duplicate student_id (HTTP 409 Conflict)
    dup_response = await async_client.post("/api/v1/students", json=payload)
    assert dup_response.status_code == status.HTTP_409_CONFLICT
    assert "already exists" in dup_response.json()["detail"]

    # 3. Retrieve student by ID
    get_response = await async_client.get("/api/v1/students/API-CS-2026-001")
    assert get_response.status_code == status.HTTP_200_OK
    assert get_response.json()["name"] == "Gordon Freeman"

    # 4. Retrieve non-existent student (HTTP 404)
    missing_response = await async_client.get("/api/v1/students/DOES-NOT-EXIST")
    assert missing_response.status_code == status.HTTP_404_NOT_FOUND

    # 5. List students
    list_response = await async_client.get("/api/v1/students")
    assert list_response.status_code == status.HTTP_200_OK
    items = list_response.json()
    assert len(items) >= 1
    assert any(s["student_id"] == "API-CS-2026-001" for s in items)


@pytest.mark.asyncio
async def test_enrollment_api_validation(async_client: httpx.AsyncClient) -> None:
    """Test enrollment endpoint validation constraints: image count and non-existent student."""
    # 1. Create student first
    await async_client.post(
        "/api/v1/students",
        json={
            "student_id": "API-ENROLL-001",
            "name": "Isaac Clarke",
            "department": "Mining Engineering",
            "semester": "5th",
            "section": "USG-Ishimura",
        },
    )

    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    # 2. Submit only 3 images -> Expect HTTP 400 Bad Request
    files_3 = [
        ("files", (f"img_{i}.jpg", img_bytes, "image/jpeg"))
        for i in range(3)
    ]
    res_under = await async_client.post("/api/v1/students/API-ENROLL-001/enroll", files=files_3)
    assert res_under.status_code == status.HTTP_400_BAD_REQUEST
    assert "between 5 and 10" in res_under.json()["detail"]

    # 3. Submit for non-existent student -> Expect HTTP 404
    files_5 = [
        ("files", (f"img_{i}.jpg", img_bytes, "image/jpeg"))
        for i in range(5)
    ]
    res_404 = await async_client.post("/api/v1/students/NON-EXISTENT-STUDENT/enroll", files=files_5)
    assert res_404.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_enrollment_api_success_and_retrieval(async_client: httpx.AsyncClient) -> None:
    """Test complete enrollment via API with 5 valid face images and verify student enrollment status."""
    # 1. Create student
    await async_client.post(
        "/api/v1/students",
        json={
            "student_id": "API-ENROLL-SUCCESS",
            "name": "Alyx Vance",
            "department": "Robotics",
            "semester": "4th",
            "section": "City17",
        },
    )

    # 2. Prepare 5 varied face images
    base_img = cv2.imread(str(SAMPLE_IMAGE_PATH))
    upload_files = []
    for i in range(5):
        altered = cv2.convertScaleAbs(base_img, alpha=1.0, beta=(i - 2) * 2)
        _, buf = cv2.imencode(".jpg", altered)
        upload_files.append(("files", (f"face_{i+1}.jpg", buf.tobytes(), "image/jpeg")))

    # 3. POST /api/v1/students/{student_id}/enroll
    enroll_res = await async_client.post("/api/v1/students/API-ENROLL-SUCCESS/enroll", files=upload_files)
    assert enroll_res.status_code == status.HTTP_201_CREATED, enroll_res.text
    data = enroll_res.json()
    assert data["student_id"] == "API-ENROLL-SUCCESS"
    assert data["total_submitted"] == 5
    assert data["total_enrolled"] == 5
    assert data["model_name"] == "ArcFace"
    assert len(data["images_summary"]) == 5

    # 4. Check that GET /api/v1/students/{student_id} now reflects the enrolled state
    student_res = await async_client.get("/api/v1/students/API-ENROLL-SUCCESS")
    assert student_res.status_code == status.HTTP_200_OK
    student_data = student_res.json()
    assert student_data["face_embeddings_count"] == 5
    assert student_data["is_enrolled"] is True


@pytest.mark.asyncio
async def test_enrollment_api_rejects_corrupted_or_blank_image(
    async_client: httpx.AsyncClient,
) -> None:
    """Test that submitting an image without a face returns HTTP 422 with itemized diagnostics."""
    await async_client.post(
        "/api/v1/students",
        json={
            "student_id": "API-ENROLL-FAIL",
            "name": "Test Subject",
            "department": "Testing",
            "semester": "1st",
            "section": "T1",
        },
    )

    base_img = cv2.imread(str(SAMPLE_IMAGE_PATH))
    upload_files = []
    # 4 valid images
    for i in range(4):
        _, buf = cv2.imencode(".jpg", base_img)
        upload_files.append(("files", (f"valid_{i+1}.jpg", buf.tobytes(), "image/jpeg")))

    # 5th image: blank black image (no face)
    blank = np.zeros((400, 400, 3), dtype=np.uint8)
    _, blank_buf = cv2.imencode(".jpg", blank)
    upload_files.append(("files", ("blank_no_face.jpg", blank_buf.tobytes(), "image/jpeg")))

    res = await async_client.post("/api/v1/students/API-ENROLL-FAIL/enroll", files=upload_files)
    assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    detail = res.json()["detail"]
    assert detail["rejected_count"] >= 1
    assert detail["total_submitted"] == 5
    assert len(detail["diagnostics"]) == 5
    # Find the rejected image diagnostic
    blank_diag = [d for d in detail["diagnostics"] if d["filename"] == "blank_no_face.jpg"][0]
    assert blank_diag["status"] == "rejected"
    assert "no face" in blank_diag["reason"].lower()
