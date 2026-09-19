"""Automated tests for PostgreSQL database models, relationships, and constraints."""

import struct
from datetime import datetime, timezone
import pytest
import pytest_asyncio
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.exceptions import DuplicateAttendanceError
from app.db.models.attendance import Attendance
from app.db.models.face_embedding import FaceEmbedding
from app.db.models.student import Student
from app.db.session import async_engine, check_database_connection, handle_integrity_error

# In-memory SQLite async engine for isolated unit tests
TEST_ASYNC_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def test_session() -> AsyncSession:
    """Fixture providing an isolated in-memory async database session with created tables."""
    engine = create_async_engine(TEST_ASYNC_DB_URL, echo=False)

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


def test_schema_metadata_and_constraints() -> None:
    """Verify that all required tables, columns, indexes, and constraints exist in Base.metadata."""
    tables = Base.metadata.tables

    # Check table existence
    assert "students" in tables, "Table 'students' must exist in metadata"
    assert "face_embeddings" in tables, "Table 'face_embeddings' must exist in metadata"
    assert "attendances" in tables, "Table 'attendances' must exist in metadata"
    assert "attendance_sessions" in tables, "Table 'attendance_sessions' must exist in metadata"


    # Verify Student columns
    student_table = tables["students"]
    expected_student_cols = {
        "id", "student_id", "name", "department", "semester", "section", "is_active", "created_at"
    }
    assert expected_student_cols.issubset(student_table.columns.keys())

    # Verify FaceEmbedding columns
    embedding_table = tables["face_embeddings"]
    expected_embedding_cols = {
        "id", "student_id", "embedding", "model_name", "model_version", "quality_score", "created_at"
    }
    assert expected_embedding_cols.issubset(embedding_table.columns.keys())

    # Verify Attendance columns & constraints
    attendance_table = tables["attendances"]
    expected_attendance_cols = {
        "id", "student_id", "session_id", "timestamp", "similarity_score", "status"
    }
    assert expected_attendance_cols.issubset(attendance_table.columns.keys())

    # Verify duplicate attendance unique constraint
    unique_constraints = [c.name for c in attendance_table.constraints if getattr(c, "name", None)]
    assert "uq_student_session_attendance" in unique_constraints, (
        "Unique constraint 'uq_student_session_attendance' must exist on attendances table"
    )


def test_face_embedding_vector_serialization() -> None:
    """Test vector serialization and deserialization on FaceEmbedding."""
    sample_vector = [0.123, -0.456, 0.789, 0.001]
    embedding = FaceEmbedding(
        student_id=1,
        embedding=b"",
        model_name="ArcFace",
        model_version="glint360k_r100",
        quality_score=0.98,
    )
    embedding.set_vector(sample_vector)

    # Check raw binary format (4 floats = 16 bytes)
    assert len(embedding.embedding) == 4 * struct.calcsize("f")

    # Check vector deserialization
    retrieved_vector = embedding.get_vector()
    assert len(retrieved_vector) == 4
    for orig, retr in zip(sample_vector, retrieved_vector):
        assert abs(orig - retr) < 1e-5


@pytest.mark.asyncio
async def test_student_and_relationship_crud(test_session: AsyncSession) -> None:
    """Test inserting and querying a student with face embedding and attendance."""
    # Create student
    student = Student(
        student_id="CS-2026-001",
        name="Alex Morgan",
        department="Computer Science",
        semester="6th",
        section="A",
        is_active=True,
    )
    test_session.add(student)
    await test_session.flush()
    assert student.id is not None

    # Add face embedding
    raw_vector = [0.05] * 512
    face_emb = FaceEmbedding(
        student_id=student.id,
        embedding=b"",
        model_name="ArcFace",
        model_version="glint360k_r100",
        quality_score=0.95,
    )
    face_emb.set_vector(raw_vector)
    test_session.add(face_emb)

    # Add attendance record
    attendance = Attendance(
        student_id=student.id,
        session_id="CS601-20260913-01",
        timestamp=datetime.now(timezone.utc),
        similarity_score=0.875,
        status="PRESENT",
    )
    test_session.add(attendance)
    await test_session.commit()

    # Query back
    result = await test_session.execute(
        select(Student).where(Student.student_id == "CS-2026-001")
    )
    queried_student = result.scalar_one_or_none()
    assert queried_student is not None
    assert queried_student.name == "Alex Morgan"


@pytest.mark.asyncio
async def test_prevent_duplicate_attendance(test_session: AsyncSession) -> None:
    """Verify that inserting duplicate attendance for the same student and session is rejected."""
    student = Student(
        student_id="CS-2026-002",
        name="Jordan Lee",
        department="Computer Science",
        semester="6th",
        section="B",
    )
    test_session.add(student)
    await test_session.commit()

    # First attendance record - should succeed
    att1 = Attendance(
        student_id=student.id,
        session_id="SESSION-A1",
        similarity_score=0.92,
        status="PRESENT",
    )
    test_session.add(att1)
    await test_session.commit()

    # Second attendance record for same student and session - must fail
    att2 = Attendance(
        student_id=student.id,
        session_id="SESSION-A1",
        similarity_score=0.89,
        status="PRESENT",
    )
    test_session.add(att2)

    with pytest.raises(IntegrityError):
        await test_session.commit()

    await test_session.rollback()


def test_handle_integrity_error_translation() -> None:
    """Test custom exception handling for duplicate attendance error translation."""
    orig_error = Exception('duplicate key value violates unique constraint "uq_student_session_attendance"')
    mock_integrity_err = IntegrityError("statement", "params", orig_error)

    translated = handle_integrity_error(mock_integrity_err)
    assert isinstance(translated, DuplicateAttendanceError)
    assert "Attendance has already been recorded" in translated.message


@pytest.mark.asyncio
async def test_postgres_live_connection_and_crud() -> None:
    """Integration test connecting directly to PostgreSQL, testing CRUD, constraints, and cascade delete."""
    is_connected = await check_database_connection()
    if not is_connected:
        pytest.skip("PostgreSQL instance is not reachable on configured port")


    session_factory = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        # 1. Clean up any previous test student with this ID
        existing = await session.execute(
            select(Student).where(Student.student_id == "TEST-PG-999")
        )
        old_student = existing.scalar_one_or_none()
        if old_student:
            await session.delete(old_student)
            await session.commit()

        # 2. Insert Student into PostgreSQL
        pg_student = Student(
            student_id="TEST-PG-999",
            name="Postgres Test User",
            department="Artificial Intelligence",
            semester="4th",
            section="AI-1",
            is_active=True,
        )
        session.add(pg_student)
        await session.commit()
        assert pg_student.id is not None
        student_pk = pg_student.id

        # 3. Insert Face Embedding
        emb = FaceEmbedding(
            student_id=student_pk,
            embedding=b"",
            model_name="ArcFace",
            model_version="glint360k_r100",
            quality_score=0.99,
        )
        emb.set_vector([0.01 * i for i in range(512)])
        session.add(emb)

        # 4. Record Attendance
        att = Attendance(
            student_id=student_pk,
            session_id="PG-SESSION-LIVE-01",
            similarity_score=0.965,
            status="PRESENT",
        )
        session.add(att)
        await session.commit()

        # 5. Verify duplicate attendance constraint in PostgreSQL
        dup_att = Attendance(
            student_id=student_pk,
            session_id="PG-SESSION-LIVE-01",
            similarity_score=0.950,
            status="PRESENT",
        )
        session.add(dup_att)
        with pytest.raises(IntegrityError) as exc_info:
            await session.commit()
        assert "uq_student_session_attendance" in str(exc_info.value)
        await session.rollback()

        # 6. Test Cascade Delete
        res = await session.execute(select(Student).where(Student.id == student_pk))
        student_to_delete = res.scalar_one()
        await session.delete(student_to_delete)
        await session.commit()

        # Verify child face_embeddings and attendances were cascade deleted
        emb_check = await session.execute(
            select(FaceEmbedding).where(FaceEmbedding.student_id == student_pk)
        )
        assert emb_check.scalar_one_or_none() is None

        att_check = await session.execute(
            select(Attendance).where(Attendance.student_id == student_pk)
        )
        assert att_check.scalar_one_or_none() is None
