import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models.face_embedding import FaceEmbedding
from app.schemas.student import StudentCreate
from app.services.exceptions import DuplicateStudentError, StudentNotFoundError
from app.services.student_service import StudentService

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


@pytest.mark.asyncio
async def test_create_student_success(test_session: AsyncSession) -> None:
    """Verify that a valid student can be registered."""
    student_in = StudentCreate(
        student_id="CS-2026-101",
        name="Elena Fisher",
        department="Computer Science",
        semester="6th",
        section="A",
        is_active=True,
    )
    student = await StudentService.create_student(test_session, student_in)
    await test_session.commit()

    assert student.id is not None
    assert student.student_id == "CS-2026-101"
    assert student.name == "Elena Fisher"
    assert student.department == "Computer Science"
    assert student.is_active is True


@pytest.mark.asyncio
async def test_create_duplicate_student_id_rejected(test_session: AsyncSession) -> None:
    """Verify that attempting to register an identical student_id raises DuplicateStudentError."""
    student_in_1 = StudentCreate(
        student_id="CS-2026-DUPLICATE",
        name="Original Student",
        department="Software Engineering",
        semester="4th",
        section="B",
    )
    await StudentService.create_student(test_session, student_in_1)
    await test_session.commit()

    # Attempt second registration with same student_id
    student_in_2 = StudentCreate(
        student_id="CS-2026-DUPLICATE",
        name="Duplicate Impostor",
        department="Software Engineering",
        semester="4th",
        section="B",
    )
    with pytest.raises(DuplicateStudentError) as exc_info:
        await StudentService.create_student(test_session, student_in_2)

    assert "CS-2026-DUPLICATE" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_student_by_student_id(test_session: AsyncSession) -> None:
    """Verify fetching a student by student_id and checking embedding relationships."""
    student_in = StudentCreate(
        student_id="CS-2026-102",
        name="Nathan Drake",
        department="Archaeology & Computing",
        semester="8th",
        section="C",
    )
    created = await StudentService.create_student(test_session, student_in)
    await test_session.commit()

    retrieved = await StudentService.get_student_by_student_id(test_session, "CS-2026-102")
    assert retrieved is not None
    assert retrieved.name == "Nathan Drake"
    assert len(retrieved.face_embeddings) == 0

    # Non-existent student returns None
    missing = await StudentService.get_student_by_student_id(test_session, "NON-EXISTENT")
    assert missing is None


@pytest.mark.asyncio
async def test_list_and_delete_student(test_session: AsyncSession) -> None:
    """Verify listing students and deleting a student with cascade delete."""
    s1 = await StudentService.create_student(
        test_session,
        StudentCreate(
            student_id="LIST-001",
            name="Alice",
            department="AI",
            semester="1st",
            section="A",
        ),
    )
    s2 = await StudentService.create_student(
        test_session,
        StudentCreate(
            student_id="LIST-002",
            name="Bob",
            department="Data Science",
            semester="2nd",
            section="B",
        ),
    )
    await test_session.commit()

    students = await StudentService.list_students(test_session)
    assert len(students) >= 2

    # Filter by department
    ai_students = await StudentService.list_students(test_session, department="AI")
    assert len(ai_students) == 1
    assert ai_students[0].student_id == "LIST-001"

    # Delete student
    deleted = await StudentService.delete_student(test_session, "LIST-001")
    assert deleted is True
    await test_session.commit()

    check = await StudentService.get_student_by_student_id(test_session, "LIST-001")
    assert check is None
