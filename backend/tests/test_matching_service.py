"""Integration tests for IdentityMatchingService with database embedding persistence."""

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models.face_embedding import FaceEmbedding
from app.db.models.student import Student
from app.ml.vector_search import VectorSearchEngine
from app.schemas.matching import MatchStatus
from app.services.matching_service import IdentityMatchingService

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


@pytest.fixture
def matching_service() -> IdentityMatchingService:
    """Provide an isolated IdentityMatchingService with a dedicated VectorSearchEngine."""
    engine = VectorSearchEngine(dimension=512)
    return IdentityMatchingService(vector_engine=engine)


@pytest.mark.asyncio
async def test_rebuild_from_empty_database(
    test_session: AsyncSession,
    matching_service: IdentityMatchingService,
) -> None:
    """Verify rebuilding index from an empty database yields 0 vectors and EMPTY_INDEX search."""
    count = await matching_service.rebuild_index_from_db(test_session)
    assert count == 0
    assert matching_service.vector_engine.total_vectors == 0

    query = np.random.randn(512).astype(np.float32)
    result = matching_service.match_face_embedding(query)
    assert result.status == MatchStatus.EMPTY_INDEX
    assert result.matched is False


@pytest.mark.asyncio
async def test_rebuild_and_match_from_database_embeddings(
    test_session: AsyncSession,
    matching_service: IdentityMatchingService,
) -> None:
    """Verify loading student embeddings from PostgreSQL/SQLite and matching identities."""
    # 1. Register student in DB
    student = Student(
        student_id="CS-MATCH-001",
        name="Ada Lovelace",
        department="Computer Science",
        semester="1st",
        section="A",
        is_active=True,
    )
    test_session.add(student)
    await test_session.flush()

    # 2. Add 2 normalized face embeddings for Ada
    np.random.seed(42)
    vec1 = np.random.randn(512).astype(np.float32)
    vec1 /= float(np.linalg.norm(vec1))

    vec2 = vec1 + 0.03 * np.random.randn(512).astype(np.float32)
    vec2 /= float(np.linalg.norm(vec2))

    emb1 = FaceEmbedding(
        student_id=student.id,
        embedding=b"",
        model_name="ArcFace",
        model_version="buffalo_sc",
        quality_score=0.92,
    )
    emb1.set_vector(vec1.tolist())

    emb2 = FaceEmbedding(
        student_id=student.id,
        embedding=b"",
        model_name="ArcFace",
        model_version="buffalo_sc",
        quality_score=0.94,
    )
    emb2.set_vector(vec2.tolist())

    test_session.add_all([emb1, emb2])
    await test_session.commit()

    # 3. Rebuild FAISS index from DB
    indexed_count = await matching_service.rebuild_index_from_db(test_session)
    assert indexed_count == 2
    assert matching_service.vector_engine.total_vectors == 2

    # 4. Search with a near vector -> MATCHED
    query_near = vec1 + 0.01 * np.random.randn(512).astype(np.float32)
    match_res = matching_service.match_face_embedding(query_near, threshold=0.60)
    assert match_res.status == MatchStatus.MATCHED
    assert match_res.matched is True
    assert match_res.student_id == "CS-MATCH-001"
    assert match_res.student_pk == student.id
    assert match_res.similarity_score > 0.85

    # 5. Search with an unrelated random vector -> UNKNOWN
    np.random.seed(999)
    query_unknown = np.random.randn(512).astype(np.float32)
    unknown_res = matching_service.match_face_embedding(query_unknown, threshold=0.60)
    assert unknown_res.status == MatchStatus.UNKNOWN
    assert unknown_res.matched is False
    assert unknown_res.student_id is None


@pytest.mark.asyncio
async def test_inactive_student_excluded_from_index(
    test_session: AsyncSession,
    matching_service: IdentityMatchingService,
) -> None:
    """Verify that embeddings belonging to inactive students are excluded from the FAISS index."""
    inactive_student = Student(
        student_id="INACTIVE-001",
        name="Deactivated User",
        department="History",
        semester="Graduated",
        section="Alumni",
        is_active=False,  # Inactive
    )
    test_session.add(inactive_student)
    await test_session.flush()

    emb = FaceEmbedding(
        student_id=inactive_student.id,
        embedding=b"",
        model_name="ArcFace",
        model_version="buffalo_sc",
        quality_score=0.90,
    )
    emb.set_vector([0.1] * 512)
    test_session.add(emb)
    await test_session.commit()

    count = await matching_service.rebuild_index_from_db(test_session)
    assert count == 0
    assert matching_service.vector_engine.total_vectors == 0
