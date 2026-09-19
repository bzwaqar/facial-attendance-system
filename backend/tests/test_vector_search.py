"""Unit tests for standalone FAISS VectorSearchEngine and cosine similarity matching."""

import numpy as np
import pytest

from app.ml.vector_search import VectorMetadata, VectorSearchEngine
from app.schemas.matching import MatchStatus


@pytest.fixture
def empty_engine() -> VectorSearchEngine:
    """Provide a fresh, empty VectorSearchEngine instance."""
    return VectorSearchEngine(dimension=512)


@pytest.fixture
def populated_engine() -> VectorSearchEngine:
    """Provide a VectorSearchEngine populated with 3 known student identities."""
    engine = VectorSearchEngine(dimension=512)

    np.random.seed(100)
    # Generate 3 distinct basis-like vectors for 3 students
    v1 = np.random.randn(512).astype(np.float32)
    v2 = np.random.randn(512).astype(np.float32)
    v3 = np.random.randn(512).astype(np.float32)

    # Add 2 reference photos for student 1, 1 for student 2, 1 for student 3
    vectors = np.vstack([
        v1,
        v1 + 0.05 * np.random.randn(512).astype(np.float32),  # Slight variation of student 1
        v2,
        v3,
    ])

    metadata = [
        VectorMetadata(student_id="STU-001", student_pk=1, embedding_id=101),
        VectorMetadata(student_id="STU-001", student_pk=1, embedding_id=102),
        VectorMetadata(student_id="STU-002", student_pk=2, embedding_id=103),
        VectorMetadata(student_id="STU-003", student_pk=3, embedding_id=104),
    ]

    engine.build_index(vectors, metadata)
    return engine


def test_empty_index_returns_empty_status(empty_engine: VectorSearchEngine) -> None:
    """Verify searching against an empty FAISS index returns MatchStatus.EMPTY_INDEX without error."""
    assert empty_engine.total_vectors == 0
    assert empty_engine.is_indexed is False

    query = np.random.randn(512).astype(np.float32)
    result = empty_engine.search(query, top_k=5, threshold=0.60)

    assert result.status == MatchStatus.EMPTY_INDEX
    assert result.matched is False
    assert result.student_id is None
    assert result.similarity_score == 0.0
    assert len(result.candidates) == 0
    assert "empty" in result.message.lower()


def test_known_face_match_exact_and_near(populated_engine: VectorSearchEngine) -> None:
    """Verify that querying a known student vector yields MATCHED with high similarity."""
    # Query vector close to STU-001
    np.random.seed(100)
    v1 = np.random.randn(512).astype(np.float32)
    query_stu1 = v1 + 0.02 * np.random.randn(512).astype(np.float32)

    result = populated_engine.search(query_stu1, top_k=3, threshold=0.60)

    assert result.status == MatchStatus.MATCHED
    assert result.matched is True
    assert result.student_id == "STU-001"
    assert result.student_pk == 1
    assert result.similarity_score > 0.85
    assert len(result.candidates) <= 3
    # Top candidate must be STU-001
    assert result.candidates[0].student_id == "STU-001"


def test_unknown_face_low_similarity(populated_engine: VectorSearchEngine) -> None:
    """Verify that an orthogonal/unrelated face vector returns UNKNOWN and never forces a match."""
    # Generate an unrelated random vector distinct from seeds
    np.random.seed(9999)
    unknown_query = np.random.randn(512).astype(np.float32)

    # In 512 dimensions, random orthogonal Gaussian vectors have expected cosine similarity ~ 0.0
    result = populated_engine.search(unknown_query, top_k=5, threshold=0.60)

    assert result.status == MatchStatus.UNKNOWN
    assert result.matched is False
    assert result.student_id is None
    assert result.similarity_score < 0.60
    assert len(result.candidates) > 0
    assert "unrecognized" in result.message.lower()


def test_similarity_threshold_boundary_precision() -> None:
    """Verify boundary precision: score >= threshold is MATCHED, score < threshold is UNKNOWN."""
    engine = VectorSearchEngine(dimension=512)

    # Create synthetic reference vector
    ref_vec = np.zeros(512, dtype=np.float32)
    ref_vec[0] = 1.0  # Unit vector along axis 0

    engine.build_index(
        vectors=ref_vec.reshape(1, -1),
        metadata_list=[VectorMetadata(student_id="STU-TEST", student_pk=99, embedding_id=1)],
    )

    # 1. Query with dot product = 0.59 (below threshold 0.60)
    # Construct query: cos(theta) = 0.59 -> x = 0.59, y = sqrt(1 - 0.59^2)
    cos_val = 0.59
    sin_val = np.sqrt(1.0 - cos_val**2)
    query_below = np.zeros(512, dtype=np.float32)
    query_below[0] = cos_val
    query_below[1] = sin_val

    res_below = engine.search(query_below, threshold=0.60)
    assert res_below.status == MatchStatus.UNKNOWN
    assert res_below.matched is False
    assert res_below.student_id is None
    assert abs(res_below.similarity_score - 0.59) < 0.01

    # 2. Query with dot product = 0.61 (above threshold 0.60)
    cos_val_above = 0.61
    sin_val_above = np.sqrt(1.0 - cos_val_above**2)
    query_above = np.zeros(512, dtype=np.float32)
    query_above[0] = cos_val_above
    query_above[1] = sin_val_above

    res_above = engine.search(query_above, threshold=0.60)
    assert res_above.status == MatchStatus.MATCHED
    assert res_above.matched is True
    assert res_above.student_id == "STU-TEST"
    assert abs(res_above.similarity_score - 0.61) < 0.01


def test_top_k_candidates_ordering(populated_engine: VectorSearchEngine) -> None:
    """Verify that candidates are returned sorted in strictly descending similarity order."""
    np.random.seed(100)
    v1 = np.random.randn(512).astype(np.float32)

    result = populated_engine.search(v1, top_k=4, threshold=0.60)
    assert len(result.candidates) == 4

    scores = [c.similarity_score for c in result.candidates]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Candidates not sorted descending: {scores}"


def test_index_rebuilding_and_reset(populated_engine: VectorSearchEngine) -> None:
    """Verify rebuilding and resetting the FAISS index cleanly clears and replaces entries."""
    assert populated_engine.total_vectors == 4

    # 1. Reset
    populated_engine.reset()
    assert populated_engine.total_vectors == 0
    assert populated_engine.is_indexed is False

    # 2. Rebuild with new single identity
    new_vec = np.ones((1, 512), dtype=np.float32) / np.sqrt(512)
    new_meta = [VectorMetadata(student_id="NEW-001", student_pk=50, embedding_id=500)]

    populated_engine.build_index(new_vec, new_meta)
    assert populated_engine.total_vectors == 1

    res = populated_engine.search(new_vec[0], threshold=0.60)
    assert res.status == MatchStatus.MATCHED
    assert res.student_id == "NEW-001"
    assert abs(res.similarity_score - 1.0) < 1e-4
