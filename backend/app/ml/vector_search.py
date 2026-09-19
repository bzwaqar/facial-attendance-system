"""Dedicated FAISS vector search engine utilizing normalized inner product (cosine similarity).

NOTE ON RECOGNITION THRESHOLD:
The default recognition threshold is an operational baseline hyperparameter loaded
from settings. It does NOT represent a scientifically proven optimal value.
In production deployment, this threshold must be calibrated empirically using ROC/FAR/FRR
analysis on domain-specific validation sets under representative lighting and camera conditions.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import faiss
import numpy as np

from app.core.config import settings
from app.schemas.matching import IdentityMatchResult, MatchCandidate, MatchStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorMetadata:
    """Metadata mapped to each continuous FAISS vector index position."""

    student_id: str
    student_pk: Optional[int] = None
    embedding_id: Optional[int] = None


class VectorSearchEngine:
    """Thread-safe FAISS vector index manager performing cosine similarity searches."""

    def __init__(self, dimension: int = 512):
        """Initialize the vector search engine.
        
        Args:
            dimension: Feature dimensionality of the embedding vectors (512 for ArcFace).
        """
        self.dimension: int = dimension
        self._lock: threading.Lock = threading.Lock()
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self._id_map: List[VectorMetadata] = []

    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        """L2 normalize a single vector or batch of vectors to unit length.
        
        Args:
            vector: Float32 NumPy array of shape (D,) or (N, D).
            
        Returns:
            L2-normalized unit-length vector array of shape (N, D) in float32.
        """
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        norms = np.where(norms > 1e-10, norms, 1.0)
        return arr / norms

    def build_index(
        self,
        vectors: np.ndarray,
        metadata_list: List[VectorMetadata],
    ) -> int:
        """Reconstruct the FAISS index from scratch with provided vectors and metadata.
        
        Args:
            vectors: 2D NumPy array of shape (N, dimension).
            metadata_list: Parallel list of VectorMetadata of length N.
            
        Returns:
            Number of vectors successfully indexed.
            
        Raises:
            ValueError: If vector count does not match metadata length or dimension is invalid.
        """
        if len(metadata_list) == 0 or vectors.size == 0:
            with self._lock:
                self.reset()
            logger.info("Built empty FAISS vector index.")
            return 0

        norm_vectors = self.normalize_vector(vectors)
        if norm_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {norm_vectors.shape[1]}."
            )

        if norm_vectors.shape[0] != len(metadata_list):
            raise ValueError(
                f"Mismatch between vector count ({norm_vectors.shape[0]}) "
                f"and metadata count ({len(metadata_list)})."
            )

        with self._lock:
            # Create fresh IndexFlatIP for exact cosine similarity (Inner Product on unit vectors)
            new_index = faiss.IndexFlatIP(self.dimension)
            new_index.add(norm_vectors)
            self._index = new_index
            self._id_map = list(metadata_list)

        logger.info("Successfully built FAISS index with %d vectors.", self.total_vectors)
        return self.total_vectors

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata_list: List[VectorMetadata],
    ) -> int:
        """Append additional vectors to the active index.
        
        Args:
            vectors: 2D NumPy array of shape (M, dimension).
            metadata_list: Parallel list of VectorMetadata of length M.
            
        Returns:
            Total vector count in index after addition.
        """
        if len(metadata_list) == 0 or vectors.size == 0:
            return self.total_vectors

        norm_vectors = self.normalize_vector(vectors)
        if norm_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {norm_vectors.shape[1]}."
            )

        with self._lock:
            self._index.add(norm_vectors)
            self._id_map.extend(metadata_list)

        logger.info("Added %d vectors to FAISS index. Total: %d", len(metadata_list), self.total_vectors)
        return self.total_vectors

    def search(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> IdentityMatchResult:
        """Search top-K most similar enrolled identities for a given query face embedding.
        
        Evaluates cosine similarity via normalized inner product. Never forces an unknown face
        to match; if top candidate similarity < threshold, status is UNKNOWN.
        
        Args:
            query_vector: 512-dimensional embedding array or list.
            top_k: Number of nearest candidates to return for inspection/auditing.
            threshold: Recognition threshold. If None, uses settings.FACE_SIMILARITY_THRESHOLD.
            
        Returns:
            IdentityMatchResult with match decision, confidence score, and top-K candidates.
        """
        start_time = time.perf_counter()
        thresh = (
            float(threshold)
            if threshold is not None
            else settings.FACE_SIMILARITY_THRESHOLD
        )

        with self._lock:
            total = self.total_vectors
            if total == 0:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return IdentityMatchResult(
                    status=MatchStatus.EMPTY_INDEX,
                    matched=False,
                    student_id=None,
                    student_pk=None,
                    similarity_score=0.0,
                    threshold_used=thresh,
                    candidates=[],
                    search_time_ms=round(elapsed_ms, 3),
                    message="Vector index is empty. No reference embeddings enrolled.",
                )

            # Preprocess and normalize query vector
            q_norm = self.normalize_vector(np.asarray(query_vector, dtype=np.float32))
            k = min(max(1, top_k), total)

            # Query FAISS index: Inner Product = Cosine Similarity for unit vectors
            similarities, indices = self._index.search(q_norm, k)

            candidates: List[MatchCandidate] = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < 0 or idx >= len(self._id_map):
                    continue
                meta = self._id_map[idx]
                candidates.append(
                    MatchCandidate(
                        student_id=meta.student_id,
                        student_pk=meta.student_pk,
                        similarity_score=round(float(sim), 4),
                        vector_id=int(idx),
                        embedding_id=meta.embedding_id,
                    )
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if not candidates:
            return IdentityMatchResult(
                status=MatchStatus.UNKNOWN,
                matched=False,
                student_id=None,
                student_pk=None,
                similarity_score=0.0,
                threshold_used=thresh,
                candidates=[],
                search_time_ms=round(elapsed_ms, 3),
                message="No search candidates found in vector index.",
            )

        best_candidate = candidates[0]
        best_score = best_candidate.similarity_score

        # Strict recognition threshold gating: never force a match
        if best_score >= thresh:
            return IdentityMatchResult(
                status=MatchStatus.MATCHED,
                matched=True,
                student_id=best_candidate.student_id,
                student_pk=best_candidate.student_pk,
                similarity_score=best_score,
                threshold_used=thresh,
                candidates=candidates,
                search_time_ms=round(elapsed_ms, 3),
                message=(
                    f"Identity matched to student '{best_candidate.student_id}' "
                    f"with cosine similarity {best_score:.4f} (threshold: {thresh:.4f})."
                ),
            )
        else:
            return IdentityMatchResult(
                status=MatchStatus.UNKNOWN,
                matched=False,
                student_id=None,
                student_pk=None,
                similarity_score=best_score,
                threshold_used=thresh,
                candidates=candidates,
                search_time_ms=round(elapsed_ms, 3),
                message=(
                    f"Query face unrecognized. Highest similarity {best_score:.4f} "
                    f"fell below required threshold {thresh:.4f} (closest: '{best_candidate.student_id}')."
                ),
            )

    def reset(self) -> None:
        """Clear all indexed vectors and ID mappings."""
        with self._lock:
            self._index = faiss.IndexFlatIP(self.dimension)
            self._id_map = []
        logger.info("FAISS vector search engine reset to clean state.")

    @property
    def total_vectors(self) -> int:
        """Return total number of vectors currently held in the FAISS index."""
        return self._index.ntotal

    @property
    def is_indexed(self) -> bool:
        """Return True if at least one reference vector is indexed."""
        return self.total_vectors > 0


_global_engine: Optional[VectorSearchEngine] = None
_engine_lock: threading.Lock = threading.Lock()


def get_vector_search_engine() -> VectorSearchEngine:
    """Return the global singleton instance of VectorSearchEngine."""
    global _global_engine
    if _global_engine is None:
        with _engine_lock:
            if _global_engine is None:
                _global_engine = VectorSearchEngine(dimension=512)
    return _global_engine
