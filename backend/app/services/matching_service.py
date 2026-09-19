"""Identity matching service coordinating between PostgreSQL embeddings and the FAISS vector index.

NOTE ON RECOGNITION THRESHOLD:
The recognition threshold defaults to settings.FACE_SIMILARITY_THRESHOLD (0.60).
This value is an operational starting baseline. It must be empirically calibrated using
ROC (Receiver Operating Characteristic) curve analysis, plotting True Accept Rate vs False Accept Rate
on representative benchmark samples before production deployment.
"""

import logging
from typing import List, Optional, Union

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.face_embedding import FaceEmbedding
from app.db.models.student import Student
from app.ml.engine import FaceEngine, get_face_engine
from app.ml.vector_search import (
    VectorMetadata,
    VectorSearchEngine,
    get_vector_search_engine,
)
from app.schemas.matching import IdentityMatchResult, MatchStatus

logger = logging.getLogger(__name__)


class IdentityMatchingService:
    """Service orchestrating student identity matching using FAISS and cosine similarity."""

    def __init__(
        self,
        vector_engine: Optional[VectorSearchEngine] = None,
        face_engine: Optional[FaceEngine] = None,
    ):
        """Initialize the identity matching service.
        
        Args:
            vector_engine: Standalone FAISS vector search engine (defaults to global singleton).
            face_engine: Biometric face engine for live image feature extraction.
        """
        self.vector_engine = vector_engine or get_vector_search_engine()
        self.face_engine = face_engine or get_face_engine()

    async def rebuild_index_from_db(self, session: AsyncSession) -> int:
        """Load all active student embeddings from PostgreSQL and rebuild the FAISS vector index.
        
        Args:
            session: Active SQLAlchemy async session.
            
        Returns:
            Number of biometric reference vectors loaded and indexed.
        """
        logger.info("Loading reference embeddings from database to rebuild FAISS index...")

        # Query all active students and their associated face embeddings
        query = (
            select(FaceEmbedding, Student)
            .join(Student, FaceEmbedding.student_id == Student.id)
            .where(Student.is_active.is_(True))
            .order_by(FaceEmbedding.id.asc())
        )
        result = await session.execute(query)
        records = result.all()

        if not records:
            logger.warning("No active enrolled face embeddings found in database. Resetting index.")
            self.vector_engine.reset()
            return 0

        vectors: List[List[float]] = []
        metadata_list: List[VectorMetadata] = []

        for emb, student in records:
            vec = emb.get_vector()
            if not vec or len(vec) != self.vector_engine.dimension:
                logger.warning(
                    "Skipping malformed embedding (ID: %s, Student: %s, len: %d)",
                    emb.id,
                    student.student_id,
                    len(vec) if vec else 0,
                )
                continue

            vectors.append(vec)
            metadata_list.append(
                VectorMetadata(
                    student_id=student.student_id,
                    student_pk=student.id,
                    embedding_id=emb.id,
                )
            )

        if not vectors:
            self.vector_engine.reset()
            return 0

        vectors_arr = np.asarray(vectors, dtype=np.float32)
        count = self.vector_engine.build_index(vectors_arr, metadata_list)
        logger.info("FAISS index rebuilt successfully with %d reference vectors.", count)
        return count

    def match_face_embedding(
        self,
        query_vector: Union[np.ndarray, List[float]],
        threshold: Optional[float] = None,
        top_k: int = 5,
    ) -> IdentityMatchResult:
        """Match a 512-dimensional ArcFace query embedding against enrolled identities.
        
        Args:
            query_vector: Normalized 512-d embedding vector.
            threshold: Configurable cosine similarity threshold. Defaults to settings.
            top_k: Number of candidate matches to retrieve.
            
        Returns:
            IdentityMatchResult with match status, student ID, and top-K candidates.
        """
        return self.vector_engine.search(
            query_vector=query_vector,
            top_k=top_k,
            threshold=threshold,
        )

    def match_live_image(
        self,
        bgr_image: np.ndarray,
        threshold: Optional[float] = None,
        top_k: int = 5,
    ) -> IdentityMatchResult:
        """Process a live BGR image, extract its ArcFace embedding, and match identity.
        
        Args:
            bgr_image: 3-channel BGR image array.
            threshold: Configurable cosine similarity threshold.
            top_k: Number of nearest candidates to return.
            
        Returns:
            IdentityMatchResult.
        """
        # Run SCRFD detection
        result = self.face_engine.process_image(
            bgr_image,
            confidence_threshold=settings.DETECTION_CONFIDENCE_THRESHOLD,
        )

        thresh = threshold if threshold is not None else settings.FACE_SIMILARITY_THRESHOLD

        if result.total_detected == 0:
            return IdentityMatchResult(
                status=MatchStatus.UNKNOWN,
                matched=False,
                student_id=None,
                similarity_score=0.0,
                threshold_used=thresh,
                candidates=[],
                message="No face detected in live query image.",
            )

        # Use primary detected face (highest confidence)
        primary_face = result.faces[0]
        return self.match_face_embedding(
            query_vector=primary_face.embedding,
            threshold=thresh,
            top_k=top_k,
        )


def get_identity_matching_service() -> IdentityMatchingService:
    """Convenience factory returning the IdentityMatchingService."""
    return IdentityMatchingService()
