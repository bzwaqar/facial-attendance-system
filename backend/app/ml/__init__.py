"""Machine Learning & Biometrics package: SCRFD Face Detection, ArcFace Embeddings, and FAISS Vector Search."""

from app.ml.engine import FaceEngine, get_face_engine
from app.ml.exceptions import FaceProcessingError, InvalidImageError, ModelNotLoadedError
from app.ml.loader import InsightFaceModelLoader, get_model_loader
from app.ml.models import DetectedFace, FaceProcessingResult
from app.ml.quality import (
    FaceQualityAssessment,
    assess_face_quality,
    validate_embedding_vector,
)
from app.ml.vector_search import (
    VectorMetadata,
    VectorSearchEngine,
    get_vector_search_engine,
)

__all__ = [
    "FaceEngine",
    "get_face_engine",
    "InsightFaceModelLoader",
    "get_model_loader",
    "DetectedFace",
    "FaceProcessingResult",
    "FaceProcessingError",
    "InvalidImageError",
    "ModelNotLoadedError",
    "FaceQualityAssessment",
    "assess_face_quality",
    "validate_embedding_vector",
    "VectorMetadata",
    "VectorSearchEngine",
    "get_vector_search_engine",
]
