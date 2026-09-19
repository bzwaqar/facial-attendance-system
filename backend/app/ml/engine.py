"""Face processing engine orchestrating SCRFD detection, alignment, and ArcFace embeddings."""

import io
import logging
import time
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from app.core.config import settings
from app.ml.exceptions import InvalidImageError
from app.ml.loader import InsightFaceModelLoader, get_model_loader
from app.ml.models import DetectedFace, FaceProcessingResult

logger = logging.getLogger(__name__)


class FaceEngine:
    """High-level face recognition engine handling detection, alignment, and embedding."""

    def __init__(self, loader: Optional[InsightFaceModelLoader] = None):
        """Initialize the face engine with an InsightFace model loader.
        
        Args:
            loader: Model loader instance. If None, the global singleton loader is used.
        """
        self.loader = loader or get_model_loader()

    def _load_and_validate_image(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
    ) -> np.ndarray:
        """Decode and validate an image into a standard BGR NumPy array.
        
        Args:
            image_input: Filepath, raw image bytes, or existing NumPy image array.
            
        Returns:
            Validated 3-channel BGR NumPy array (uint8).
            
        Raises:
            InvalidImageError: If the image cannot be read, decoded, or has invalid dimensions.
        """
        if image_input is None:
            raise InvalidImageError("Image input cannot be None.")

        # 1. Existing NumPy array
        if isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise InvalidImageError("Image array is empty (size 0).")
            if image_input.ndim == 2:
                # Grayscale to BGR
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.ndim == 3:
                if image_input.shape[2] == 4:
                    # BGRA to BGR
                    return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
                elif image_input.shape[2] == 3:
                    return image_input.copy()
            raise InvalidImageError(f"Unsupported image shape: {image_input.shape}")

        # 2. Raw bytes input (e.g. from file upload or network buffer)
        if isinstance(image_input, (bytes, bytearray)):
            if len(image_input) == 0:
                raise InvalidImageError("Image byte payload is empty.")
            arr = np.frombuffer(image_input, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise InvalidImageError("Failed to decode image from byte buffer.")
            return img

        # 3. Path or string filepath
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise InvalidImageError(f"Image file does not exist: '{path}'")
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise InvalidImageError(f"Failed to read image file at '{path}'.")
            return img

        raise InvalidImageError(f"Unsupported image input type: {type(image_input).__name__}")

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """Compute L2 normalized embedding vector.
        
        Args:
            embedding: Raw ArcFace embedding array of shape (512,).
            
        Returns:
            L2-normalized unit-length embedding vector.
        """
        emb = embedding.astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-10:
            return emb / norm
        return emb

    def process_image(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        confidence_threshold: Optional[float] = None,
    ) -> FaceProcessingResult:
        """Execute the full face processing pipeline:
        Image -> SCRFD Detection -> Alignment -> ArcFace Embedding -> Normalization.
        
        Args:
            image_input: Input image as file path, raw bytes, or BGR ndarray.
            confidence_threshold: Minimum detection confidence score (default from settings).
            
        Returns:
            FaceProcessingResult containing detected faces and normalized embeddings.
            
        Raises:
            InvalidImageError: If the input cannot be processed as a valid image.
        """
        start_time = time.perf_counter()

        # Step 1: Validate and decode image
        bgr_image = self._load_and_validate_image(image_input)

        # Step 2: Acquire model instance (loaded once and reused)
        model = self.loader.get_model()

        # Step 3: Run SCRFD detection, alignment, and ArcFace feature extraction
        raw_faces = model.get(bgr_image)

        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.DETECTION_CONFIDENCE_THRESHOLD
        )

        detected_faces: List[DetectedFace] = []

        # Step 4: Filter by confidence and normalize embeddings
        for face in raw_faces:
            det_score = float(getattr(face, "det_score", 0.0))
            if det_score < threshold:
                continue

            # Bounding box coordinates [x1, y1, x2, y2]
            bbox = [float(coord) for coord in face.bbox.tolist()] if hasattr(face, "bbox") else []

            # Facial landmarks (5 keypoints)
            landmarks = (
                face.kps.tolist()
                if hasattr(face, "kps") and face.kps is not None
                else None
            )

            # ArcFace embedding vector
            raw_embedding = getattr(face, "embedding", None)
            if raw_embedding is None:
                continue

            # Step 5: L2 normalize embedding vector
            norm_embedding = self.normalize_embedding(raw_embedding)

            detected_faces.append(
                DetectedFace(
                    bbox=bbox,
                    confidence=det_score,
                    embedding=norm_embedding,
                    landmarks=landmarks,
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return FaceProcessingResult(
            total_detected=len(detected_faces),
            faces=detected_faces,
            processing_time_ms=elapsed_ms,
        )

    def detect_faces_only(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        confidence_threshold: Optional[float] = None,
    ) -> List[DetectedFace]:
        """Execute SCRFD face detection only, without running ArcFace feature extraction.
        
        Args:
            image_input: Input image as file path, raw bytes, or BGR ndarray.
            confidence_threshold: Minimum detection confidence score.
            
        Returns:
            List of DetectedFace instances with bounding box and landmarks (embedding is None).
        """
        bgr_image = self._load_and_validate_image(image_input)
        model = self.loader.get_model()

        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.DETECTION_CONFIDENCE_THRESHOLD
        )

        bboxes, kpss = model.det_model.detect(bgr_image)
        if bboxes.shape[0] == 0:
            return []

        detected_faces: List[DetectedFace] = []
        for i in range(bboxes.shape[0]):
            det_score = float(bboxes[i, 4])
            if det_score < threshold:
                continue

            bbox = [float(coord) for coord in bboxes[i, :4].tolist()]
            landmarks = kpss[i].tolist() if kpss is not None else None

            detected_faces.append(
                DetectedFace(
                    bbox=bbox,
                    confidence=det_score,
                    embedding=None,
                    landmarks=landmarks,
                )
            )
        return detected_faces

    def extract_face_embedding(
        self,
        bgr_image: np.ndarray,
        detected_face: DetectedFace,
    ) -> np.ndarray:
        """Extract and normalize 512-dimensional ArcFace embedding for an already detected face.
        
        Args:
            bgr_image: BGR image array.
            detected_face: DetectedFace with valid bbox and landmarks.
            
        Returns:
            L2-normalized 512-dimensional float32 NumPy array.
        """
        from insightface.app.common import Face
        from app.ml.exceptions import FaceProcessingError

        model = self.loader.get_model()
        rec_model = model.models.get("recognition")
        if rec_model is None:
            raise FaceProcessingError("InsightFace recognition module is not loaded.")

        kps = np.array(detected_face.landmarks) if detected_face.landmarks is not None else None
        face_obj = Face(
            bbox=np.array(detected_face.bbox, dtype=np.float32),
            kps=kps,
            det_score=detected_face.confidence,
        )

        rec_model.get(bgr_image, face_obj)
        raw_emb = getattr(face_obj, "embedding", None)
        if raw_emb is None:
            raise FaceProcessingError("Failed to extract ArcFace embedding from face.")

        norm_emb = self.normalize_embedding(raw_emb)
        detected_face.embedding = norm_emb
        return norm_emb


def get_face_engine() -> FaceEngine:
    """Convenience factory returning a FaceEngine instance."""
    return FaceEngine()
