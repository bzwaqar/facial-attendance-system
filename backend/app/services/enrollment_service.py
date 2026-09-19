"""Facial enrollment service orchestrating image validation, ArcFace embeddings, and persistence."""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.face_embedding import FaceEmbedding
from app.ml.engine import FaceEngine, get_face_engine
from app.ml.exceptions import InvalidImageError
from app.ml.quality import assess_face_quality, validate_embedding_vector
from app.schemas.enrollment import EnrollmentResponse, ImageDiagnostic
from app.services.exceptions import (
    FaceEnrollmentValidationError,
    InvalidImageCountError,
    StudentNotFoundError,
)
from app.services.student_service import StudentService

logger = logging.getLogger(__name__)


class EnrollmentService:
    """Service managing facial biometric enrollment for registered students."""

    def __init__(self, engine: Optional[FaceEngine] = None):
        """Initialize enrollment service with FaceEngine instance."""
        self.engine = engine or get_face_engine()

    async def enroll_student_faces(
        self,
        session: AsyncSession,
        student_id: str,
        images: List[Tuple[str, bytes]],
        min_images: int = 5,
        max_images: int = 10,
        replace_existing: bool = False,
    ) -> EnrollmentResponse:
        """Enroll 5–10 face images for a registered student.
        
        Pipeline:
        1. Validates image count (5 <= count <= 10).
        2. Verifies student exists in database.
        3. Decodes each image and runs SCRFD face detection.
        4. Validates exactly one face per image (rejects 0 or >1 face).
        5. Quality filtering (confidence, sharpness, illumination, resolution).
        6. Generates 512-d normalized ArcFace embedding and validates vector integrity.
        7. If any image fails or accepted count < 5, aborts and returns diagnostics.
        8. Persists all valid embeddings with model name/version in PostgreSQL.
        
        Args:
            session: Active SQLAlchemy async database session.
            student_id: University student ID.
            images: List of tuples (filename, raw_image_bytes).
            min_images: Minimum acceptable images (default: 5).
            max_images: Maximum acceptable images (default: 10).
            replace_existing: If True, clears existing embeddings before storing new ones.
            
        Returns:
            EnrollmentResponse with detailed diagnostics for all images.
            
        Raises:
            StudentNotFoundError: If student is not found.
            InvalidImageCountError: If image count is < 5 or > 10.
            FaceEnrollmentValidationError: If any image fails face detection or quality checks.
        """
        # Step 1: Validate image count
        total_submitted = len(images)
        if total_submitted < min_images or total_submitted > max_images:
            logger.warning(
                "Enrollment rejected for student '%s': image count %d outside [%d, %d]",
                student_id,
                total_submitted,
                min_images,
                max_images,
            )
            raise InvalidImageCountError(
                received_count=total_submitted,
                min_count=min_images,
                max_count=max_images,
            )

        # Step 2: Validate student existence
        student = await StudentService.get_student_by_student_id(session, student_id)
        if not student:
            logger.warning("Enrollment rejected: Student '%s' not found.", student_id)
            raise StudentNotFoundError(identifier=student_id)

        diagnostics: List[ImageDiagnostic] = []
        accepted_embeddings: List[FaceEmbedding] = []

        # Step 3 & 4 & 5: Process each image through detection, quality filtering & ArcFace
        for idx, (filename, img_bytes) in enumerate(images, start=1):
            file_label = filename or f"image_{idx}.jpg"

            # 3a. Decode image
            try:
                bgr_img = self.engine._load_and_validate_image(img_bytes)
            except InvalidImageError as exc:
                diagnostics.append(
                    ImageDiagnostic(
                        filename=file_label,
                        status="rejected",
                        reason=f"Failed to decode image: {exc.message}",
                    )
                )
                continue

            # 3b. Run SCRFD Face Detection
            result = self.engine.process_image(
                bgr_img,
                confidence_threshold=settings.DETECTION_CONFIDENCE_THRESHOLD,
            )

            # 4. Face count validation: exactly one usable face required per image
            if result.total_detected == 0:
                diagnostics.append(
                    ImageDiagnostic(
                        filename=file_label,
                        status="rejected",
                        reason="No face detected in the image. Ensure the student's face is clearly visible.",
                    )
                )
                continue

            if result.total_detected > 1:
                diagnostics.append(
                    ImageDiagnostic(
                        filename=file_label,
                        status="rejected",
                        reason=(
                            f"Multiple faces detected ({result.total_detected} faces found). "
                            "Each enrollment image must contain exactly one face."
                        ),
                    )
                )
                continue

            detected_face = result.faces[0]

            # 5. Biometric Quality Assessment (resolution, sharpness, lighting, confidence)
            quality = assess_face_quality(
                bgr_image=bgr_img,
                detected_face=detected_face,
                min_confidence=settings.DETECTION_CONFIDENCE_THRESHOLD,
            )

            if not quality.is_valid:
                diagnostics.append(
                    ImageDiagnostic(
                        filename=file_label,
                        status="rejected",
                        reason=quality.reason,
                        confidence=quality.confidence,
                        quality_score=quality.quality_score,
                        bbox=detected_face.bbox,
                    )
                )
                continue

            # 6. Embedding vector validation
            try:
                validate_embedding_vector(detected_face.embedding, expected_dim=512)
            except ValueError as val_err:
                diagnostics.append(
                    ImageDiagnostic(
                        filename=file_label,
                        status="rejected",
                        reason=f"Embedding vector validation error: {val_err}",
                        confidence=quality.confidence,
                        bbox=detected_face.bbox,
                    )
                )
                continue

            # 7. Create FaceEmbedding ORM record
            embedding_record = FaceEmbedding(
                student_id=student.id,
                embedding=b"",
                model_name="ArcFace",
                model_version=settings.INSIGHTFACE_MODEL_NAME,
                quality_score=quality.quality_score,
            )
            embedding_record.set_vector(detected_face.embedding_list)
            accepted_embeddings.append(embedding_record)

            diagnostics.append(
                ImageDiagnostic(
                    filename=file_label,
                    status="accepted",
                    confidence=quality.confidence,
                    quality_score=quality.quality_score,
                    bbox=detected_face.bbox,
                )
            )

        accepted_count = len(accepted_embeddings)
        rejected_count = total_submitted - accepted_count

        # Step 8: Strict validation policy:
        # All submitted images must be valid, and accepted count must meet min_images (5)
        if rejected_count > 0 or accepted_count < min_images:
            logger.warning(
                "Enrollment validation failed for student '%s': %d accepted, %d rejected.",
                student_id,
                accepted_count,
                rejected_count,
            )
            raise FaceEnrollmentValidationError(
                diagnostics=diagnostics,
                total_submitted=total_submitted,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
            )

        # Step 9: Database Persistence
        if replace_existing and student.face_embeddings:
            for old_emb in list(student.face_embeddings):
                await session.delete(old_emb)
            student.face_embeddings.clear()

        for new_emb in accepted_embeddings:
            student.face_embeddings.append(new_emb)

        await session.flush()
        logger.info(
            "Successfully enrolled %d face embeddings for student '%s' (model: ArcFace:%s).",
            accepted_count,
            student_id,
            settings.INSIGHTFACE_MODEL_NAME,
        )

        return EnrollmentResponse(
            student_id=student_id,
            total_submitted=total_submitted,
            total_enrolled=accepted_count,
            model_name="ArcFace",
            model_version=settings.INSIGHTFACE_MODEL_NAME,
            images_summary=diagnostics,
            message=(
                f"Successfully enrolled {accepted_count} reference face embeddings "
                f"for student '{student.name}' ({student_id})."
            ),
        )


def get_enrollment_service() -> EnrollmentService:
    """Convenience factory returning EnrollmentService."""
    return EnrollmentService()
