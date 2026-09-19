"""FastAPI router endpoints for on-demand face recognition and biometric verification."""

import base64
import logging
import time
from typing import List, Optional

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.student import Student
from app.db.session import get_db
from app.liveness import BaseLivenessDetector, get_liveness_detector
from app.ml.engine import FaceEngine, get_face_engine
from app.ml.exceptions import InvalidImageError
from app.schemas.recognition import (
    RecognitionPredictBase64Request,
    RecognitionPredictResponse,
    RecognizedFaceMatch,
)
from app.services.matching_service import (
    IdentityMatchingService,
    get_identity_matching_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Face Recognition"])


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to BGR numpy array."""
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image payload is empty.",
        )
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not decode image from byte buffer.",
        )
    return img


@router.post(
    "/predict",
    response_model=RecognitionPredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Recognize Face from Uploaded Image",
    description=(
        "Detect all faces in an uploaded image, extract ArcFace embeddings, "
        "query FAISS vector index, and evaluate anti-spoofing liveness."
    ),
)
async def predict_recognition_file(
    file: UploadFile = File(..., description="Image file (JPEG/PNG) containing faces to recognize."),
    threshold: Optional[float] = Form(None, description="Optional custom similarity threshold override."),
    check_liveness: bool = Form(True, description="Whether to evaluate liveness on detected faces."),
    db: AsyncSession = Depends(get_db),
    face_engine: FaceEngine = Depends(get_face_engine),
    matching_service: IdentityMatchingService = Depends(get_identity_matching_service),
) -> RecognitionPredictResponse:
    """Process an uploaded image file and return facial recognition matches."""
    start_time = time.perf_counter()
    image_bytes = await file.read()
    bgr_frame = _decode_image_bytes(image_bytes)

    return await _process_recognition_frame(
        bgr_frame=bgr_frame,
        threshold=threshold,
        check_liveness=check_liveness,
        db=db,
        face_engine=face_engine,
        matching_service=matching_service,
        start_time=start_time,
    )


@router.post(
    "/predict-base64",
    response_model=RecognitionPredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Recognize Face from Base64 Image",
    description="Recognize faces from a base64 encoded image string (e.g. from web canvas / webcam snapshot).",
)
async def predict_recognition_base64(
    request: RecognitionPredictBase64Request,
    db: AsyncSession = Depends(get_db),
    face_engine: FaceEngine = Depends(get_face_engine),
    matching_service: IdentityMatchingService = Depends(get_identity_matching_service),
) -> RecognitionPredictResponse:
    """Process a base64 encoded image string and return recognition matches."""
    start_time = time.perf_counter()
    raw_b64 = request.image_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_b64)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 image data: {exc}",
        ) from exc

    bgr_frame = _decode_image_bytes(image_bytes)

    return await _process_recognition_frame(
        bgr_frame=bgr_frame,
        threshold=request.threshold,
        check_liveness=request.check_liveness,
        db=db,
        face_engine=face_engine,
        matching_service=matching_service,
        start_time=start_time,
    )


async def _process_recognition_frame(
    bgr_frame: np.ndarray,
    threshold: Optional[float],
    check_liveness: bool,
    db: AsyncSession,
    face_engine: FaceEngine,
    matching_service: IdentityMatchingService,
    start_time: float,
) -> RecognitionPredictResponse:
    """Core recognition evaluation logic shared across file and base64 endpoints."""
    eval_threshold = (
        threshold
        if threshold is not None
        else settings.FACE_SIMILARITY_THRESHOLD
    )

    # 1. Run SCRFD Detection and ArcFace feature extraction
    try:
        proc_result = face_engine.process_image(bgr_frame)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # 2. Pre-fetch active student names for ID lookup
    name_lookup = {}
    try:
        query = select(Student.student_id, Student.name).where(Student.is_active.is_(True))
        db_res = await db.execute(query)
        name_lookup = {row[0]: row[1] for row in db_res.all()}
    except Exception as exc:
        logger.debug("Database student lookup omitted: %s", exc)

    liveness_detector = get_liveness_detector() if check_liveness else None
    matches: List[RecognizedFaceMatch] = []

    # 3. Match each detected face against the FAISS index
    for idx, face in enumerate(proc_result.faces):
        match_res = matching_service.match_face_embedding(
            query_vector=face.embedding,
            threshold=eval_threshold,
            top_k=1,
        )

        student_name = name_lookup.get(match_res.student_id) if match_res.student_id else None

        # 4. Anti-spoofing liveness evaluation
        is_live = None
        liveness_score = None
        liveness_status = None
        liveness_reason = None

        if liveness_detector is not None:
            try:
                live_res = liveness_detector.is_live(
                    frame=bgr_frame,
                    face=face,
                    bbox=face.bbox,
                    landmarks=face.landmarks,
                )
                is_live = live_res.is_live
                liveness_score = live_res.score
                liveness_status = live_res.status.value
                liveness_reason = live_res.reason
            except Exception as live_err:
                logger.warning("Error evaluating liveness on face #%d: %s", idx, live_err)

        matches.append(
            RecognizedFaceMatch(
                face_index=idx,
                matched=match_res.matched,
                student_id=match_res.student_id,
                student_name=student_name,
                similarity_score=round(float(match_res.similarity_score), 4),
                detection_confidence=round(float(face.confidence), 4),
                bbox=[round(float(c), 2) for c in face.bbox],
                landmarks=[[round(float(p[0]), 2), round(float(p[1]), 2)] for p in face.landmarks] if face.landmarks else None,
                is_live=is_live,
                liveness_score=round(float(liveness_score), 4) if liveness_score is not None else None,
                liveness_status=liveness_status,
                liveness_reason=liveness_reason,
            )
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return RecognitionPredictResponse(
        total_faces_detected=len(proc_result.faces),
        faces=matches,
        inference_time_ms=round(elapsed_ms, 2),
    )
