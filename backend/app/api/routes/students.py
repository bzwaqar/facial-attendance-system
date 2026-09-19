"""FastAPI router endpoints for student management and facial biometric enrollment."""

import logging
from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.student import Student
from app.db.session import get_db
from app.schemas.enrollment import EnrollmentErrorDetail, EnrollmentResponse
from app.schemas.student import StudentCreate, StudentResponse
from app.services.enrollment_service import EnrollmentService, get_enrollment_service
from app.services.exceptions import (
    DuplicateStudentError,
    FaceEnrollmentValidationError,
    InvalidImageCountError,
    StudentNotFoundError,
)
from app.services.student_service import StudentService, get_student_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Students & Enrollment"])


def _to_student_response(student: Student) -> StudentResponse:
    """Map Student ORM model to StudentResponse schema."""
    emb_count = len(student.face_embeddings) if student.face_embeddings is not None else 0
    return StudentResponse(
        id=student.id,
        student_id=student.student_id,
        name=student.name,
        department=student.department,
        semester=student.semester,
        section=student.section,
        is_active=student.is_active,
        created_at=student.created_at,
        face_embeddings_count=emb_count,
        is_enrolled=emb_count >= 5,
    )


@router.post(
    "",
    response_model=StudentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register Student",
    description="Create a new student record. Enforces unique student_id constraint.",
    responses={
        201: {"description": "Student created successfully."},
        409: {"description": "Student with this student_id already exists."},
    },
)
async def create_student(
    student_in: StudentCreate,
    session: AsyncSession = Depends(get_db),
    student_service: StudentService = Depends(get_student_service),
) -> StudentResponse:
    """Register a new student entity in the system."""
    try:
        created = await student_service.create_student(session, student_in)
        return _to_student_response(created)
    except DuplicateStudentError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=exc.message,
        ) from exc


@router.get(
    "/{student_id}",
    response_model=StudentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Student Details",
    description="Retrieve student details along with biometric enrollment status and embeddings count.",
    responses={
        200: {"description": "Student details retrieved."},
        404: {"description": "Student not found."},
    },
)
async def get_student(
    student_id: str,
    session: AsyncSession = Depends(get_db),
    student_service: StudentService = Depends(get_student_service),
) -> StudentResponse:
    """Retrieve an existing student by student_id."""
    student = await student_service.get_student_by_student_id(session, student_id)
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with ID '{student_id}' was not found.",
        )
    return _to_student_response(student)


@router.get(
    "",
    response_model=List[StudentResponse],
    status_code=status.HTTP_200_OK,
    summary="List Students",
    description="Retrieve a paginated list of registered students with enrollment metrics.",
)
async def list_students(
    skip: int = Query(default=0, ge=0, description="Offset for pagination"),
    limit: int = Query(default=50, ge=1, le=100, description="Page size limit"),
    department: Optional[str] = Query(default=None, description="Filter by academic department"),
    is_active: Optional[bool] = Query(default=None, description="Filter by active status"),
    session: AsyncSession = Depends(get_db),
    student_service: StudentService = Depends(get_student_service),
) -> List[StudentResponse]:
    """Retrieve list of registered students."""
    students = await student_service.list_students(
        session=session,
        skip=skip,
        limit=limit,
        department=department,
        is_active=is_active,
    )
    return [_to_student_response(s) for s in students]


@router.post(
    "/{student_id}/enroll",
    response_model=EnrollmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll Facial Biometrics",
    description=(
        "Upload 5 to 10 facial images for biometric enrollment. Each image must contain "
        "exactly one usable face meeting sharpness, lighting, and detection quality standards. "
        "Extracts 512-d normalized ArcFace embeddings and stores them in PostgreSQL."
    ),
    responses={
        201: {"description": "Biometric face embeddings successfully created and persisted."},
        400: {"description": "Invalid image count (< 5 or > 10 images)."},
        404: {"description": "Student not found."},
        422: {
            "model": EnrollmentErrorDetail,
            "description": "Validation failure (no face, multiple faces, blurry, or low quality).",
        },
    },
)
async def enroll_student_faces(
    student_id: str,
    files: List[UploadFile] = File(
        ...,
        description="5 to 10 facial portrait images for biometric enrollment.",
    ),
    replace_existing: bool = Query(
        default=False,
        description="If True, replaces any previously enrolled embeddings for this student.",
    ),
    session: AsyncSession = Depends(get_db),
    enrollment_service: EnrollmentService = Depends(get_enrollment_service),
) -> EnrollmentResponse:
    """Process and enroll 5-10 facial images for the specified student."""
    # Read raw bytes from uploaded files
    image_payloads = []
    for file in files:
        contents = await file.read()
        image_payloads.append((file.filename or "unnamed.jpg", contents))

    try:
        response = await enrollment_service.enroll_student_faces(
            session=session,
            student_id=student_id,
            images=image_payloads,
            min_images=5,
            max_images=10,
            replace_existing=replace_existing,
        )
        return response

    except StudentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.message,
        ) from exc

    except InvalidImageCountError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        ) from exc

    except FaceEnrollmentValidationError as exc:
        # Return structured HTTP 422 with full itemized diagnostics
        error_detail = EnrollmentErrorDetail(
            message=exc.message,
            total_submitted=exc.total_submitted,
            accepted_count=exc.accepted_count,
            rejected_count=exc.rejected_count,
            diagnostics=exc.diagnostics,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_detail.model_dump(),
        ) from exc
