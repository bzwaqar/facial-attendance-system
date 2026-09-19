"""Pydantic schemas module exporting request and response entities."""

from app.schemas.enrollment import (
    EnrollmentErrorDetail,
    EnrollmentResponse,
    ImageDiagnostic,
)
from app.schemas.health import HealthResponse
from app.schemas.matching import (
    IdentityMatchResult,
    MatchCandidate,
    MatchStatus,
)
from app.schemas.student import (
    StudentBase,
    StudentCreate,
    StudentResponse,
    StudentUpdate,
)

from app.schemas.attendance import (
    AttendanceMarkRequest,
    AttendanceMarkResponse,
    AttendanceSessionCreate,
    AttendanceSessionResponse,
    AttendanceStatus,
    SessionAttendanceRecord,
    SessionAttendanceSummary,
)

__all__ = [
    "HealthResponse",
    "StudentBase",
    "StudentCreate",
    "StudentUpdate",
    "StudentResponse",
    "ImageDiagnostic",
    "EnrollmentResponse",
    "EnrollmentErrorDetail",
    "MatchStatus",
    "MatchCandidate",
    "IdentityMatchResult",
    "AttendanceStatus",
    "AttendanceSessionCreate",
    "AttendanceSessionResponse",
    "AttendanceMarkRequest",
    "AttendanceMarkResponse",
    "SessionAttendanceRecord",
    "SessionAttendanceSummary",
]

