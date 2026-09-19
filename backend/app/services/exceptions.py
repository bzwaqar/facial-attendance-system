"""Domain and business logic exceptions for student and enrollment services."""

from typing import List, Optional
from app.schemas.enrollment import ImageDiagnostic


class ServiceError(Exception):
    """Base exception for service layer operations."""

    def __init__(self, message: str = "A service error occurred."):
        self.message = message
        super().__init__(self.message)


class DuplicateStudentError(ServiceError):
    """Raised when attempting to register a student with an existing student_id."""

    def __init__(self, student_id: str):
        self.student_id = student_id
        super().__init__(f"Student with ID '{student_id}' already exists.")


class StudentNotFoundError(ServiceError):
    """Raised when a requested student is not found in the database."""

    def __init__(self, identifier: str | int):
        self.identifier = identifier
        super().__init__(f"Student '{identifier}' was not found.")


class InvalidImageCountError(ServiceError):
    """Raised when the submitted enrollment image count is outside the [5, 10] range."""

    def __init__(self, received_count: int, min_count: int = 5, max_count: int = 10):
        self.received_count = received_count
        self.min_count = min_count
        self.max_count = max_count
        super().__init__(
            f"Enrollment requires between {min_count} and {max_count} face images, "
            f"but {received_count} images were provided."
        )


class FaceEnrollmentValidationError(ServiceError):
    """Raised when one or more submitted images fail face detection or quality checks."""

    def __init__(
        self,
        diagnostics: List[ImageDiagnostic],
        total_submitted: int,
        accepted_count: int,
        rejected_count: int,
        message: Optional[str] = None,
    ):
        self.diagnostics = diagnostics
        self.total_submitted = total_submitted
        self.accepted_count = accepted_count
        self.rejected_count = rejected_count
        msg = message or (
            f"Face enrollment rejected: {rejected_count} of {total_submitted} submitted images "
            f"failed biometric quality criteria (minimum 5 valid images required)."
        )
        super().__init__(msg)


class SessionNotFoundError(ServiceError):
    """Raised when a requested attendance session is not found."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Attendance session '{session_id}' was not found.")


class DuplicateSessionError(ServiceError):
    """Raised when attempting to create an attendance session with an existing session_id."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Attendance session '{session_id}' already exists.")


class AttendanceRuleViolationError(ServiceError):
    """Raised when an attendance request violates attendance business rules."""

    def __init__(self, message: str):
        super().__init__(message)

