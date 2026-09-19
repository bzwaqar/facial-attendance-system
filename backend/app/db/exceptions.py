"""Custom database layer exceptions."""


class DatabaseError(Exception):
    """Base exception for database-related failures."""

    def __init__(self, message: str = "A database error occurred."):
        self.message = message
        super().__init__(self.message)


class DatabaseConnectionError(DatabaseError):
    """Raised when connecting to the database fails."""

    def __init__(self, message: str = "Failed to establish a connection to the database."):
        super().__init__(message)


class DuplicateRecordError(DatabaseError):
    """Raised when a unique constraint is violated."""

    def __init__(self, message: str = "A record with these unique attributes already exists."):
        super().__init__(message)


class DuplicateAttendanceError(DuplicateRecordError):
    """Raised when attendance for a student in a session has already been recorded."""

    def __init__(
        self,
        student_id: int | str,
        session_id: str,
        message: str | None = None,
    ):
        self.student_id = student_id
        self.session_id = session_id
        msg = (
            message
            or f"Attendance for student '{student_id}' in session '{session_id}' has already been recorded."
        )
        super().__init__(msg)


class RecordNotFoundError(DatabaseError):
    """Raised when a queried entity does not exist."""

    def __init__(self, entity_name: str, identifier: int | str):
        self.entity_name = entity_name
        self.identifier = identifier
        super().__init__(f"{entity_name} with identifier '{identifier}' was not found.")
