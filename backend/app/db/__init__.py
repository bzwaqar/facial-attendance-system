"""Database package initializing SQLAlchemy models, session, and initialization tools."""

from app.db.base import Base
from app.db.models.student import Student
from app.db.models.face_embedding import FaceEmbedding
from app.db.models.attendance import Attendance
from app.db.session import (
    async_engine,
    sync_engine,
    AsyncSessionLocal,
    SyncSessionLocal,
    get_db,
    get_async_session,
    get_sync_db,
    check_database_connection,
)
from app.db.exceptions import (
    DatabaseError,
    DatabaseConnectionError,
    DuplicateRecordError,
    DuplicateAttendanceError,
    RecordNotFoundError,
)

__all__ = [
    "Base",
    "Student",
    "FaceEmbedding",
    "Attendance",
    "async_engine",
    "sync_engine",
    "AsyncSessionLocal",
    "SyncSessionLocal",
    "get_db",
    "get_async_session",
    "get_sync_db",
    "check_database_connection",
    "DatabaseError",
    "DatabaseConnectionError",
    "DuplicateRecordError",
    "DuplicateAttendanceError",
    "RecordNotFoundError",
]
