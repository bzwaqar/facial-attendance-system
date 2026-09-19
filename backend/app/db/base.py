"""SQLAlchemy 2.x Declarative Base and metadata aggregation."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


# Import all models here so that Base.metadata includes them
from app.db.models.student import Student  # noqa: E402
from app.db.models.face_embedding import FaceEmbedding  # noqa: E402
from app.db.models.attendance import Attendance  # noqa: E402
from app.db.models.attendance_session import AttendanceSession  # noqa: E402

__all__ = ["Base", "Student", "FaceEmbedding", "Attendance", "AttendanceSession"]

