"""Student service managing registration, queries, and duplicate prevention."""

import logging
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.student import Student
from app.schemas.student import StudentCreate, StudentUpdate
from app.services.exceptions import DuplicateStudentError, StudentNotFoundError

logger = logging.getLogger(__name__)


class StudentService:
    """Service layer handling student database operations."""

    @staticmethod
    async def create_student(session: AsyncSession, student_in: StudentCreate) -> Student:
        """Register a new student in the database.
        
        Args:
            session: Active SQLAlchemy async session.
            student_in: Validated student creation payload.
            
        Returns:
            Created Student ORM model.
            
        Raises:
            DuplicateStudentError: If student_id is already registered.
        """
        # Pre-check to provide immediate clear error before database constraint fires
        existing = await StudentService.get_student_by_student_id(
            session=session,
            student_id=student_in.student_id,
        )
        if existing:
            raise DuplicateStudentError(student_id=student_in.student_id)

        student = Student(
            student_id=student_in.student_id,
            name=student_in.name,
            department=student_in.department or "Computer Science",
            semester=student_in.semester or "1st",
            section=student_in.section or "A",
            is_active=student_in.is_active,
        )
        session.add(student)

        try:
            await session.flush()
            await session.refresh(student, attribute_names=["face_embeddings"])
            logger.info("Successfully registered student: %s (ID: %s)", student.name, student.student_id)
            return student
        except IntegrityError as exc:
            await session.rollback()
            logger.warning("IntegrityError on student creation for '%s': %s", student_in.student_id, exc)
            raise DuplicateStudentError(student_id=student_in.student_id) from exc

    @staticmethod
    async def get_student_by_student_id(
        session: AsyncSession,
        student_id: str,
        load_embeddings: bool = True,
    ) -> Optional[Student]:
        """Fetch student by university student_id.
        
        Args:
            session: Active SQLAlchemy async session.
            student_id: University student identifier (e.g. 'CS-2026-001').
            load_embeddings: If True, eagerly loads related face_embeddings.
            
        Returns:
            Student instance or None if not found.
        """
        query = select(Student).where(Student.student_id == student_id)
        if load_embeddings:
            query = query.options(selectinload(Student.face_embeddings))

        result = await session.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_student_by_id(
        session: AsyncSession,
        student_pk: int,
        load_embeddings: bool = True,
    ) -> Optional[Student]:
        """Fetch student by internal primary key."""
        query = select(Student).where(Student.id == student_pk)
        if load_embeddings:
            query = query.options(selectinload(Student.face_embeddings))

        result = await session.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def list_students(
        session: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        department: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> List[Student]:
        """List students with pagination and optional filtering."""
        query = select(Student).options(selectinload(Student.face_embeddings))

        if department:
            query = query.where(Student.department == department)
        if is_active is not None:
            query = query.where(Student.is_active == is_active)

        query = query.order_by(Student.id.asc()).offset(skip).limit(limit)
        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def delete_student(session: AsyncSession, student_id: str) -> bool:
        """Delete a student and cascade delete associated face embeddings."""
        student = await StudentService.get_student_by_student_id(session, student_id)
        if not student:
            raise StudentNotFoundError(identifier=student_id)

        await session.delete(student)
        await session.flush()
        logger.info("Deleted student '%s' and cascade deleted embeddings.", student_id)
        return True


def get_student_service() -> StudentService:
    """Convenience factory returning StudentService."""
    return StudentService()
