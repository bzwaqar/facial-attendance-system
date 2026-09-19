"""Dedicated Attendance Service managing attendance sessions, verification rules, and duplicate checks."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.attendance import Attendance
from app.db.models.attendance_session import AttendanceSession
from app.db.models.student import Student
from app.schemas.attendance import (
    AttendanceMarkRequest,
    AttendanceMarkResponse,
    AttendanceSessionCreate,
    AttendanceSessionResponse,
    AttendanceStatus,
    SessionAttendanceRecord,
    SessionAttendanceSummary,
)
from app.services.exceptions import (
    DuplicateSessionError,
    SessionNotFoundError,
)

logger = logging.getLogger(__name__)


class AttendanceService:
    """Core attendance engine enforcing temporal stability, business rules, and duplicate prevention."""

    def __init__(self, db: Optional[AsyncSession] = None):
        """Initialize attendance service with optional database session."""
        self.db = db

    def _resolve_session(self, session: Optional[AsyncSession] = None) -> AsyncSession:
        """Resolve the active database session from argument or instance state."""
        db = session or self.db
        if db is None:
            raise ValueError("An active database session is required for AttendanceService operations.")
        return db

    async def create_session(
        self,
        session_in: AttendanceSessionCreate,
        session: Optional[AsyncSession] = None,
    ) -> AttendanceSessionResponse:
        """Create a new attendance session / class.
        
        Args:
            session_in: Session creation payload.
            session: Optional active AsyncSession.
            
        Returns:
            AttendanceSessionResponse representation of created session.
            
        Raises:
            DuplicateSessionError: If a session with session_id already exists.
        """
        db = self._resolve_session(session)

        # Check for existing session with this session_id
        existing_stmt = select(AttendanceSession).where(AttendanceSession.session_id == session_in.session_id)
        result = await db.execute(existing_stmt)
        if result.scalar_one_or_none() is not None:
            raise DuplicateSessionError(session_id=session_in.session_id)

        db_session = AttendanceSession(
            session_id=session_in.session_id,
            course_name=session_in.course_name,
            instructor_name=session_in.instructor_name,
            is_active=True,
            min_similarity_threshold=session_in.min_similarity_threshold,
            min_confirmations=session_in.min_confirmations,
            created_at=datetime.now(timezone.utc),
        )
        db.add(db_session)
        await db.flush()

        logger.info("Created attendance session '%s' (course: %s)", db_session.session_id, db_session.course_name)
        return AttendanceSessionResponse(
            id=db_session.id,
            session_id=db_session.session_id,
            course_name=db_session.course_name,
            instructor_name=db_session.instructor_name,
            is_active=db_session.is_active,
            min_similarity_threshold=db_session.min_similarity_threshold,
            min_confirmations=db_session.min_confirmations,
            created_at=db_session.created_at,
            closed_at=db_session.closed_at,
            total_attendees=0,
        )

    async def get_session(
        self,
        session_id: str,
        session: Optional[AsyncSession] = None,
    ) -> AttendanceSession:
        """Retrieve AttendanceSession ORM entity by session_id.
        
        Args:
            session_id: Target session identifier string.
            session: Optional active AsyncSession.
            
        Returns:
            AttendanceSession ORM instance.
            
        Raises:
            SessionNotFoundError: If session does not exist.
        """
        db = self._resolve_session(session)
        stmt = select(AttendanceSession).where(AttendanceSession.session_id == session_id)
        result = await db.execute(stmt)
        session_obj = result.scalar_one_or_none()
        if session_obj is None:
            raise SessionNotFoundError(session_id=session_id)
        return session_obj

    async def list_sessions(
        self,
        is_active: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
        session: Optional[AsyncSession] = None,
    ) -> List[AttendanceSessionResponse]:
        """List attendance sessions with attendee counts.
        
        Args:
            is_active: Optional filter for active / closed sessions.
            limit: Pagination limit.
            offset: Pagination offset.
            session: Optional active AsyncSession.
        """
        db = self._resolve_session(session)

        # Count attendees per session
        count_subq = (
            select(
                Attendance.session_id,
                func.count(Attendance.id).label("attendee_count"),
            )
            .group_by(Attendance.session_id)
            .subquery()
        )

        query = (
            select(
                AttendanceSession,
                func.coalesce(count_subq.c.attendee_count, 0).label("attendee_count"),
            )
            .outerjoin(count_subq, AttendanceSession.session_id == count_subq.c.session_id)
            .order_by(AttendanceSession.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        if is_active is not None:
            query = query.where(AttendanceSession.is_active == is_active)

        result = await db.execute(query)
        sessions = []
        for sess_row, count in result.all():
            sessions.append(
                AttendanceSessionResponse(
                    id=sess_row.id,
                    session_id=sess_row.session_id,
                    course_name=sess_row.course_name,
                    instructor_name=sess_row.instructor_name,
                    is_active=sess_row.is_active,
                    min_similarity_threshold=sess_row.min_similarity_threshold,
                    min_confirmations=sess_row.min_confirmations,
                    created_at=sess_row.created_at,
                    closed_at=sess_row.closed_at,
                    total_attendees=int(count),
                )
            )
        return sessions

    async def close_session(
        self,
        session_id: str,
        session: Optional[AsyncSession] = None,
    ) -> AttendanceSessionResponse:
        """Mark an active session as closed.
        
        Args:
            session_id: Session identifier to close.
            session: Optional active AsyncSession.
        """
        db = self._resolve_session(session)
        sess = await self.get_session(session_id=session_id, session=db)
        sess.is_active = False
        sess.closed_at = datetime.now(timezone.utc)
        await db.flush()

        logger.info("Closed attendance session '%s'", session_id)
        return AttendanceSessionResponse.model_validate(sess)

    async def mark_attendance(
        self,
        request: AttendanceMarkRequest,
        session: Optional[AsyncSession] = None,
    ) -> AttendanceMarkResponse:
        """Evaluate recognition event against business rules and record attendance.
        
        Verification Pipeline:
        1. Validate target session exists in database.
        2. Verify session active status.
        3. Check UNKNOWN identity token.
        4. Evaluate configurable similarity and temporal observation criteria.
        5. Verify registered and active student entity.
        6. Enforce duplicate check (student marked at most once per session).
        7. Record attendance transactionally in PostgreSQL.
        
        Returns:
            AttendanceMarkResponse with status PRESENT, ALREADY_MARKED, UNKNOWN, or REJECTED.
        """
        db = self._resolve_session(session)
        event_time = request.timestamp or datetime.now(timezone.utc)

        # 1. Validate target session exists
        session_obj = await self.get_session(session_id=request.session_id, session=db)

        # 2. Verify session active status
        if settings.ATTENDANCE_REQUIRE_ACTIVE_SESSION and not session_obj.is_active:
            logger.warning("Attendance rejected: Session '%s' is inactive or closed.", request.session_id)
            return AttendanceMarkResponse(
                status=AttendanceStatus.REJECTED,
                student_id=request.student_id,
                session_id=request.session_id,
                similarity_score=request.similarity_score,
                timestamp=event_time,
                message=f"Attendance session '{request.session_id}' is inactive or closed.",
            )

        # 3. Check for UNKNOWN recognition outcome
        if not request.student_id or request.student_id.strip().upper() == "UNKNOWN":
            logger.info("Attendance unrecorded: Face recognition produced UNKNOWN identity.")
            return AttendanceMarkResponse(
                status=AttendanceStatus.UNKNOWN,
                student_id=None,
                session_id=request.session_id,
                similarity_score=request.similarity_score,
                timestamp=event_time,
                is_live=request.is_live,
                liveness_score=request.liveness_score,
                message="Face identity is UNKNOWN; attendance cannot be recorded.",
            )

        # 4. Anti-Spoofing & Liveness Verification
        if settings.LIVENESS_ENABLED:
            if request.is_live is False:
                logger.warning(
                    "Attendance REJECTED for student '%s' in session '%s': Liveness verification failed "
                    "(presentation attack suspected: score=%s, status=%s).",
                    request.student_id,
                    request.session_id,
                    request.liveness_score,
                    request.liveness_status,
                )
                return AttendanceMarkResponse(
                    status=AttendanceStatus.REJECTED,
                    student_id=request.student_id,
                    session_id=request.session_id,
                    similarity_score=request.similarity_score,
                    timestamp=event_time,
                    is_live=False,
                    liveness_score=request.liveness_score,
                    message="Attendance rejected: Liveness verification failed (anti-spoofing alert: presentation attack suspected).",
                )

            if settings.LIVENESS_STRICT_MODE and request.is_live is not True:
                logger.warning(
                    "Attendance REJECTED for student '%s' in session '%s': Strict liveness is required but was not verified.",
                    request.student_id,
                    request.session_id,
                )
                return AttendanceMarkResponse(
                    status=AttendanceStatus.REJECTED,
                    student_id=request.student_id,
                    session_id=request.session_id,
                    similarity_score=request.similarity_score,
                    timestamp=event_time,
                    is_live=False,
                    liveness_score=request.liveness_score,
                    message="Attendance rejected: Strict liveness verification required but not provided.",
                )

        # 5. Resolve and evaluate configurable rules (Session override -> Global default)
        effective_min_similarity = (
            session_obj.min_similarity_threshold
            if session_obj.min_similarity_threshold is not None
            else settings.ATTENDANCE_MIN_SIMILARITY
        )
        effective_min_confirmations = (
            session_obj.min_confirmations
            if session_obj.min_confirmations is not None
            else settings.ATTENDANCE_MIN_CONFIRMATIONS
        )

        if request.similarity_score < effective_min_similarity:
            logger.info(
                "Attendance rejected for student '%s': score %.3f below threshold %.3f",
                request.student_id,
                request.similarity_score,
                effective_min_similarity,
            )
            return AttendanceMarkResponse(
                status=AttendanceStatus.REJECTED,
                student_id=request.student_id,
                session_id=request.session_id,
                similarity_score=request.similarity_score,
                timestamp=event_time,
                message=(
                    f"Recognition similarity score ({request.similarity_score:.3f}) fell below "
                    f"required threshold ({effective_min_similarity:.3f})."
                ),
            )

        if not request.is_confirmed or request.observation_count < effective_min_confirmations:
            logger.info(
                "Attendance rejected for student '%s': observations %d below required %d",
                request.student_id,
                request.observation_count,
                effective_min_confirmations,
            )
            return AttendanceMarkResponse(
                status=AttendanceStatus.REJECTED,
                student_id=request.student_id,
                session_id=request.session_id,
                similarity_score=request.similarity_score,
                timestamp=event_time,
                message=(
                    f"Temporal verification requirement not met: received {request.observation_count} "
                    f"stable observations (minimum required: {effective_min_confirmations})."
                ),
            )

        # 5. Resolve Student from database
        student_query = select(Student).where(
            (Student.student_id == request.student_id)
            if request.student_pk is None
            else (Student.id == request.student_pk)
        )
        student_res = await db.execute(student_query)
        student = student_res.scalar_one_or_none()

        if student is None:
            logger.warning("Attendance rejected: Student '%s' not found in database.", request.student_id)
            return AttendanceMarkResponse(
                status=AttendanceStatus.REJECTED,
                student_id=request.student_id,
                session_id=request.session_id,
                similarity_score=request.similarity_score,
                timestamp=event_time,
                message=f"Student ID '{request.student_id}' is not registered in the system.",
            )

        if settings.ATTENDANCE_REQUIRE_ACTIVE_STUDENT and not student.is_active:
            logger.warning("Attendance rejected: Student '%s' account is inactive.", student.student_id)
            return AttendanceMarkResponse(
                status=AttendanceStatus.REJECTED,
                student_id=student.student_id,
                student_name=student.name,
                session_id=request.session_id,
                similarity_score=request.similarity_score,
                timestamp=event_time,
                message=f"Student '{student.name}' ({student.student_id}) account is deactivated.",
            )

        # 6. Duplicate check: Ensure student is marked at most once per session
        dup_query = select(Attendance).where(
            Attendance.student_id == student.id,
            Attendance.session_id == request.session_id,
        )
        dup_res = await db.execute(dup_query)
        existing_record = dup_res.scalar_one_or_none()

        if existing_record is not None:
            logger.info(
                "Student '%s' is already marked present in session '%s'.",
                student.student_id,
                request.session_id,
            )
            return AttendanceMarkResponse(
                status=AttendanceStatus.ALREADY_MARKED,
                student_id=student.student_id,
                student_name=student.name,
                session_id=request.session_id,
                similarity_score=existing_record.similarity_score,
                timestamp=existing_record.timestamp,
                record_id=existing_record.id,
                is_live=existing_record.is_live if hasattr(existing_record, "is_live") else request.is_live,
                liveness_score=existing_record.liveness_score if hasattr(existing_record, "liveness_score") else request.liveness_score,
                message=f"Student '{student.name}' ({student.student_id}) was already marked PRESENT in this session.",
            )

        # 7. Atomic record insertion with transaction safety
        new_attendance = Attendance(
            student_id=student.id,
            session_id=request.session_id,
            timestamp=event_time,
            similarity_score=request.similarity_score,
            status=AttendanceStatus.PRESENT.value,
            is_live=request.is_live if request.is_live is not None else True,
            liveness_score=request.liveness_score,
        )
        db.add(new_attendance)

        try:
            await db.flush()
        except IntegrityError:
            await db.rollback()
            # Race condition handling: re-fetch existing duplicate record
            concurrent_dup_res = await db.execute(dup_query)
            dup = concurrent_dup_res.scalar_one_or_none()
            return AttendanceMarkResponse(
                status=AttendanceStatus.ALREADY_MARKED,
                student_id=student.student_id,
                student_name=student.name,
                session_id=request.session_id,
                similarity_score=dup.similarity_score if dup else request.similarity_score,
                timestamp=dup.timestamp if dup else event_time,
                record_id=dup.id if dup else None,
                is_live=request.is_live,
                liveness_score=request.liveness_score,
                message=f"Student '{student.name}' ({student.student_id}) was already marked PRESENT in this session.",
            )

        logger.info(
            "Marked attendance PRESENT for student '%s' in session '%s' (score: %.3f)",
            student.student_id,
            request.session_id,
            request.similarity_score,
        )
        return AttendanceMarkResponse(
            status=AttendanceStatus.PRESENT,
            student_id=student.student_id,
            student_name=student.name,
            session_id=request.session_id,
            similarity_score=request.similarity_score,
            timestamp=new_attendance.timestamp,
            record_id=new_attendance.id,
            is_live=new_attendance.is_live,
            liveness_score=new_attendance.liveness_score,
            message=f"Student '{student.name}' ({student.student_id}) marked PRESENT.",
        )

    async def get_session_attendance(
        self,
        session_id: str,
        session: Optional[AsyncSession] = None,
    ) -> SessionAttendanceSummary:
        """Retrieve all verified attendance records for a specified session.
        
        Args:
            session_id: Session identifier.
            session: Optional active AsyncSession.
            
        Returns:
            SessionAttendanceSummary including course details and verified attendees.
            
        Raises:
            SessionNotFoundError: If session does not exist.
        """
        db = self._resolve_session(session)
        sess = await self.get_session(session_id=session_id, session=db)

        stmt = (
            select(Attendance, Student)
            .join(Student, Attendance.student_id == Student.id)
            .where(Attendance.session_id == session_id)
            .order_by(Attendance.timestamp.asc())
        )
        result = await db.execute(stmt)
        rows = result.all()

        records = [
            SessionAttendanceRecord(
                id=att.id,
                student_id=stu.student_id,
                student_name=stu.name,
                department=stu.department,
                similarity_score=att.similarity_score,
                status=att.status,
                timestamp=att.timestamp,
            )
            for att, stu in rows
        ]

        return SessionAttendanceSummary(
            session_id=sess.session_id,
            course_name=sess.course_name,
            instructor_name=sess.instructor_name,
            is_active=sess.is_active,
            created_at=sess.created_at,
            total_present=len(records),
            records=records,
        )


def get_attendance_service() -> AttendanceService:
    """FastAPI dependency provider returning an AttendanceService instance."""
    return AttendanceService()

