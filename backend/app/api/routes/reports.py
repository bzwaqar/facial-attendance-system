"""FastAPI router endpoints for administrative reports and dashboard analytics."""

from datetime import datetime, timezone
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.attendance import Attendance
from app.db.models.attendance_session import AttendanceSession
from app.db.models.face_embedding import FaceEmbedding
from app.db.models.student import Student
from app.db.session import get_db
from app.schemas.reports import (
    RecentActivityRecord,
    SessionDetailedReport,
    SystemOverviewReport,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Reports & Analytics"])


@router.get(
    "/overview",
    response_model=SystemOverviewReport,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Dashboard System Overview",
    description="Aggregate key metrics including student counts, session counts, attendance totals, and recent activity.",
)
async def get_system_overview(
    db: AsyncSession = Depends(get_db),
    recent_limit: int = Query(default=15, ge=1, le=50, description="Number of recent records to retrieve"),
) -> SystemOverviewReport:
    """Compile aggregated statistics for dashboard display."""
    # 1. Student metrics
    total_students_res = await db.execute(select(func.count(Student.id)))
    total_students = int(total_students_res.scalar() or 0)

    active_students_res = await db.execute(
        select(func.count(Student.id)).where(Student.is_active.is_(True))
    )
    active_students = int(active_students_res.scalar() or 0)

    # Students with >= 5 embeddings
    enrolled_subq = (
        select(FaceEmbedding.student_id)
        .group_by(FaceEmbedding.student_id)
        .having(func.count(FaceEmbedding.id) >= 5)
        .subquery()
    )
    enrolled_res = await db.execute(select(func.count()).select_from(enrolled_subq))
    enrolled_students = int(enrolled_res.scalar() or 0)

    # 2. Session metrics
    total_sessions_res = await db.execute(select(func.count(AttendanceSession.id)))
    total_sessions = int(total_sessions_res.scalar() or 0)

    active_sessions_res = await db.execute(
        select(func.count(AttendanceSession.id)).where(AttendanceSession.is_active.is_(True))
    )
    active_sessions = int(active_sessions_res.scalar() or 0)

    # 3. Attendance records metrics
    total_att_res = await db.execute(select(func.count(Attendance.id)))
    total_attendance = int(total_att_res.scalar() or 0)

    # Today's records (UTC date)
    now_utc = datetime.now(timezone.utc)
    start_of_day = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc)
    today_att_res = await db.execute(
        select(func.count(Attendance.id)).where(Attendance.timestamp >= start_of_day)
    )
    today_attendance = int(today_att_res.scalar() or 0)

    # Average similarity
    avg_sim_res = await db.execute(select(func.avg(Attendance.similarity_score)))
    avg_sim = float(avg_sim_res.scalar() or 0.0)

    # 4. Recent activity feed
    recent_query = (
        select(Attendance, Student)
        .join(Student, Attendance.student_id == Student.id)
        .order_by(Attendance.timestamp.desc())
        .limit(recent_limit)
    )
    recent_res = await db.execute(recent_query)
    recent_rows = recent_res.all()

    recent_records = [
        RecentActivityRecord(
            id=att.id,
            student_id=stu.student_id,
            student_name=stu.name,
            department=stu.department,
            session_id=att.session_id,
            similarity_score=round(float(att.similarity_score), 4),
            status=att.status,
            is_live=att.is_live,
            liveness_score=round(float(att.liveness_score), 4) if att.liveness_score is not None else None,
            timestamp=att.timestamp,
        )
        for att, stu in recent_rows
    ]

    return SystemOverviewReport(
        total_students=total_students,
        active_students=active_students,
        enrolled_students=enrolled_students,
        total_sessions=total_sessions,
        active_sessions=active_sessions,
        total_attendance_records=total_attendance,
        today_attendance_records=today_attendance,
        average_similarity=round(avg_sim, 4),
        recent_activity=recent_records,
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetailedReport,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Session Attendance Detailed Report",
    description="Fetch per-session attendance statistics including attendee rates and present student details.",
)
async def get_session_report(
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> SessionDetailedReport:
    """Generate a detailed report for a specific session."""
    sess_query = select(AttendanceSession).where(AttendanceSession.session_id == session_id)
    sess_res = await db.execute(sess_query)
    session_obj = sess_res.scalar_one_or_none()
    if not session_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attendance session '{session_id}' not found.",
        )

    # Fetch all attendances for this session
    att_query = (
        select(Attendance, Student)
        .join(Student, Attendance.student_id == Student.id)
        .where(Attendance.session_id == session_id)
        .order_by(Attendance.timestamp.asc())
    )
    att_res = await db.execute(att_query)
    rows = att_res.all()

    # Total active students as baseline
    total_active_res = await db.execute(
        select(func.count(Student.id)).where(Student.is_active.is_(True))
    )
    total_active = int(total_active_res.scalar() or 0)

    present_list = [
        RecentActivityRecord(
            id=att.id,
            student_id=stu.student_id,
            student_name=stu.name,
            department=stu.department,
            session_id=session_id,
            similarity_score=round(float(att.similarity_score), 4),
            status=att.status,
            is_live=att.is_live,
            liveness_score=round(float(att.liveness_score), 4) if att.liveness_score is not None else None,
            timestamp=att.timestamp,
        )
        for att, stu in rows
    ]

    total_present = len(present_list)
    att_rate = float((total_present / max(1, total_active)) * 100.0) if total_active > 0 else 0.0

    return SessionDetailedReport(
        session_id=session_obj.session_id,
        course_name=session_obj.course_name,
        instructor_name=session_obj.instructor_name,
        is_active=session_obj.is_active,
        created_at=session_obj.created_at,
        total_present=total_present,
        total_enrolled=total_active,
        attendance_rate_percent=round(att_rate, 2),
        present_students=present_list,
    )
