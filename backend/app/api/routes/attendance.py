"""FastAPI router endpoints for attendance session management and attendance engine."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.attendance import (
    AttendanceMarkRequest,
    AttendanceMarkResponse,
    AttendanceSessionCreate,
    AttendanceSessionResponse,
    SessionAttendanceSummary,
)
from app.services.attendance_service import AttendanceService, get_attendance_service
from app.services.exceptions import DuplicateSessionError, SessionNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Attendance Engine"])


@router.post(
    "/sessions",
    response_model=AttendanceSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Attendance Session",
    description="Initialize a new attendance session / lecture with configurable verification rules.",
    responses={
        201: {"description": "Attendance session created successfully."},
        409: {"description": "Session with this session_id already exists."},
    },
)
async def create_attendance_session(
    session_in: AttendanceSessionCreate,
    db: AsyncSession = Depends(get_db),
    attendance_service: AttendanceService = Depends(get_attendance_service),
) -> AttendanceSessionResponse:
    """Create a new attendance session."""
    try:
        return await attendance_service.create_session(session_in=session_in, session=db)
    except DuplicateSessionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=exc.message,
        ) from exc


@router.get(
    "/sessions",
    response_model=List[AttendanceSessionResponse],
    summary="List Attendance Sessions",
    description="Retrieve list of attendance sessions with attendee counts.",
)
async def list_attendance_sessions(
    is_active: Optional[bool] = Query(None, description="Filter by active / inactive status"),
    limit: int = Query(50, ge=1, le=100, description="Pagination limit"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: AsyncSession = Depends(get_db),
    attendance_service: AttendanceService = Depends(get_attendance_service),
) -> List[AttendanceSessionResponse]:
    """List attendance sessions with pagination."""
    return await attendance_service.list_sessions(
        is_active=is_active,
        limit=limit,
        offset=offset,
        session=db,
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionAttendanceSummary,
    summary="Retrieve Session Attendance",
    description="Fetch all verified attendance records and summary for a given session.",
    responses={
        200: {"description": "Session attendance retrieved successfully."},
        404: {"description": "Attendance session not found."},
    },
)
async def get_session_attendance(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    attendance_service: AttendanceService = Depends(get_attendance_service),
) -> SessionAttendanceSummary:
    """Retrieve full attendance list for a session."""
    try:
        return await attendance_service.get_session_attendance(session_id=session_id, session=db)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.message,
        ) from exc


@router.patch(
    "/sessions/{session_id}/close",
    response_model=AttendanceSessionResponse,
    summary="Close Attendance Session",
    description="Close an active attendance session to prevent further records from being marked.",
    responses={
        200: {"description": "Session closed successfully."},
        404: {"description": "Attendance session not found."},
    },
)
async def close_attendance_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    attendance_service: AttendanceService = Depends(get_attendance_service),
) -> AttendanceSessionResponse:
    """Close an attendance session."""
    try:
        return await attendance_service.close_session(session_id=session_id, session=db)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.message,
        ) from exc


@router.post(
    "/mark",
    response_model=AttendanceMarkResponse,
    status_code=status.HTTP_200_OK,
    summary="Mark Attendance",
    description=(
        "Process a verified face recognition event through temporal verification rules, "
        "duplicate check, and record attendance in PostgreSQL."
    ),
    responses={
        200: {"description": "Attendance evaluated (returns PRESENT, ALREADY_MARKED, UNKNOWN, or REJECTED)."},
        404: {"description": "Attendance session not found."},
    },
)
async def mark_attendance(
    request: AttendanceMarkRequest,
    db: AsyncSession = Depends(get_db),
    attendance_service: AttendanceService = Depends(get_attendance_service),
) -> AttendanceMarkResponse:
    """Process recognition event and record attendance."""
    try:
        return await attendance_service.mark_attendance(request=request, session=db)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.message,
        ) from exc
