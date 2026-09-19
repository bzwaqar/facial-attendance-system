"""Pydantic schemas for attendance and system overview reporting."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class RecentActivityRecord(BaseModel):
    """A single attendance log record displayed in the live feed."""

    id: int
    student_id: str
    student_name: str
    department: str
    session_id: str
    similarity_score: float
    status: str
    is_live: Optional[bool] = None
    liveness_score: Optional[float] = None
    timestamp: datetime


class SystemOverviewReport(BaseModel):
    """Aggregated system metrics for the administrative dashboard."""

    total_students: int = Field(..., description="Total registered students.")
    active_students: int = Field(..., description="Students with active status.")
    enrolled_students: int = Field(..., description="Students with >= 5 face embeddings.")
    total_sessions: int = Field(..., description="Total attendance sessions created.")
    active_sessions: int = Field(..., description="Currently active attendance sessions.")
    total_attendance_records: int = Field(..., description="Total attendance entries recorded.")
    today_attendance_records: int = Field(..., description="Attendance entries marked today.")
    average_similarity: float = Field(..., description="Average similarity score across all attendance records.")
    recent_activity: List[RecentActivityRecord] = Field(default_factory=list, description="Latest attendance events.")


class SessionDetailedReport(BaseModel):
    """Comprehensive attendance report for a specific session."""

    session_id: str
    course_name: Optional[str] = None
    instructor_name: Optional[str] = None
    is_active: bool
    created_at: datetime
    total_present: int
    total_enrolled: int
    attendance_rate_percent: float
    present_students: List[RecentActivityRecord] = Field(default_factory=list)
