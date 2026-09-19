"""Pydantic schemas and enums for attendance sessions and attendance engine."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class AttendanceStatus(str, Enum):
    """Attendance verification and processing outcome statuses."""

    PRESENT = "PRESENT"                  # Attendance successfully verified and recorded
    ALREADY_MARKED = "ALREADY_MARKED"    # Student was already marked present in this session
    UNKNOWN = "UNKNOWN"                  # Face identity is unknown / unverified
    REJECTED = "REJECTED"                # Attendance rejected due to business rules (e.g. low score, inactive, closed session)


class AttendanceSessionCreate(BaseModel):
    """Payload to create a new attendance session / class."""

    session_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Unique identifier for the session (e.g., 'CS601-20260919-01').",
        examples=["CS601-20260919-01"],
    )
    course_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Course or class title.",
        examples=["Computer Vision and Biometrics"],
    )
    instructor_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Faculty member or instructor in charge.",
        examples=["Dr. Sarah Jenkins"],
    )
    min_similarity_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional session-specific similarity threshold override.",
    )
    min_confirmations: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="Optional session-specific minimum temporal confirmations override.",
    )


class AttendanceSessionResponse(BaseModel):
    """Attendance session details representation."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str
    course_name: Optional[str] = None
    instructor_name: Optional[str] = None
    is_active: bool
    min_similarity_threshold: Optional[float] = None
    min_confirmations: Optional[int] = None
    created_at: datetime
    closed_at: Optional[datetime] = None
    total_attendees: int = 0


class AttendanceMarkRequest(BaseModel):
    """Payload to submit a verified face recognition event to the attendance engine."""

    session_id: str = Field(..., description="Target attendance session identifier.")
    student_id: Optional[str] = Field(
        None,
        description="University student identifier (e.g. 'STU1001'). None if UNKNOWN face.",
    )
    student_pk: Optional[int] = Field(
        None,
        description="Database primary key of student if known.",
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Recognition similarity score from biometric matcher.",
    )
    observation_count: int = Field(
        default=1,
        ge=1,
        description="Number of stable temporal observations recorded for this face.",
    )
    is_confirmed: bool = Field(
        default=True,
        description="Whether temporal verifier marked this identity as CONFIRMED.",
    )
    timestamp: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Verification event timestamp (defaults to current UTC time).",
    )
    is_live: Optional[bool] = Field(
        None,
        description="Whether anti-spoofing liveness verification passed.",
    )
    liveness_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score from the liveness detection layer.",
    )
    liveness_status: Optional[str] = Field(
        None,
        description="Liveness status string (e.g., LIVE, SPOOF, UNCERTAIN).",
    )
    liveness_details: Optional[dict] = Field(
        None,
        description="Detailed telemetry or cues from liveness evaluation.",
    )


class AttendanceMarkResponse(BaseModel):
    """Result returned by the attendance engine when processing an attendance request."""

    status: AttendanceStatus = Field(..., description="Attendance outcome status.")
    student_id: Optional[str] = Field(None, description="University student ID if applicable.")
    student_name: Optional[str] = Field(None, description="Student name if resolved.")
    session_id: str = Field(..., description="Attendance session ID.")
    similarity_score: float = Field(..., description="Similarity score evaluated.")
    timestamp: datetime = Field(..., description="Timestamp of the attendance event.")
    record_id: Optional[int] = Field(None, description="Database record ID if marked present.")
    is_live: Optional[bool] = Field(None, description="Liveness verification status evaluated.")
    liveness_score: Optional[float] = Field(None, description="Liveness score evaluated.")
    message: str = Field(..., description="Human-readable result explanation.")


class SessionAttendanceRecord(BaseModel):
    """A single attendance log record for a student in a session."""

    id: int
    student_id: str
    student_name: str
    department: str
    similarity_score: float
    status: str
    timestamp: datetime


class SessionAttendanceSummary(BaseModel):
    """Full attendance report for a specified session."""

    session_id: str
    course_name: Optional[str] = None
    instructor_name: Optional[str] = None
    is_active: bool
    created_at: datetime
    total_present: int
    records: List[SessionAttendanceRecord] = Field(default_factory=list)
