"""Pydantic schemas for Student entity management."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class StudentBase(BaseModel):
    """Base student attributes shared across schemas."""

    student_id: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Unique student identifier (e.g., 'CS-2026-001').",
        examples=["CS-2026-001"],
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Full legal or registered name of the student.",
        examples=["Sarah Connor"],
    )
    department: Optional[str] = Field(
        default="Computer Science",
        max_length=100,
        description="Academic department name.",
        examples=["Computer Science"],
    )
    semester: Optional[str] = Field(
        default="1st",
        max_length=20,
        description="Current active semester.",
        examples=["6th"],
    )
    section: Optional[str] = Field(
        default="A",
        max_length=20,
        description="Section code.",
        examples=["A"],
    )
    is_active: bool = Field(
        default=True,
        description="Whether the student record is currently active.",
    )


class StudentCreate(StudentBase):
    """Schema for registering a new student."""

    pass


class StudentUpdate(BaseModel):
    """Schema for updating an existing student's details."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    department: Optional[str] = Field(None, min_length=1, max_length=100)
    semester: Optional[str] = Field(None, min_length=1, max_length=20)
    section: Optional[str] = Field(None, min_length=1, max_length=20)
    is_active: Optional[bool] = None


class StudentResponse(StudentBase):
    """Response schema returned to clients representing a student."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Database primary key integer ID.")
    created_at: datetime = Field(..., description="UTC timestamp of record creation.")
    face_embeddings_count: int = Field(
        default=0,
        description="Number of valid biometric face embeddings enrolled for this student.",
    )
    is_enrolled: bool = Field(
        default=False,
        description="True if the student has sufficient reference embeddings enrolled (>= 5).",
    )
