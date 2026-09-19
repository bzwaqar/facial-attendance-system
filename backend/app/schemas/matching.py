"""Pydantic schemas and enums for biometric identity matching and vector search."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class MatchStatus(str, Enum):
    """Recognition status outcomes for identity verification."""

    MATCHED = "MATCHED"          # Candidate similarity exceeded configured threshold
    UNKNOWN = "UNKNOWN"          # All candidate similarities fell below threshold
    EMPTY_INDEX = "EMPTY_INDEX"  # No enrolled embeddings present in the index


class MatchCandidate(BaseModel):
    """A single top-K search candidate retrieved from the vector index."""

    student_id: str = Field(..., description="University student identifier.")
    student_pk: Optional[int] = Field(None, description="Primary key ID in database.")
    similarity_score: float = Field(..., description="Cosine similarity score [-1.0 to 1.0].")
    vector_id: int = Field(..., description="Continuous 0-indexed position in FAISS.")
    embedding_id: Optional[int] = Field(None, description="FaceEmbedding database record ID.")


class IdentityMatchResult(BaseModel):
    """Structured result returned by the identity matching subsystem."""

    status: MatchStatus = Field(..., description="High-level matching status.")
    matched: bool = Field(..., description="True if top candidate exceeded recognition threshold.")
    student_id: Optional[str] = Field(None, description="Identified student ID, or None if UNKNOWN.")
    student_pk: Optional[int] = Field(None, description="Database primary key, or None if UNKNOWN.")
    similarity_score: float = Field(default=0.0, description="Highest candidate similarity score.")
    threshold_used: float = Field(..., description="Configurable recognition threshold applied.")
    candidates: List[MatchCandidate] = Field(
        default_factory=list,
        description="Top-K candidate list sorted by similarity descending for auditing/debugging.",
    )
    search_time_ms: float = Field(default=0.0, description="Search inference latency in milliseconds.")
    message: str = Field(default="", description="Human-readable description of the match outcome.")
