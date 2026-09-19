"""Schema definitions for system health checks."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Payload returned by the /health endpoint."""

    status: str = Field(default="healthy", description="Operational status of the backend")
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Active runtime environment")
    database: Optional[str] = Field(default=None, description="Database connectivity status")
    timestamp: datetime = Field(..., description="Current server UTC timestamp")
