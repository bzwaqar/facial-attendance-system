"""Health check endpoint router."""

from datetime import datetime, timezone
from fastapi import APIRouter, status

from app.core.config import settings
from app.db.session import check_database_connection
from app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    description="Returns the operational status, database connectivity, version, environment, and server timestamp.",
)
async def get_health() -> HealthResponse:
    """Check health, operational readiness, and database connectivity."""
    db_connected = await check_database_connection()
    return HealthResponse(
        status="healthy",
        app_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        database="connected" if db_connected else "disconnected",
        timestamp=datetime.now(timezone.utc),
    )
