"""FastAPI Application entry point and factory."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.health import router as health_router
from app.api.v1.api import api_v1_router
from app.core.config import settings
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for handling startup and shutdown events."""
    setup_logging(settings.LOG_LEVEL)
    logger.info(
        "Starting %s v%s [Env: %s, Debug: %s]",
        settings.PROJECT_NAME,
        settings.VERSION,
        settings.ENVIRONMENT,
        settings.DEBUG,
    )
    # Warm up FAISS index from DB
    try:
        from app.db.session import AsyncSessionLocal
        from app.services.matching_service import get_identity_matching_service
        async with AsyncSessionLocal() as session:
            matching_service = get_identity_matching_service()
            count = await matching_service.rebuild_index_from_db(session)
            logger.info("Startup: Loaded %d face embeddings into FAISS index.", count)
    except Exception as exc:
        logger.warning("Startup: FAISS index warmup skipped: %s", exc)
    yield
    logger.info("Shutting down %s", settings.PROJECT_NAME)


def create_app() -> FastAPI:
    """FastAPI Application factory."""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        debug=settings.DEBUG,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # Configure CORS
    if settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Global Exception Handlers for standardized error payloads
    @application.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "status_code": exc.status_code,
                "detail": exc.detail,
                "path": request.url.path,
            },
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "status_code": 422,
                "detail": exc.errors(),
                "path": request.url.path,
            },
        )

    @application.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server error at %s: %s", request.url.path, str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "status_code": 500,
                "detail": "Internal server error occurred.",
                "path": request.url.path,
            },
        )

    # Mount static assets if directory exists
    if STATIC_DIR.is_dir():
        application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @application.get("/dashboard", include_in_schema=False)
        async def dashboard_view():
            return FileResponse(STATIC_DIR / "index.html")

    # Root redirect to dashboard (or informational API JSON if Accept is json)
    @application.get("/", include_in_schema=False)
    async def root(request: Request):
        accept_header = request.headers.get("accept", "")
        if "text/html" in accept_header and STATIC_DIR.is_dir():
            return RedirectResponse(url="/dashboard", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
        return JSONResponse(
            content={
                "name": settings.PROJECT_NAME,
                "version": settings.VERSION,
                "status": "online",
                "dashboard": "/dashboard" if STATIC_DIR.is_dir() else None,
                "health_check": "/health",
                "documentation": "/docs" if settings.DEBUG else None,
            }
        )

    # Register root health check endpoint: GET /health
    application.include_router(health_router, prefix="")

    # Register versioned API endpoints: /api/v1
    application.include_router(api_v1_router, prefix=settings.API_V1_STR)

    return application


# Global application instance for ASGI servers (e.g. uvicorn)
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

