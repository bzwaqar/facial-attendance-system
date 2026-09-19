"""Database connection and session management using SQLAlchemy 2.x."""

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.exceptions import (
    DatabaseConnectionError,
    DatabaseError,
    DuplicateAttendanceError,
    DuplicateRecordError,
)

logger = logging.getLogger(__name__)

# Asynchronous Engine & Session Factory (Primary for FastAPI endpoints)
async_engine: AsyncEngine = create_async_engine(
    settings.async_database_uri,
    echo=settings.DEBUG,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Synchronous Engine & Session Factory (For CLI scripts, migrations, table init)
sync_engine = create_engine(
    settings.sync_database_uri,
    echo=settings.DEBUG,
    future=True,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)


def handle_integrity_error(err: IntegrityError) -> DatabaseError:
    """Translate raw SQLAlchemy IntegrityError into domain-specific exceptions."""
    err_str = str(err.orig) if err.orig else str(err)
    if "uq_student_session_attendance" in err_str:
        return DuplicateAttendanceError(
            student_id="<constrained>",
            session_id="<constrained>",
            message="Attendance has already been recorded for this student in this session.",
        )
    if "unique constraint" in err_str.lower():
        return DuplicateRecordError(f"Duplicate constraint violation: {err_str}")
    return DatabaseError(f"Database integrity violation: {err_str}")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an AsyncSession with automatic transaction rollback."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            logger.error("Database integrity error: %s", exc)
            raise handle_integrity_error(exc) from exc
        except OperationalError as exc:
            await session.rollback()
            logger.error("Database connection operational failure: %s", exc)
            raise DatabaseConnectionError(str(exc)) from exc
        except SQLAlchemyError as exc:
            await session.rollback()
            logger.error("Database error occurred: %s", exc)
            raise DatabaseError(str(exc)) from exc


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for standalone scripts or background tasks."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@contextmanager
def get_sync_db() -> Generator[Session, None, None]:
    """Synchronous context manager for sync scripts and database initialization."""
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        logger.error("Sync database integrity error: %s", exc)
        raise handle_integrity_error(exc) from exc
    except OperationalError as exc:
        session.rollback()
        logger.error("Sync database operational failure: %s", exc)
        raise DatabaseConnectionError(str(exc)) from exc
    except SQLAlchemyError as exc:
        session.rollback()
        logger.error("Sync database error: %s", exc)
        raise DatabaseError(str(exc)) from exc
    finally:
        session.close()


async def check_database_connection() -> bool:
    """Verify active database connectivity with a lightweight ping query."""
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database connectivity check failed: %s", exc)
        return False
