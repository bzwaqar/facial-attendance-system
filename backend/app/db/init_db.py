"""Database initialization script to create tables in PostgreSQL."""

import argparse
import asyncio
import logging
import sys

from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.base import Base
from app.db.session import async_engine, sync_engine

logger = logging.getLogger(__name__)


async def async_create_tables(engine: AsyncEngine, drop_existing: bool = False) -> None:
    """Create all defined tables asynchronously using AsyncEngine."""
    async with engine.begin() as conn:
        if drop_existing:
            logger.warning("Dropping all existing database tables...")
            await conn.run_sync(Base.metadata.drop_all)
            logger.info("All existing tables dropped.")

        logger.info("Creating database tables according to SQLAlchemy metadata...")
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Tables created successfully.")


def sync_create_tables(drop_existing: bool = False) -> None:
    """Create all defined tables synchronously using sync_engine."""
    if drop_existing:
        logger.warning("Dropping all existing database tables (sync)...")
        Base.metadata.drop_all(bind=sync_engine)
        logger.info("All existing tables dropped.")

    logger.info("Creating database tables according to SQLAlchemy metadata (sync)...")
    Base.metadata.create_all(bind=sync_engine)
    inspector = inspect(sync_engine)
    tables = inspector.get_table_names()
    logger.info("Current tables in database: %s", tables)


async def main() -> None:
    """CLI entry point for initializing database schema."""
    parser = argparse.ArgumentParser(description="Initialize database schema.")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop all existing tables before recreating them.",
    )
    args = parser.parse_args()

    setup_logging(settings.LOG_LEVEL)
    logger.info("Connecting to database: %s:%s/%s", settings.POSTGRES_SERVER, settings.POSTGRES_PORT, settings.POSTGRES_DB)

    try:
        await async_create_tables(async_engine, drop_existing=args.drop)
        logger.info("Database initialization completed successfully.")
    except Exception as exc:
        logger.error("Failed to initialize database: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
