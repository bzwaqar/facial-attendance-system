"""Logging configuration for the application."""

import logging
import os
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """Configure structured console and file logging across the application.

    Args:
        log_level: Desired log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory to persist log files.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "app.log", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)
    except Exception as exc:
        sys.stderr.write(f"Warning: Could not configure file logging: {exc}\n")

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )

    # Silence overly verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(numeric_level)
    logging.getLogger("uvicorn.error").setLevel(numeric_level)

