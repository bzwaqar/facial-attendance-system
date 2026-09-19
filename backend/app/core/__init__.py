"""Core module containing application configuration and logging setup."""

from app.core.config import get_settings, settings
from app.core.logging import setup_logging

__all__ = ["get_settings", "settings", "setup_logging"]
