"""
Core Module

Provides foundational utilities used across the application:
- Configuration management
- Logging setup
"""

from .logger import get_logger, set_log_level, setup_logging
from .settings import (
    Settings,
    get_allowed_origins,
    get_settings,
)

__all__ = [
    # Config
    "Settings",
    "get_settings",
    "get_allowed_origins",
    # Logging
    "get_logger",
    "setup_logging",
    "set_log_level",
]
