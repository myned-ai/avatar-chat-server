"""
Core Module

Provides foundational utilities used across the application:
- Configuration management
- Logging setup
"""

from .config import (
    Settings,
    AudioConstants,
    get_settings,
    get_audio_constants,
    get_allowed_origins,
)
from .logger import setup_logging, get_logger, set_log_level

__all__ = [
    # Config
    "Settings",
    "AudioConstants",
    "get_settings",
    "get_audio_constants",
    "get_allowed_origins",
    # Logging
    "setup_logging",
    "get_logger",
    "set_log_level",
]
