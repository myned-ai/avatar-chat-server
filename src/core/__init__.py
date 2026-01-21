"""
Core Module

Provides foundational utilities used across the application:
- Configuration management
- Logging setup
"""

from .config import (
    AudioConstants,
    Settings,
    get_allowed_origins,
    get_audio_constants,
    get_settings,
)
from .logger import get_logger, set_log_level, setup_logging

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
