"""
Centralized logging configuration for avatar_chat_server.

Provides a configured logger instance with appropriate formatting and levels
optimized for realtime communication performance.
"""

import logging
import sys
from typing import Optional

# Global log level - can be controlled via environment
_log_level: Optional[int] = None


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global _log_level

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    _log_level = numeric_level

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Reduce noise from verbose libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """
    Change the log level at runtime.

    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global _log_level

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    _log_level = numeric_level

    # Update root logger
    logging.getLogger().setLevel(numeric_level)
