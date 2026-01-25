"""
Centralized logging configuration for avatar_chat_server.

Provides a configured logger instance with appropriate formatting and levels
optimized for realtime communication performance.
"""

import logging
import sys

# Global log level - can be controlled via environment
_log_level: int | None = None


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
    root_logger = logging.getLogger()
    
    # Remove all existing handlers to prevent duplication
    while root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])
        
    root_logger.setLevel(numeric_level)
    
    # Add a single handler with our format
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Clean up uvicorn loggers to use our root configuration
    # This acts to unify the log format and prevent double-logging
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        log = logging.getLogger(logger_name)
        log.handlers = []
        log.propagate = True

    # Reduce noise from verbose libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    noisy_loggers = [
        "numba",
        "numba.core", 
        "numba.core.byteflow",
        "numba.core.ssa",
        "numba.core.interpreter",
        "httpcore",
        "httpx",
        "websockets",
        "multipart",
        "asyncio"
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)



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
