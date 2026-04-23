"""
TARA ADAS — Logging Utility
Configurable logging for all ADAS modules.
"""
import logging
import sys
import os


def setup_logger(name="TARA", level="INFO", log_file=None):
    """
    Set up and return a configured logger.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console formatter — compact for real-time viewing
    console_fmt = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_fmt = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger(module_name):
    """
    Get a child logger for a specific module.
    Call setup_logger() first to configure the root TARA logger.

    Args:
        module_name: Name of the ADAS module (e.g., "LaneDetect", "TSR")

    Returns:
        Child logger instance
    """
    return logging.getLogger(f"TARA.{module_name}")
