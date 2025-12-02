"""
Logging configuration for CourierIQ
Place in: src/utils/logger.py
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )

        # Format the message
        result = super().format(record)

        # Reset levelname (in case record is used again)
        record.levelname = levelname

        return result


def setup_logger(
    name: str = "courieriq",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    console: bool = True,
    file: bool = True,
    colored: bool = True,
) -> logging.Logger:
    """
    Setup logger with console and file handlers

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        console: Enable console logging
        file: Enable file logging
        colored: Use colored console output

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()  # Clear existing handlers

    # Create formatters
    detailed_format = (
        "%(asctime)s | %(name)s | %(levelname)s | "
        "%(filename)s:%(lineno)d | %(funcName)s() | %(message)s"
    )

    simple_format = "%(asctime)s | %(levelname)s | %(message)s"

    date_format = "%Y-%m-%d %H:%M:%S"

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        if colored:
            console_formatter = ColoredFormatter(simple_format, datefmt=date_format)
        else:
            console_formatter = logging.Formatter(simple_format, datefmt=date_format)

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handlers
    if file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "logs"

        log_dir.mkdir(parents=True, exist_ok=True)

        # General log file (rotating by size)
        general_log = log_dir / f"{name}.log"
        file_handler = RotatingFileHandler(
            general_log,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(detailed_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Error log file (only errors and above)
        error_log = log_dir / f"{name}_error.log"
        error_handler = RotatingFileHandler(
            error_log,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)

        # Daily log file (rotating by time)
        daily_log = log_dir / f"{name}_daily.log"
        daily_handler = TimedRotatingFileHandler(
            daily_log,
            when="midnight",
            interval=1,
            backupCount=30,  # Keep 30 days
        )
        daily_handler.setLevel(logging.INFO)
        daily_handler.setFormatter(file_formatter)
        logger.addHandler(daily_handler)

    return logger


def get_logger(name: str = "courieriq") -> logging.Logger:
    """
    Get existing logger or create new one

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up
    if not logger.handlers:
        return setup_logger(name)

    return logger


class LoggerContext:
    """Context manager for temporary log level changes"""

    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = logger.level

    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def log_function_call(func):
    """Decorator to log function calls"""

    def wrapper(*args, **kwargs):
        logger = get_logger()
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}", exc_info=True)
            raise

    return wrapper


def log_execution_time(func):
    """Decorator to log function execution time"""
    import time

    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = time.time()

        result = func(*args, **kwargs)

        elapsed_time = time.time() - start_time
        logger.debug(f"{func.__name__} executed in {elapsed_time:.4f} seconds")

        return result

    return wrapper


# Example usage
if __name__ == "__main__":
    # Setup logger
    logger = setup_logger("courieriq", level="DEBUG")

    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Test context manager
    logger.info("Normal logging level")
    with LoggerContext(logger, "WARNING"):
        logger.debug("This won't appear")
        logger.info("This won't appear either")
        logger.warning("But this will")
    logger.info("Back to normal level")

    # Test decorators
    @log_function_call
    @log_execution_time
    def example_function(x, y):
        import time

        time.sleep(0.1)
        return x + y

    result = example_function(5, 3)
    logger.info(f"Result: {result}")
