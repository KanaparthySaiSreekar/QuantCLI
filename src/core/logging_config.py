"""
Logging configuration for QuantCLI.
"""

import sys
import logging
from pathlib import Path
from loguru import logger
from typing import Optional


class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages and redirect to loguru.
    """

    def emit(self, record):
        # Get corresponding Loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "500 MB",
    retention: str = "30 days",
    json_logs: bool = False
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for console only)
        rotation: Log rotation policy
        retention: Log retention policy
        json_logs: Whether to use JSON format for structured logging
    """
    # Remove default logger
    logger.remove()

    # Console logging
    if json_logs:
        logger.add(
            sys.stderr,
            format="{message}",
            level=level,
            serialize=True  # JSON format
        )
    else:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level=level,
            colorize=True
        )

    # File logging
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        if json_logs:
            logger.add(
                str(log_file),
                format="{message}",
                level=level,
                rotation=rotation,
                retention=retention,
                compression="gz",
                serialize=True
            )
        else:
            logger.add(
                str(log_file),
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                       "{level: <8} | "
                       "{name}:{function}:{line} | "
                       "{message}",
                level=level,
                rotation=rotation,
                retention=retention,
                compression="gz"
            )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0)

    # Intercept specific loggers
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]


def get_logger(name: str):
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Setup default logging
log_level = "INFO"
log_dir = Path(__file__).parent.parent.parent / "logs"
log_file = log_dir / "quantcli.log"

setup_logging(
    level=log_level,
    log_file=log_file,
    rotation="500 MB",
    retention="30 days",
    json_logs=False
)
