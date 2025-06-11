"""
Utility module for setting up and retrieving loggers for different components.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from docrag.schema.settings import logging_settings


def get_logger(
    name: str, level: int = logging.INFO, log_file_path: Path | None = None
) -> logging.Logger:
    """
    Create or retrieve a configured logger for a given component.

    Args:
        name (str): Name of the logger (also the key in `logging_settings.log_files`).
        level (int): Logging level (e.g. `logging.INFO`).
        log_file_path (Path): Optional override path. If None, uses the path from settings.

    Returns:
        logging.Logger: A configured logger.

    Raises:
        ValueError: If no log file is configured for `name` in settings.
    """
    # 1) Resolve the path
    if log_file_path is None:
        try:
            log_file_path = logging_settings.log_files[name]
        except KeyError:
            raise ValueError(
                f"No log file configured for component '{name}'. "
                "Check `logging_settings.log_files`."
            )

    # 2) Ensure parent directory exists
    path = Path(log_file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 3) Get or create the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 4) Attach handler if not already configured
    if not logger.handlers:
        handler = RotatingFileHandler(
            filename=str(path),
            maxBytes=10 * 1024 * 1024,  # rotate after 10 MiB
            backupCount=5,  # keep up to 5 old log files
            encoding="utf-8",
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
