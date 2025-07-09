from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logger with Rich handler and optional file handler."""
    logging_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [RichHandler(rich_tracebacks=True, markup=True)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger by name (after root configured)."""
    return logging.getLogger(name) 