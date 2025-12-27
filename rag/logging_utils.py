"""Logging utilities.

Use `configure_logging()` from entrypoints/scripts to get consistent formatting.
"""

from __future__ import annotations

import logging
from typing import Optional


def _parse_level(level: str) -> int:
    if not isinstance(level, str) or not level.strip():
        raise ValueError("log level must be a non-empty string (e.g., 'INFO', 'DEBUG')")
    name = level.strip().upper()
    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    if name in mapping:
        return mapping[name]
    # Also accept numeric levels like "20"
    try:
        return int(name)
    except ValueError as e:
        raise ValueError(f"Unknown log level: {level!r}") from e


def configure_logging(level: str = "INFO", logger_name: Optional[str] = None) -> None:
    """Configure console logging with a consistent format.

    This is safe to call multiple times: it won't add duplicate handlers.

    Args:
        level: log level name (e.g., "INFO", "DEBUG") or numeric (e.g., "20")
        logger_name: if provided, configures this named logger; otherwise configures root logger
    """
    target = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    target.setLevel(_parse_level(level))

    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Avoid duplicate handlers if called repeatedly.
    for h in list(target.handlers):
        if isinstance(h, logging.StreamHandler):
            h.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
            h.setLevel(target.level)
            return

    handler = logging.StreamHandler()
    handler.setLevel(target.level)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    target.addHandler(handler)


