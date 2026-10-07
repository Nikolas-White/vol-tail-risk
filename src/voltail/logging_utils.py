"""Logging helpers: one consistent format for the CLI, tests and the API."""

from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger that writes to stderr with a uniform format."""
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers and not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
    return logger
