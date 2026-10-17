"""Consistent console logging for scripts, training runs and the API."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
