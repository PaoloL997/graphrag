import logging
from typing import Optional


def get_logger(
    name: Optional[str] = None, level: int = logging.WARNING
) -> logging.Logger:
    """Return a configured logger for the given name.

    The logger is configured with a StreamHandler and a simple formatter only
    if it has no handlers yet, so calling this multiple times is safe.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
