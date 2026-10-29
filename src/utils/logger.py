"""
Structured logging for LabSentinel-RAG
"""

import sys
from loguru import logger

def setup_logger(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> — <level>{message}</level>",
        level=level,
        colorize=True,
    )
    logger.add(
        "outputs/labsentinel.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time} | {level} | {name}:{function} — {message}",
    )

setup_logger()
