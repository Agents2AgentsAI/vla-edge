"""Small colorized logging helpers.

Trimmed from the upstream ``gello.utils.logging_utils`` to the loggers and
helpers the MolmoAct eval path actually uses: the ``molmoact`` and
``collect_demos`` loggers plus ``log_collect_demos`` (used to report per-step
inference timing).
"""

import logging


class LogColors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a console logger with consistent formatting (no duplicate handlers)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_collect_demos_logger() -> logging.Logger:
    return setup_logger("collect_demos")


def get_molmoact_logger() -> logging.Logger:
    return setup_logger("molmoact")


def _colored(logger: logging.Logger, color: str, msg: str) -> None:
    logger.info(f"{color}{msg}{LogColors.END}")


def log_collect_demos(msg: str, level: str = "info") -> None:
    """Log a data/inference message with a level-dependent color."""
    logger = get_collect_demos_logger()
    color = {
        "info": LogColors.CYAN,
        "warning": LogColors.YELLOW,
        "success": LogColors.GREEN,
        "error": LogColors.RED,
        "config": LogColors.CYAN,
        "connect": LogColors.BLUE,
        "instruction": LogColors.YELLOW,
        "failure": LogColors.RED,
        "important": LogColors.BOLD + LogColors.HEADER,
        "data_info": LogColors.BLUE,
    }.get(level, LogColors.CYAN)
    _colored(logger, color, msg)
