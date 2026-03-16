"""Structured logging setup."""
import logging
import sys
import io
from pathlib import Path


def _safe_stream():
    """Return a UTF-8 safe stream for console logging on Windows."""
    try:
        # Python 3.7+ on Windows: reconfigure stdout to UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        return sys.stdout
    except (AttributeError, io.UnsupportedOperation):
        # Fallback: wrap in a TextIOWrapper
        try:
            return io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True,
            )
        except AttributeError:
            return sys.stdout


def get_logger(name: str, log_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to console and optionally to a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler (UTF-8 safe)
    ch = logging.StreamHandler(_safe_stream())
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

