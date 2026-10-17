"""Rotating log and a startup phase watchdog.

Each phase is a named checkpoint. If the process hangs, the last phase in the
log is where it stopped. Does not enumerate windows.
"""
from __future__ import annotations

import logging
import os
import time
from logging.handlers import RotatingFileHandler

from qs.version import LOG_PATH, TOOL_VERSION

__all__ = ["LOG_PATH", "get_logger", "log_line", "phase", "current_phase", "log_start"]

_logger = None
_phase = "boot"


def get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    log = logging.getLogger("qs")
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = RotatingFileHandler(
            LOG_PATH, maxBytes=2_000_000, backupCount=4, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
        log.addHandler(handler)
        log.propagate = False
    _logger = log
    return log


def log_line(msg: str) -> None:
    get_logger().info(str(msg).rstrip())


def phase(name: str, detail: str = "") -> None:
    """Record that startup (or a job) reached `name`."""
    global _phase
    _phase = name
    extra = f" {detail}" if detail else ""
    log_line(f"phase {name}{extra}")


def current_phase() -> str:
    return _phase


def log_start(kind: str, extra: str = "") -> None:
    phase("start", f"{kind} v{TOOL_VERSION} pid={os.getpid()} t={time.strftime('%H:%M:%S')} {extra}".strip())
