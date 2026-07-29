"""Structured logging setup shared by scripts, API and dashboard.

Call :func:`configure_logging` once at process start (entrypoints do this).
Library modules should only call :func:`get_logger` and must never configure
handlers themselves.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_CONFIGURED = False

_RESERVED_RECORD_KEYS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise ``record`` to a JSON string.

        Args:
            record: The log record.

        Returns:
            A single-line JSON document.
        """
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value for key, value in record.__dict__.items() if key not in _RESERVED_RECORD_KEYS
        }
        if extras:
            payload["context"] = extras
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%dT%H:%M:%S",
    json_output: bool = False,
    force: bool = False,
) -> None:
    """Install a single stdout handler on the root logger.

    Repeated calls are ignored unless ``force`` is set, so importing an
    entrypoint from another entrypoint cannot duplicate handlers.

    Args:
        level: Root log level name.
        fmt: ``logging`` format string used when ``json_output`` is false.
        datefmt: Timestamp format.
        json_output: Emit JSON lines instead of formatted text.
        force: Reconfigure even if logging was already configured.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else logging.Formatter(fmt, datefmt))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Third-party loggers are noisy at INFO; keep them at WARNING.
    for noisy in ("matplotlib", "shap", "numba", "urllib3", "PIL", "watchdog"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def configure_from_config(config: Any, force: bool = False) -> None:
    """Configure logging from a loaded :class:`~wind_turbine_pm.config.Config`.

    Args:
        config: Object exposing ``logging.level`` / ``logging.format`` keys.
        force: Reconfigure even if logging was already configured.
    """
    configure_logging(
        level=str(config.get("logging.level", "INFO")),
        fmt=str(
            config.get("logging.format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        ),
        datefmt=str(config.get("logging.datefmt", "%Y-%m-%dT%H:%M:%S")),
        json_output=bool(config.get("logging.json", False)),
        force=force,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.

    Args:
        name: Usually ``__name__``.

    Returns:
        The named :class:`logging.Logger`.
    """
    return logging.getLogger(name)
