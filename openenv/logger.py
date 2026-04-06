from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .runtime_config import DEFAULT_LOG_LEVEL, runtime_log_level


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
        }
        extra = getattr(record, "payload", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, sort_keys=True)


@dataclass(slots=True)
class StructuredLogger:
    name: str
    log_path: Optional[Path] = None
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(self.name)
        level_name = runtime_log_level(DEFAULT_LOG_LEVEL)
        self._logger.setLevel(getattr(logging, level_name, logging.WARNING))
        self._logger.propagate = False
        if not self._logger.handlers:
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setFormatter(JsonFormatter())
            self._logger.addHandler(stream_handler)
        if self.log_path is not None and not any(
            isinstance(handler, logging.FileHandler) for handler in self._logger.handlers
        ):
            file_handler = logging.FileHandler(self.log_path)
            file_handler.setFormatter(JsonFormatter())
            self._logger.addHandler(file_handler)

    def info(self, message: str, **payload: Any) -> None:
        self._logger.info(message, extra={"payload": payload})

    def warning(self, message: str, **payload: Any) -> None:
        self._logger.warning(message, extra={"payload": payload})

    def error(self, message: str, **payload: Any) -> None:
        self._logger.error(message, extra={"payload": payload})
