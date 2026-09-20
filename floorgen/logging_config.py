"""
Logging and observability setup for FloorGen.
Supports standard console formatting and structured JSON logging.
"""

import sys
import logging
import json
from typing import Optional


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON lines for production log aggregators."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging(level: str = "INFO", json_format: bool = False) -> logging.Logger:
    """Configures application-wide logging."""
    root = logging.getLogger("floorgen")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Avoid duplicate handlers
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if json_format:
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
        root.addHandler(handler)
        
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Returns a named logger under the 'floorgen' namespace."""
    if name:
        return logging.getLogger(f"floorgen.{name}")
    return logging.getLogger("floorgen")
