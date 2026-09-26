import json
import logging
from datetime import datetime
from typing import Any, Dict

LOG_NAME = "phoenyx"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOG_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}'
    )

    file_handler = logging.FileHandler("logs/trading.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def log_event(event: Dict[str, Any]) -> None:
    logger = get_logger()
    payload = {"ts": datetime.utcnow().isoformat(), **event}
    logger.info(json.dumps(payload, ensure_ascii=False))
