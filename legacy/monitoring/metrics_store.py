import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np

METRICS_PATH = Path("logs/metrics.jsonl")


def _to_native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ _to_native(v) for v in value ]
    return value


def log_metrics(metrics: Dict[str, Any]) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = _to_native({"timestamp": datetime.utcnow().isoformat(), **metrics})
    with METRICS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
