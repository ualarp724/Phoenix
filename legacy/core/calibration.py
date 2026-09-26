import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch

import phoenix_config as config


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits
    return logits / temperature


def load_temperature() -> float:
    path = Path(config.CALIBRATION_PATH)
    if not path.exists():
        return 1.0
    try:
        data = json.loads(path.read_text())
        return float(data.get("temperature", 1.0))
    except Exception:
        return 1.0


def save_temperature(temperature: float) -> None:
    Path(config.CALIBRATION_PATH).write_text(
        json.dumps({"temperature": float(temperature)})
    )


def find_best_temperature(
    logits: Iterable[torch.Tensor],
    labels: Iterable[torch.Tensor],
    grid: Tuple[float, float, int] = (0.5, 2.5, 21),
) -> float:
    """Grid search simple para temperature scaling."""
    all_logits = torch.cat(list(logits))
    all_labels = torch.cat(list(labels))

    t_min, t_max, steps = grid
    candidates = np.linspace(t_min, t_max, steps)
    best_t = 1.0
    best_loss = float("inf")

    for t in candidates:
        scaled = apply_temperature(all_logits, float(t))
        loss = torch.nn.functional.cross_entropy(scaled, all_labels).item()
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)

    return best_t
