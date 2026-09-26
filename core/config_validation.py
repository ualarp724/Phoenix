import os
import phoenix_config as config


def validate_config() -> None:
    required_paths = [
        config.DATA_RAW,
        config.MODEL_SAVE_PATH,
        config.SCALER_SAVE_PATH,
    ]
    for path in required_paths:
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    if config.MIN_STOP_LOSS_PIPS > config.MAX_STOP_LOSS_PIPS:
        raise ValueError("Invalid stop loss bounds")
    if config.RISK_REWARD_RATIO_MIN <= 0:
        raise ValueError("Invalid risk reward minimum")
