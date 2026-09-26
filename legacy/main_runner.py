#!/usr/bin/env python3
"""CLI principal para orquestar validación, entrenamiento, optimización y backtest."""

import argparse

from core.config_validation import validate_config
from core.logging_utils import log_event


def run_validate() -> None:
    validate_config()
    log_event({"event": "VALIDATION_OK"})


def run_train() -> None:
    from phoenix_evolution import ejecutar_modo_dios

    ejecutar_modo_dios()
    log_event({"event": "TRAIN_COMPLETE"})


def run_optimize() -> None:
    from phoenix_parameter_optimizer_v2 import FastParameterOptimizer
    import phoenix_config as config

    optimizer = FastParameterOptimizer(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
    optimizer.optimizar()
    log_event({"event": "OPTIMIZE_COMPLETE"})


def run_backtest() -> None:
    from phoenix_backtester_pro import ejecutar_backtest_profesional

    ejecutar_backtest_profesional()
    log_event({"event": "BACKTEST_COMPLETE"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    args = parser.parse_args()

    if args.validate:
        run_validate()
    if args.train:
        run_train()
    if args.optimize:
        run_optimize()
    if args.backtest:
        run_backtest()


if __name__ == "__main__":
    main()
