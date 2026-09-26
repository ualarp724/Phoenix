#!/usr/bin/env python3
"""
Optuna optimizer for Phoenix bot (XGBoost + thresholds + SL/TP + sessions).
Does not overwrite config unless --apply is provided.
"""

import argparse
import copy
import json
import os
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from phoenix_backtester_pro import ProfessionalBacktester
from phoenix_brain import preparar_secuencias_flat_con_scaler
from core.mtf import add_mtf_features_multi

try:
    import optuna
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Optuna no disponible: {exc}")

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"XGBoost no disponible: {exc}")


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _days_between(start: str, end: str) -> int:
    start_dt = _parse_date(start)
    end_dt = _parse_date(end)
    return max((end_dt.date() - start_dt.date()).days + 1, 1)


def _train_xgb(params, X_train, y_train):
    tuned = copy.deepcopy(params)
    tuned.setdefault("n_jobs", max(1, os.cpu_count() or 1))
    tuned.setdefault("tree_method", "hist")
    model = xgb.XGBClassifier(**tuned)
    model.fit(X_train, y_train, verbose=False)
    return model


def _load_data():
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df_all = add_mtf_features_multi(
        processor.clean_and_prepare(apply_train_filter=False),
        config.MTF_CONFIGS,
    )

    train_df = df_all[(df_all.index >= config.TRAIN_START_DATE) & (df_all.index <= config.TRAIN_END_DATE)]
    val_df = df_all[(df_all.index >= config.VAL_START_DATE) & (df_all.index <= config.VAL_END_DATE)]

    if len(train_df) < config.LOOKBACK_WINDOW + 100:
        raise SystemExit("Datos de entrenamiento insuficientes")
    if len(val_df) < config.LOOKBACK_WINDOW + 100:
        raise SystemExit("Datos de validación insuficientes")

    features = config.FEATURES
    scaler = joblib.load(config.SCALER_SAVE_PATH) if Path(config.SCALER_SAVE_PATH).exists() else None
    if scaler is None or getattr(scaler, "n_features_in_", None) != len(features):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_df[features].values)
        joblib.dump(scaler, config.SCALER_SAVE_PATH)

    X_train, y_train = preparar_secuencias_flat_con_scaler(train_df, scaler)
    val_df_bt = add_mtf_features_multi(val_df.copy(), config.MTF_CONFIGS)
    return X_train, y_train, val_df_bt, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if config.MODEL_TYPE not in {"xgboost", "ensemble"}:
        raise SystemExit("MODEL_TYPE debe ser xgboost o ensemble")

    X_train, y_train, val_df_bt, scaler = _load_data()
    days = _days_between(config.VAL_START_DATE, config.VAL_END_DATE)

    backtester = ProfessionalBacktester(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
    backtester.scaler = scaler
    backtester.model_lgbm = None

    def objective(trial: optuna.Trial):
        params = copy.deepcopy(config.XGB_PARAMS)
        params.update(
            {
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
                "gamma": trial.suggest_float("gamma", 0.0, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 3.0),
                "max_delta_step": trial.suggest_int("max_delta_step", 0, 1),
            }
        )

        threshold = trial.suggest_float("threshold", 0.14, 0.30)
        sl_mult = trial.suggest_float("sl_mult", 0.8, 1.6)
        tp_mult = trial.suggest_float("tp_mult", 1.2, 2.2)

        use_time_filter = trial.suggest_categorical("use_time_filter", [False, True])
        hour_start = trial.suggest_int("hour_start", 7, 12)
        hour_end = trial.suggest_int("hour_end", max(hour_start + 6, 13), 22)

        model = _train_xgb(params, X_train, y_train)
        backtester.model = model
        backtester.model_lgbm = None

        result = backtester.backtest_period(
            val_df_bt,
            umbral=threshold,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            precomputed_mtf=True,
            use_time_filter=use_time_filter,
            hour_start=hour_start,
            hour_end=hour_end,
        )
        if not result:
            return -1e9

        metrics = result["metrics"]
        profit = result["profit"]
        avg_daily = profit / days
        max_dd = metrics.get("max_drawdown_pct", 0.0)

        score = avg_daily - (max_dd * 0.02)
        trial.set_user_attr("profit", profit)
        trial.set_user_attr("avg_daily", avg_daily)
        trial.set_user_attr("win_rate", metrics.get("win_rate", 0.0))
        trial.set_user_attr("profit_factor", metrics.get("profit_factor", 0.0))
        trial.set_user_attr("max_drawdown_pct", max_dd)

        return score

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    best_payload = {
        "score": best.value,
        "params": best.params,
        "metrics": {
            "profit": best.user_attrs.get("profit"),
            "avg_daily": best.user_attrs.get("avg_daily"),
            "win_rate": best.user_attrs.get("win_rate"),
            "profit_factor": best.user_attrs.get("profit_factor"),
            "max_drawdown_pct": best.user_attrs.get("max_drawdown_pct"),
        },
    }

    output_path = Path(__file__).parent / "optuna_best.json"
    output_path.write_text(json.dumps(best_payload, indent=2))

    print("\n✅ MEJOR CONFIGURACIÓN (Optuna)")
    print(best_payload)
    print(f"Guardado en {output_path}")

    if not args.apply:
        return

    # Aplicar en config
    updated_params = copy.deepcopy(config.XGB_PARAMS)
    for key in (
        "max_depth",
        "n_estimators",
        "learning_rate",
        "min_child_weight",
        "gamma",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "max_delta_step",
    ):
        if key in best.params:
            updated_params[key] = best.params[key]

    config.XGB_PARAMS.update(updated_params)
    config.UMBRAL_CONFIANZA = float(best.params["threshold"])
    config.ATR_SL_MULTIPLIER = float(best.params["sl_mult"])
    config.ATR_TP_MULTIPLIER = float(best.params["tp_mult"])
    config.USE_TIME_FILTER = bool(best.params["use_time_filter"])
    config.HORA_INICIO = int(best.params["hour_start"])
    config.HORA_CIERRE = int(best.params["hour_end"])

    # Reentrenar y guardar el mejor modelo
    best_model = _train_xgb(updated_params, X_train, y_train)
    joblib.dump(best_model, config.MODEL_SAVE_PATH)

    # Persistir en phoenix_config.py
    config_path = Path(__file__).parent / "phoenix_config.py"
    text = config_path.read_text()

    def _replace_block(src, key, new_block):
        start = src.find(key)
        if start == -1:
            return src
        end = src.find("}\n", start)
        if end == -1:
            return src
        return src[:start] + new_block + src[end + 2:]

    xgb_block = "XGB_PARAMS = {\n" + "\n".join([
        f"    \"{k}\": {repr(v)}," for k, v in config.XGB_PARAMS.items()
    ]) + "\n}\n"

    text = _replace_block(text, "XGB_PARAMS = {", xgb_block)
    text = re.sub(
        r"UMBRAL_CONFIANZA\s*=\s*[^\n]+",
        f"UMBRAL_CONFIANZA = {config.UMBRAL_CONFIANZA}",
        text,
    )
    text = re.sub(
        r"ATR_SL_MULTIPLIER\s*=\s*[^\n]+",
        f"ATR_SL_MULTIPLIER = {config.ATR_SL_MULTIPLIER}",
        text,
    )
    text = re.sub(
        r"ATR_TP_MULTIPLIER\s*=\s*[^\n]+",
        f"ATR_TP_MULTIPLIER = {config.ATR_TP_MULTIPLIER}",
        text,
    )
    text = re.sub(
        r"USE_TIME_FILTER\s*=\s*[^\n]+",
        f"USE_TIME_FILTER = {config.USE_TIME_FILTER}",
        text,
    )
    text = re.sub(
        r"HORA_INICIO\s*=\s*[^\n]+",
        f"HORA_INICIO = {config.HORA_INICIO}",
        text,
    )
    text = re.sub(
        r"HORA_CIERRE\s*=\s*[^\n]+",
        f"HORA_CIERRE = {config.HORA_CIERRE}",
        text,
    )
    config_path.write_text(text)


if __name__ == "__main__":
    main()
