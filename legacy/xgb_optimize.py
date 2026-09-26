#!/usr/bin/env python3
"""
XGBoost optimizer for Phoenix bot.
Trains multiple XGB configs on TRAIN window and evaluates on VAL window
using the same backtester logic. Updates phoenix_config.py with best params.
"""

import copy
from datetime import datetime
import os
from pathlib import Path
import re
import random

import joblib
import numpy as np

import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from phoenix_backtester_pro import ProfessionalBacktester
from phoenix_brain import preparar_secuencias_flat_con_scaler
from core.mtf import add_mtf_features_multi

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


def main():
    if config.MODEL_TYPE not in {"xgboost", "ensemble"}:
        raise SystemExit("MODEL_TYPE debe ser xgboost o ensemble")

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

    # Fit scaler only on train
    features = config.FEATURES
    scaler = joblib.load(config.SCALER_SAVE_PATH) if Path(config.SCALER_SAVE_PATH).exists() else None
    if scaler is None or getattr(scaler, "n_features_in_", None) != len(features):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_df[features].values)
        joblib.dump(scaler, config.SCALER_SAVE_PATH)

    X_train, y_train = preparar_secuencias_flat_con_scaler(train_df, scaler)
    X_val, y_val = preparar_secuencias_flat_con_scaler(val_df, scaler)

    # Candidate params (random search)
    base = copy.deepcopy(config.XGB_PARAMS)
    random.seed(42)
    candidates = []
    for _ in range(60):
        params = copy.deepcopy(base)
        params.update(
            {
                "max_depth": random.choice([4, 5, 6, 7, 8]),
                "n_estimators": random.choice([300, 400, 500]),
                "learning_rate": random.choice([0.03, 0.05, 0.08]),
                "min_child_weight": random.choice([1, 5, 10, 20]),
                "gamma": random.choice([0, 0.05, 0.1, 0.2, 0.3]),
                "subsample": random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
                "colsample_bytree": random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
                "reg_alpha": random.choice([0.0, 0.01, 0.05, 0.1, 0.3]),
                "reg_lambda": random.choice([1.0, 1.5, 2.0, 3.0]),
                "max_delta_step": random.choice([0, 1]),
            }
        )
        candidates.append(params)

    thresholds = [0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28]
    sl_mults = [0.8, 1.0, 1.2, 1.4]
    tp_mults = [1.4, 1.6, 1.8, 2.0]

    best = None
    best_score = -1e9
    days = _days_between(config.VAL_START_DATE, config.VAL_END_DATE)

    # Ensure MTF features are present for backtest filters
    val_df_bt = add_mtf_features_multi(val_df.copy(), config.MTF_CONFIGS)

    for idx, params in enumerate(candidates, 1):
        model = _train_xgb(params, X_train, y_train)
        joblib.dump(model, config.MODEL_SAVE_PATH)
        backtester = ProfessionalBacktester(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
        backtester.model_lgbm = None

        for th in thresholds:
            for sl_mult in sl_mults:
                for tp_mult in tp_mults:
                    result = backtester.backtest_period(
                        val_df_bt,
                        umbral=th,
                        sl_mult=sl_mult,
                        tp_mult=tp_mult,
                        precomputed_mtf=True,
                    )
                    if not result:
                        continue
                    metrics = result["metrics"]
                    profit = result["profit"]
                    avg_daily = profit / days
                    win_rate = metrics.get("win_rate", 0.0)
                    profit_factor = metrics.get("profit_factor", 0.0)
                    max_dd = metrics.get("max_drawdown_pct", 0.0)
                    score = avg_daily * 10 + (win_rate / 100) * 2 + profit_factor - (max_dd * 0.1)

                    if score > best_score:
                        best_score = score
                        best = {
                            "params": params,
                            "threshold": th,
                            "sl_mult": sl_mult,
                            "tp_mult": tp_mult,
                            "profit": profit,
                            "avg_daily": avg_daily,
                            "win_rate": win_rate,
                            "profit_factor": profit_factor,
                            "max_drawdown_pct": max_dd,
                        }

        print(f"[{idx}/{len(candidates)}] params probado")

    if not best:
        raise SystemExit("No se encontró configuración válida")

    print("\n✅ MEJOR CONFIGURACIÓN")
    print(best)

    # Reentrenar y guardar mejor modelo
    best_model = _train_xgb(best["params"], X_train, y_train)
    joblib.dump(best_model, config.MODEL_SAVE_PATH)

    # Update config
    config.XGB_PARAMS.update(best["params"])
    config.UMBRAL_CONFIANZA = float(best["threshold"])
    config.ATR_SL_MULTIPLIER = float(best["sl_mult"])
    config.ATR_TP_MULTIPLIER = float(best["tp_mult"])

    # Persist config updates
    config_path = Path(__file__).parent / "phoenix_config.py"
    text = config_path.read_text()
    text = text.replace("UMBRAL_CONFIANZA = ", "UMBRAL_CONFIANZA = ")

    # Simple replacement for XGB_PARAMS and UMBRAL
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

    # Replace UMBRAL
    text = re.sub(
        r"UMBRAL_CONFIANZA\s*=\s*[^\n]+",
        f"UMBRAL_CONFIANZA = {best['threshold']}",
        text,
    )
    text = re.sub(
        r"ATR_SL_MULTIPLIER\s*=\s*[^\n]+",
        f"ATR_SL_MULTIPLIER = {best['sl_mult']}",
        text,
    )
    text = re.sub(
        r"ATR_TP_MULTIPLIER\s*=\s*[^\n]+",
        f"ATR_TP_MULTIPLIER = {best['tp_mult']}",
        text,
    )
    text = _replace_block(text, "XGB_PARAMS = {", xgb_block)
    config_path.write_text(text)


if __name__ == "__main__":
    main()
