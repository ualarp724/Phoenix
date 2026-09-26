#!/usr/bin/env python3
"""
Optuna optimization for BTCUSD with TimeSeriesSplit (5 folds).
Objective: maximize Net Profit with penalty if max DD > 25%.
"""

import argparse
import copy
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import preparar_secuencias_flat_con_scaler
from phoenix_backtester_pro import ProfessionalBacktester
from core.mtf import add_mtf_features_multi

try:
    import optuna
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Optuna no disponible: {exc}")

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"XGBoost no disponible: {exc}")

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"LightGBM no disponible: {exc}")


def _train_models(params_xgb, params_lgbm, X_train, y_train):
    tuned_xgb = copy.deepcopy(params_xgb)
    tuned_xgb.setdefault("n_jobs", max(1, os.cpu_count() or 1))
    tuned_xgb.setdefault("tree_method", "hist")
    xgb_model = xgb.XGBClassifier(**tuned_xgb)
    xgb_model.fit(X_train, y_train, verbose=False)

    lgbm_model = lgb.LGBMClassifier(**params_lgbm)
    lgbm_model.fit(X_train, y_train)
    return xgb_model, lgbm_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    config.apply_asset("BTCUSD")
    config.ENSEMBLE_REQUIRE_CONSENSUS = False

    processor = PhoenixDataProcessor(config.DATA_RAW)
    df_all = add_mtf_features_multi(
        processor.clean_and_prepare(apply_train_filter=False, relax_factor=0.4),
        config.MTF_CONFIGS,
    )
    for col in config.FEATURES:
        if col not in df_all.columns:
            df_all[col] = 0.0

    train_df = df_all[(df_all.index >= config.TRAIN_START_DATE) & (df_all.index <= config.TRAIN_END_DATE)]
    if len(train_df) < config.LOOKBACK_WINDOW + 100:
        raise SystemExit("Datos de entrenamiento insuficientes")

    features = config.FEATURES
    scaler = joblib.load(config.SCALER_SAVE_PATH) if Path(config.SCALER_SAVE_PATH).exists() else None
    if scaler is None or getattr(scaler, "n_features_in_", None) != len(features):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_df[features].values)
        joblib.dump(scaler, config.SCALER_SAVE_PATH)

    X_all, y_all = preparar_secuencias_flat_con_scaler(train_df, scaler)

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial: optuna.Trial):
        params_xgb = copy.deepcopy(config.XGB_PARAMS)
        params_xgb.update(
            {
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 25),
                "gamma": trial.suggest_float("gamma", 0.0, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 3.0),
                "max_delta_step": trial.suggest_int("max_delta_step", 0, 1),
            }
        )

        params_lgbm = copy.deepcopy(config.LGBM_PARAMS)

        umbral_buy = trial.suggest_float("umbral_buy", 0.08, 0.28)
        umbral_sell = trial.suggest_float("umbral_sell", 0.08, 0.28)
        sl_mult = trial.suggest_float("sl_mult", 1.8, 3.0)
        tp_mult = trial.suggest_float("tp_mult", 1.0, 2.2)

        profit_scores = []
        max_dds = []

        for train_idx, val_idx in tscv.split(X_all):
            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_val = X_all[val_idx]
            y_val = y_all[val_idx]

            xgb_model, lgbm_model = _train_models(params_xgb, params_lgbm, X_train, y_train)

            bt = ProfessionalBacktester(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
            bt.model = xgb_model
            bt.model_lgbm = lgbm_model

            # Build val df aligned with val indices
            val_df = train_df.iloc[val_idx + config.LOOKBACK_WINDOW].copy()
            val_df = add_mtf_features_multi(val_df.copy(), config.MTF_CONFIGS)
            val_df = val_df.dropna()

            prev_buy = config.UMBRAL_BUY
            prev_sell = config.UMBRAL_SELL
            prev_sl = config.ATR_SL_MULTIPLIER
            prev_tp = config.ATR_TP_MULTIPLIER
            config.UMBRAL_BUY = float(umbral_buy)
            config.UMBRAL_SELL = float(umbral_sell)

            result = bt.backtest_period(
                val_df,
                precomputed_mtf=True,
                sl_mult=sl_mult,
                tp_mult=tp_mult,
            )

            config.UMBRAL_BUY = prev_buy
            config.UMBRAL_SELL = prev_sell
            config.ATR_SL_MULTIPLIER = prev_sl
            config.ATR_TP_MULTIPLIER = prev_tp

            if not result:
                return 0.0

            metrics = result['metrics']
            profit_scores.append(result.get("profit", 0.0))
            max_dds.append(metrics.get('max_drawdown_pct', 0.0))
        max_dd = max(max_dds) if max_dds else 0.0
        if max_dd > 25.0:
            return -1e9

        total_profit = float(np.sum(profit_scores)) if profit_scores else 0.0
        score = total_profit - (max_dd / 100.0) * 0.5
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("\n✅ BEST BTC OPTUNA")
    print(study.best_trial.params)
    print(f"Best PF: {study.best_value}")


if __name__ == "__main__":
    main()
