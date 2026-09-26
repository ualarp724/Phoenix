#!/usr/bin/env python3
"""
Optuna optimization for BTCUSD using fixed 66k/33k split.
Objective: maximize Net Profit with single-position rule.
Filter: BTC relaxed signals (40% more permissive).
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd

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


def _backtest_single_position(
    val_df: pd.DataFrame,
    model_xgb,
    model_lgbm,
    scaler,
    umbral_buy: float,
    umbral_sell: float,
    sl_mult: float,
    tp_mult: float,
) -> float:
    if val_df.empty:
        return 0.0

    data_scaled = scaler.transform(val_df[config.FEATURES].values)
    lookback = config.LOOKBACK_WINDOW
    if len(val_df) <= lookback + 20:
        return 0.0

    windows = np.lib.stride_tricks.sliding_window_view(data_scaled, lookback, axis=0)
    total_windows = windows.shape[0]
    flat = windows.reshape(total_windows, -1)

    probs_xgb = model_xgb.predict_proba(flat)
    probs_lgb = model_lgbm.predict_proba(flat)
    weights = config.ENSEMBLE_WEIGHTS
    combined = (weights["xgboost"] * probs_xgb) + (weights["lightgbm"] * probs_lgb)

    buy_scores = combined[:, 1]
    sell_scores = combined[:, 2]
    preds = np.where(
        buy_scores >= sell_scores,
        np.where(buy_scores >= umbral_buy, 1, 0),
        np.where(sell_scores >= umbral_sell, 2, 0),
    )
    confs = np.where(preds == 1, buy_scores, np.where(preds == 2, sell_scores, 0.0))

    conf_np = confs[confs > 0]
    if conf_np.size == 0:
        conf_np = confs
    dyn_threshold = float(np.quantile(conf_np, config.CONFIDENCE_PERCENTILE))

    capital = config.CAPITAL_INICIAL
    esta_en_operacion = False
    peak = capital
    max_dd = 0.0
    i = 0
    while i < (total_windows - 12):
        actual_idx = i + lookback
        row = val_df.iloc[actual_idx]
        price = row["Close"]

        pred = int(preds[i])
        conf = float(confs[i])
        if pred == 0 or conf <= max(config.UMBRAL_CONFIANZA, dyn_threshold):
            i += 1
            continue

        if esta_en_operacion:
            i += 1
            continue

        atr = val_df["NATR"].iloc[actual_idx] * price / 100
        if atr < config.MIN_ATR_THRESHOLD:
            i += 1
            continue

        sl_dist = atr * sl_mult
        tp_dist = atr * tp_mult

        # Lote dinámico para que TP = $3
        lotes = 3.0 / max(tp_dist, 1e-9)
        lotes = max(config.MIN_LOT_SIZE, min(lotes, config.MAX_LOT_SIZE))

        # Protección de capital: SL > $1.50 (0.75%) -> ignorar
        if sl_dist * lotes > 1.50:
            i += 1
            continue

        if pred == 1:
            tp_price = price + tp_dist
            sl_price = price - sl_dist
        else:
            tp_price = price - tp_dist
            sl_price = price + sl_dist

        pnl = 0.0
        exit_idx = None
        esta_en_operacion = True

        for j in range(1, 13):
            if actual_idx + j >= len(val_df):
                break
            hi = val_df["High"].iloc[actual_idx + j]
            lo = val_df["Low"].iloc[actual_idx + j]
            if pred == 1:
                if lo <= sl_price:
                    pnl = -abs(price - sl_price) * lotes
                    exit_idx = actual_idx + j
                    break
                if hi >= tp_price:
                    pnl = abs(tp_price - price) * lotes
                    exit_idx = actual_idx + j
                    break
            else:
                if hi >= sl_price:
                    pnl = -abs(sl_price - price) * lotes
                    exit_idx = actual_idx + j
                    break
                if lo <= tp_price:
                    pnl = abs(price - tp_price) * lotes
                    exit_idx = actual_idx + j
                    break

        if pnl == 0.0 and actual_idx + 12 < len(val_df):
            exit_idx = actual_idx + 12
            exit_price = val_df["Close"].iloc[exit_idx]
            pnl = (exit_price - price) * lotes if pred == 1 else (price - exit_price) * lotes

        capital += pnl
        if capital > peak:
            peak = capital
        if peak > 0:
            dd = (peak - capital) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

        if exit_idx is None:
            esta_en_operacion = False
            i += 1
        else:
            esta_en_operacion = False
            i = exit_idx + 1

    if max_dd > 10.0:
        return -1e9

    return float(capital - config.CAPITAL_INICIAL)


def main() -> None:
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
    df_all = df_all.dropna()

    if len(df_all) < 90000:
        raise SystemExit("Datos insuficientes para split fijo 66k/33k.")

    train_df = df_all.iloc[:66000].copy()
    val_df = df_all.iloc[66000:99000].copy()

    features = config.FEATURES
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(train_df[features].values)

    X_all, y_all = preparar_secuencias_flat_con_scaler(train_df, scaler)

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
        sl_mult = trial.suggest_float("sl_mult", 1.0, 3.0)
        tp_mult = trial.suggest_float("tp_mult", max(2.0 * sl_mult, 2.0), 6.0)

        xgb_model, lgbm_model = _train_models(params_xgb, params_lgbm, X_all, y_all)

        profit = _backtest_single_position(
            val_df,
            xgb_model,
            lgbm_model,
            scaler,
            umbral_buy,
            umbral_sell,
            sl_mult,
            tp_mult,
        )

        return float(profit)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("\n✅ BEST BTC OPTUNA (FIXED SPLIT)")
    print(study.best_trial.params)
    print(f"Best Profit: {study.best_value}")


if __name__ == "__main__":
    main()
