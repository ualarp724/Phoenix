#!/usr/bin/env python3
"""
Benchmark XGBoost vs LightGBM vs CatBoost.
Runs Test 1 (VAL) first, stores failures, and then runs Test 2.
Skips models that are not installed. Does not overwrite config.
"""

import copy
from datetime import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from phoenix_backtester_pro import ProfessionalBacktester
from phoenix_brain import preparar_secuencias_flat_con_scaler
from core.mtf import add_mtf_features_multi


def _days_between(start: str, end: str) -> int:
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    return max((end_dt.date() - start_dt.date()).days + 1, 1)


def _load_train_df():
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df_all = add_mtf_features_multi(
        processor.clean_and_prepare(apply_train_filter=False),
        config.MTF_CONFIGS,
    )
    return df_all[(df_all.index >= config.TRAIN_START_DATE) & (df_all.index <= config.TRAIN_END_DATE)]


def _load_data(date_start: str, date_end: str, warmup_days: int = 60):
    start_dt = datetime.fromisoformat(date_start)
    end_dt = datetime.fromisoformat(date_end)
    warmup_start = (start_dt - pd.Timedelta(days=warmup_days)).strftime('%Y-%m-%d')

    processor = PhoenixDataProcessor(config.DATA_RAW)
    df_all = add_mtf_features_multi(
        processor.clean_and_prepare(date_start=warmup_start, date_end=date_end, apply_train_filter=False),
        config.MTF_CONFIGS,
    )

    train_df = _load_train_df()
    test_df = df_all[(df_all.index >= date_start) & (df_all.index <= date_end)]

    if len(train_df) < config.LOOKBACK_WINDOW + 100:
        raise SystemExit("Datos de entrenamiento insuficientes")
    if len(test_df) < config.LOOKBACK_WINDOW + 100:
        raise SystemExit("Datos insuficientes para el rango solicitado")

    features = config.FEATURES
    scaler = joblib.load(config.SCALER_SAVE_PATH) if Path(config.SCALER_SAVE_PATH).exists() else None
    if scaler is None or getattr(scaler, "n_features_in_", None) != len(features):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_df[features].values)
        joblib.dump(scaler, config.SCALER_SAVE_PATH)

    X_train, y_train = preparar_secuencias_flat_con_scaler(train_df, scaler)
    return X_train, y_train, test_df, scaler


def _evaluate_model(model, backtester, test_df, days):
    backtester.model = model
    backtester.model_lgbm = None
    result = backtester.backtest_period(test_df, precomputed_mtf=True)
    if not result:
        return None
    metrics = result["metrics"]
    profit = result["profit"]
    return {
        "profit": profit,
        "avg_daily": profit / days,
        "win_rate": metrics.get("win_rate", 0.0),
        "profit_factor": metrics.get("profit_factor", 0.0),
        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
        "avg_daily_profit": metrics.get("avg_daily_profit", 0.0),
        "trades": metrics.get("total_trades", 0)
    }


def main():
    X_train, y_train, val_df_bt, scaler = _load_data(config.VAL_START_DATE, config.VAL_END_DATE)
    days = _days_between(config.VAL_START_DATE, config.VAL_END_DATE)

    backtester = ProfessionalBacktester(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
    backtester.scaler = scaler

    results = {}
    failures = {}

    try:
        import xgboost as xgb
        xgb_params = copy.deepcopy(config.XGB_PARAMS)
        xgb_params.setdefault("tree_method", "hist")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train, verbose=False)
        results["xgboost"] = _evaluate_model(model, backtester, val_df_bt, days)
    except Exception as exc:
        results["xgboost"] = f"XGBoost error: {exc}"

    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)
        results["lightgbm"] = _evaluate_model(model, backtester, val_df_bt, days)
    except Exception as exc:
        results["lightgbm"] = f"LightGBM no disponible: {exc}"

    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=8,
            loss_function="MultiClass",
            verbose=False,
            random_seed=42,
        )
        model.fit(X_train, y_train)
        results["catboost"] = _evaluate_model(model, backtester, val_df_bt, days)
    except Exception as exc:
        results["catboost"] = f"CatBoost no disponible: {exc}"

    print("\n📊 BENCHMARK MODELOS (TEST 1 / VAL)")
    for name, res in results.items():
        print(f"\n{name}")
        print(res)
        if not isinstance(res, dict):
            failures[name] = {"reason": "error", "detail": res}
            continue
        if res["profit"] <= 0:
            failures[name] = {"reason": "no_profit", "detail": res}

    failure_path = Path(__file__).parent / "benchmark_failures.json"
    failure_path.write_text(json.dumps(failures, indent=2))
    print(f"\n🧠 Fallos guardados en {failure_path}")

    # Ejecutar Test 2 solo con modelos que pasaron Test 1
    if failures:
        print("\n⚠️ Modelos fallidos en Test 1 se omiten en Test 2:")
        for name in failures:
            print(f"- {name}")

    X_train, y_train, test2_df_bt, scaler = _load_data(config.TEST_START_DATE, config.TEST_END_DATE)
    days_test2 = _days_between(config.TEST_START_DATE, config.TEST_END_DATE)
    backtester.scaler = scaler

    print("\n📊 BENCHMARK MODELOS (TEST 2)")
    for name, res in results.items():
        if name in failures:
            continue
        try:
            if name == "xgboost":
                import xgboost as xgb
                xgb_params = copy.deepcopy(config.XGB_PARAMS)
                xgb_params.setdefault("tree_method", "hist")
                model = xgb.XGBClassifier(**xgb_params)
                model.fit(X_train, y_train, verbose=False)
            elif name == "lightgbm":
                import lightgbm as lgb
                model = lgb.LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=-1,
                    num_leaves=64,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                )
                model.fit(X_train, y_train)
            elif name == "catboost":
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(
                    iterations=400,
                    learning_rate=0.05,
                    depth=8,
                    loss_function="MultiClass",
                    verbose=False,
                    random_seed=42,
                )
                model.fit(X_train, y_train)
            else:
                continue
            test2_result = _evaluate_model(model, backtester, test2_df_bt, days_test2)
            print(f"\n{name}")
            print(test2_result)
        except Exception as exc:
            print(f"\n{name}\nTest 2 error: {exc}")


if __name__ == "__main__":
    main()
