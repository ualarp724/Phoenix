#!/usr/bin/env python3
"""
BTC Full Audit
- Train on first 66k rows (same as Optuna split)
- Backtest FULL dataset
- Prints Net Profit, Max DD, Total Trades, Profit Factor
- Memory mindful (frees train arrays ASAP)
"""

import gc
import copy

import joblib
import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from phoenix_backtester_pro import ProfessionalBacktester
from phoenix_brain import preparar_secuencias_flat_con_scaler
from core.mtf import add_mtf_features_multi

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"XGBoost no disponible: {exc}")

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"LightGBM no disponible: {exc}")

from sklearn.preprocessing import StandardScaler

# Winning params (Trial 23)
BEST_PARAMS = {
    "max_depth": 5,
    "n_estimators": 300,
    "learning_rate": 0.061263597364696325,
    "min_child_weight": 22,
    "gamma": 0.18478745296914784,
    "subsample": 0.6920408477000773,
    "colsample_bytree": 0.9136472193306594,
    "reg_alpha": 0.4016246397170869,
    "reg_lambda": 2.759602133276472,
    "max_delta_step": 1,
}


def main() -> None:
    config.apply_asset("BTCUSD")
    config.ENSEMBLE_REQUIRE_CONSENSUS = False

    config.UMBRAL_BUY = 0.21636823317112291
    config.UMBRAL_SELL = 0.23954577564763502
    config.ATR_SL_MULTIPLIER = 2.401460108928972
    config.ATR_TP_MULTIPLIER = 2.0662472497769273

    processor = PhoenixDataProcessor(config.DATA_RAW)
    df_all = add_mtf_features_multi(
        processor.clean_and_prepare(apply_train_filter=False, relax_factor=0.4),
        config.MTF_CONFIGS,
    )
    for col in config.FEATURES:
        if col not in df_all.columns:
            df_all[col] = 0.0

    df_all = df_all.dropna()

    train_df = df_all.iloc[:66000].copy()

    scaler = StandardScaler()
    scaler.fit(train_df[config.FEATURES].values)
    X_train, y_train = preparar_secuencias_flat_con_scaler(train_df, scaler)

    params_xgb = copy.deepcopy(config.XGB_PARAMS)
    params_xgb.update(BEST_PARAMS)
    params_xgb.setdefault("n_jobs", 1)
    params_xgb.setdefault("tree_method", "hist")

    xgb_model = xgb.XGBClassifier(**params_xgb)
    xgb_model.fit(X_train, y_train, verbose=False)

    lgbm_model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
    lgbm_model.fit(X_train, y_train)

    # Free train data
    X_train = None
    y_train = None
    train_df = None
    gc.collect()

    bt = ProfessionalBacktester(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
    bt.model = xgb_model
    bt.model_lgbm = lgbm_model
    bt.scaler = scaler

    full_result = bt.backtest_period(df_all, precomputed_mtf=True)

    if not full_result:
        print("FULL: NO RESULT")
        return

    metrics = full_result["metrics"]
    print(f"Net Profit: ${full_result['profit']:.2f}")
    print(f"Max Drawdown %: {metrics.get('max_drawdown_pct', 0.0):.2f}")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Profit Factor: {metrics.get('profit_factor', 0.0):.2f}")


if __name__ == "__main__":
    main()
