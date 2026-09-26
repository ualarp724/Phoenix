
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from optuna_btc_vantage2_m15 import (
    REFINE_845_PARAMS,
    _prepare_dataset,
    preparar_secuencias_flat_con_scaler,
    _train_models,
    _backtest_multi_position,
    TRAIN_START,
    TRAIN_END,
    VAL_START,
    VAL_END,
)
from sklearn.preprocessing import StandardScaler
import phoenix_config as config

# Exporta todos los trades de BTC M15 (2024-2025) a CSV para simulación conjunta

def main():
    config.apply_asset("BTCUSD")
    config.TIMEFRAME = "M15"
    config.USE_MTF_CONFIRM = True

    df_all = _prepare_dataset("vantage_btc2.csv").sort_index()
    df_all = df_all.loc[TRAIN_START:VAL_END].copy()
    train_df = df_all.loc[TRAIN_START:TRAIN_END].copy()
    val_df = df_all.loc[VAL_START:VAL_END].copy()

    scaler = StandardScaler()
    scaler.fit(train_df[config.FEATURES].values)
    X_train, y_train = preparar_secuencias_flat_con_scaler(train_df, scaler)

    xgb_model, lgbm_model = _train_models(REFINE_845_PARAMS, config.LGBM_PARAMS, X_train, y_train)

    result = _backtest_multi_position(
        val_df,
        xgb_model,
        lgbm_model,
        scaler,
        REFINE_845_PARAMS["umbral_buy"],
        REFINE_845_PARAMS["umbral_sell"],
        REFINE_845_PARAMS["sl_mult"],
        REFINE_845_PARAMS["tp_rr"],
        weekday_only=True,
    )

    # Exportar trades a CSV
    trades = pd.DataFrame(result.trade_pnls, columns=["timestamp", "pnl"])
    trades.to_csv("reports/trades_btc_2024_2025.csv", index=False)
    print("Exported BTC trades to reports/trades_btc_2024_2025.csv")

if __name__ == "__main__":
    main()
