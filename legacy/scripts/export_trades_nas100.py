
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from optuna_nas100_vantage_m15 import (
    _prepare_dataset,
    preparar_secuencias_flat_con_scaler,
    _train_models,
    TRAIN_START,
    TRAIN_END,
    VAL_START,
    VAL_END,
)
from sklearn.preprocessing import StandardScaler
import phoenix_config as config

# Exporta todos los trades de NAS100 M15 (2024-2025) a CSV para simulación conjunta

def main():
    config.apply_asset("NAS100")
    config.TIMEFRAME = "M15"
    config.USE_MTF_CONFIRM = True

    df_all = _prepare_dataset("vantage_nas100.csv").sort_index()
    df_all = df_all.loc[TRAIN_START:VAL_END].copy()
    train_df = df_all.loc[TRAIN_START:TRAIN_END].copy()
    val_df = df_all.loc[VAL_START:VAL_END].copy()

    scaler = StandardScaler()
    scaler.fit(train_df[config.FEATURES].values)
    X_train, y_train = preparar_secuencias_flat_con_scaler(train_df, scaler)

    # Usar los mejores parámetros encontrados en tu backtest
    import json
    with open("nas100_best_dd20.json") as f:
        best_params = json.load(f)
    params = best_params["params"]
    xgb_model, lgbm_model = _train_models(params, config.LGBM_PARAMS, X_train, y_train)

    from optuna_nas100_vantage_m15 import _backtest_multi_position
    result = _backtest_multi_position(
        val_df,
        xgb_model,
        lgbm_model,
        scaler,
        params["umbral_buy"],
        params["umbral_sell"],
        params["sl_mult"],
        params["tp_rr"],
        weekday_only=False,
    )

    # Exportar trades a CSV
    trades = pd.DataFrame(result.trade_pnls, columns=["timestamp", "pnl"])
    trades.to_csv("reports/trades_nas100_2024_2025.csv", index=False)
    print("Exported NAS100 trades to reports/trades_nas100_2024_2025.csv")

if __name__ == "__main__":
    main()
