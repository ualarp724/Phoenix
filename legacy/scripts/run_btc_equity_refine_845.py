import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import matplotlib.pyplot as plt
import phoenix_config as config
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


def main() -> None:
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

    if not result.equity_curve:
        raise SystemExit("No equity data to plot.")

    # --- Reporte de métricas clave ---
    print("\n=== PROFIT TOTAL BTCUSD M15 (2024-2025) ===")
    print(f"Profit: ${result.profit:.2f}")
    print(f"Winrate: {result.win_rate:.2f}%")
    print(f"Max DD: {result.max_drawdown_pct:.2f}%")
    print(f"Trades: {result.total_trades}")

    equity_df = pd.DataFrame(result.equity_curve, columns=["timestamp", "equity"])
    # Exportar a CSV para análisis combinado
    equity_df.to_csv("reports/equity_btc_2024_2025.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["timestamp"], equity_df["equity"], color="#2E86C1", linewidth=2)
    plt.title("Equity Curve BTC (2024-2025) | REFINE_845")
    plt.xlabel("Date")
    plt.ylabel("Equity (USD)")
    plt.grid(True, alpha=0.3)
    out_path = "reports/equity_btc_refine_845_2024_2025.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"PNG: {out_path}")


if __name__ == "__main__":
    main()
