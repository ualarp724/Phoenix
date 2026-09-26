#!/usr/bin/env python3
"""
BTC Debug Script (sin Optuna)
- Entrena modelo rápido con parámetros estándar
- Backtest últimos 6 meses
- Reporta trades, Win Rate, Drawdown y Profit Factor
- Relaja filtro de 'Oportunidades de Oro' en 30%
"""

from __future__ import annotations

import joblib
import pandas as pd

import phoenix_config as config
from phoenix_brain import preparar_secuencias_flat_con_scaler
from phoenix_backtester_pro import ProfessionalBacktester
from core.mtf import add_mtf_features_multi

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"XGBoost no disponible: {exc}")

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"LightGBM no disponible: {exc}")


def _load_and_prepare_btc(
    file_path: str,
    relax_factor: float = 0.30,
    date_start: str | None = None,
    date_end: str | None = None,
) -> pd.DataFrame:
    print("--- [BTC DEBUG] Preparando datos con filtro de oro relajado ---")
    try:
        df = pd.read_csv(file_path, sep="\t")
        if len(df.columns) < 2:
            df = pd.read_csv(file_path, sep=",")
    except Exception as exc:
        raise ValueError(f"Error leyendo CSV: {exc}")

    # Limpieza estándar
    col_map: dict[str, str] = {}
    for col in df.columns:
        c = col.upper().replace("<", "").replace(">", "")
        if "DATE" in c:
            col_map[col] = "Date"
        elif "TIME" in c:
            col_map[col] = "Time"
        elif "OPEN" in c:
            col_map[col] = "Open"
        elif "HIGH" in c:
            col_map[col] = "High"
        elif "LOW" in c:
            col_map[col] = "Low"
        elif "CLOSE" in c:
            col_map[col] = "Close"
        elif "VOL" in c:
            col_map[col] = "Volume"

    df = df.rename(columns=col_map)
    df = df.loc[:, ~df.columns.duplicated()]

    if "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    else:
        df["Datetime"] = pd.to_datetime(df["Date"])
    df.set_index("Datetime", inplace=True)

    if date_start:
        df = df[df.index >= pd.to_datetime(date_start)]
    if date_end:
        df = df[df.index <= pd.to_datetime(date_end)]

    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    # --- Feature Engineering ---
    df["EMA_50"] = df["Close"].ewm(span=50).mean()
    df["Trend_Score"] = (df["Close"] - df["EMA_50"]) / df["Close"] * 1000

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["NATR"] = (tr.rolling(14).mean() / df["Close"]) * 100

    df["Vol_Rel"] = df["Volume"] / df["Volume"].rolling(50).mean()

    sma = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Width"] = (std * 4) / sma
    df["BB_Pos"] = (df["Close"] - (sma - std * 2)) / (std * 4)

    df["Dist_EMA"] = (df["Close"] - df["EMA_50"]) / df["EMA_50"]

    # --- Target relajado (Oportunidades de Oro -30%) ---
    atr = tr.rolling(14).mean()
    base_tp_mult = 2.0
    base_sl_mult = 1.5
    tp_mult = base_tp_mult * (1.0 - relax_factor)
    sl_mult = base_sl_mult * (1.0 - relax_factor)
    tp_dist = atr * tp_mult
    sl_dist = atr * sl_mult

    lookahead = getattr(config, "TARGET_LOOKAHEAD_BARS", None)
    if not lookahead:
        tf = str(getattr(config, "TIMEFRAME", "M15")).upper()
        if tf == "M5":
            lookahead = 48
        elif tf == "H1":
            lookahead = 4
        else:
            lookahead = 16

    future_high = df["High"].rolling(lookahead).max().shift(-lookahead)
    future_low = df["Low"].rolling(lookahead).min().shift(-lookahead)

    df["Target"] = 0

    rsi_buy_base = 75.0
    rsi_sell_base = 25.0
    rsi_buy_relaxed = min(100.0, rsi_buy_base + relax_factor * (100.0 - rsi_buy_base))
    rsi_sell_relaxed = max(0.0, rsi_sell_base - relax_factor * rsi_sell_base)

    tp_buy = df["Close"] + tp_dist
    sl_buy = df["Close"] - sl_dist
    valid_buy = (future_high > tp_buy) & (future_low > sl_buy) & (df["RSI"] < rsi_buy_relaxed)
    df.loc[valid_buy, "Target"] = 1

    tp_sell = df["Close"] - tp_dist
    sl_sell = df["Close"] + sl_dist
    valid_sell = (future_low < tp_sell) & (future_high < sl_sell) & (df["RSI"] > rsi_sell_relaxed)
    df.loc[valid_sell, "Target"] = 2

    df.dropna(inplace=True)

    df = df[
        [
            "RSI",
            "Vol_Rel",
            "Trend_Score",
            "NATR",
            "BB_Width",
            "BB_Pos",
            "Dist_EMA",
            "Target",
            "Close",
            "High",
            "Low",
        ]
    ]

    print(f"   > Velas Totales: {len(df)}")
    print(f"   > Oportunidades de Oro (relajado): {len(df[df['Target'] != 0])}")
    return df


def _print_report(title: str, result: dict | None) -> None:
    print("\n" + title)
    print("-" * len(title))
    if not result:
        print("Sin resultados (0 trades o datos insuficientes).")
        return
    metrics = result.get("metrics", {})
    print(f"Profit: ${result.get('profit', 0.0):.2f}")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Win Rate: {metrics.get('win_rate', 0.0):.2f}%")
    print(f"Profit Factor: {metrics.get('profit_factor', 0.0):.2f}")
    print(f"Max Drawdown %: {metrics.get('max_drawdown_pct', 0.0):.2f}")


def main() -> None:
    config.apply_asset("BTCUSD")
    config.ENSEMBLE_REQUIRE_CONSENSUS = False

    df_all = _load_and_prepare_btc(config.DATA_RAW, relax_factor=0.30)
    df_all = add_mtf_features_multi(df_all.copy(), config.MTF_CONFIGS)
    for col in config.FEATURES:
        if col not in df_all.columns:
            df_all[col] = 0.0
    df_all = df_all.dropna()

    if df_all.empty:
        raise SystemExit("Dataset vacío tras preparación.")

    end_date = df_all.index.max()
    test_4m_start = end_date - pd.DateOffset(months=4)
    train_8m_start = test_4m_start - pd.DateOffset(months=8)

    if train_8m_start < df_all.index.min():
        print("⚠️ Dataset < 12 meses. Usando el inicio disponible para entreno.")
        train_8m_start = df_all.index.min()

    train_df = df_all[(df_all.index >= train_8m_start) & (df_all.index < test_4m_start)].copy()
    test_4m_df = df_all[(df_all.index >= test_4m_start) & (df_all.index <= end_date)].copy()

    backtest_6m_start = end_date - pd.DateOffset(months=6)
    backtest_6m_df = df_all[(df_all.index >= backtest_6m_start) & (df_all.index <= end_date)].copy()

    actual_train_days = (test_4m_start - train_8m_start).days
    if actual_train_days < 240:
        print(f"⚠️ Entreno real < 8 meses: {actual_train_days} días disponibles.")

    print(f"\nRango Entreno (>=8 meses): {train_8m_start.date()} → {(test_4m_start - pd.Timedelta(days=1)).date()}")
    print(f"Rango Test (últimos 4 meses): {test_4m_start.date()} → {end_date.date()}")
    print(f"Rango Backtest solicitado (últimos 6 meses): {backtest_6m_start.date()} → {end_date.date()}")
    print(f"Velas Entreno: {len(train_df)} | Velas Test: {len(test_4m_df)} | Velas Backtest6M: {len(backtest_6m_df)}")

    if len(train_df) < config.LOOKBACK_WINDOW + 100:
        raise SystemExit("Datos de entrenamiento insuficientes tras el split.")

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(train_df[config.FEATURES].values)
    X_train, y_train = preparar_secuencias_flat_con_scaler(train_df, scaler)

    xgb_model = xgb.XGBClassifier(**config.XGB_PARAMS)
    xgb_model.fit(X_train, y_train, verbose=False)

    lgbm_model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
    lgbm_model.fit(X_train, y_train)

    debug_xgb = "phoenix_btc_debug_xgb.pkl"
    debug_lgbm = "phoenix_btc_debug_lgbm.pkl"
    debug_scaler = "phoenix_btc_debug_scaler.pkl"

    joblib.dump(xgb_model, debug_xgb)
    joblib.dump(lgbm_model, debug_lgbm)
    joblib.dump(scaler, debug_scaler)

    prev_lgbm_path = config.LGBM_MODEL_PATH
    try:
        config.LGBM_MODEL_PATH = debug_lgbm
        backtester = ProfessionalBacktester(debug_xgb, debug_scaler)
        backtester.model = xgb_model
        backtester.model_lgbm = lgbm_model
        backtester.scaler = scaler

        result_6m = backtester.backtest_period(backtest_6m_df, precomputed_mtf=True)
        result_4m = backtester.backtest_period(test_4m_df, precomputed_mtf=True)
    finally:
        config.LGBM_MODEL_PATH = prev_lgbm_path

    _print_report("BTC DEBUG · BACKTEST ÚLTIMOS 6 MESES", result_6m)
    _print_report("BTC DEBUG · TEST ÚLTIMOS 4 MESES", result_4m)

    print("\nNota: filtro de 'Oportunidades de Oro' relajado un 30% (TP/SL y RSI).")
    print("Si Win Rate ~48% pero Profit Factor > 1, el límite de 52% era demasiado alto.")


if __name__ == "__main__":
    main()
